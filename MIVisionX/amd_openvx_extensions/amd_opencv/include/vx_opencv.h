/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _VX_EXT_AMD_CV_H_
#define _VX_EXT_AMD_CV_H_

#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief The AMD extension library for OpenCV */
#define VX_LIBRARY_OPENCV         1

/*!
 * \brief The list of available vision kernels in the OpenCV extension library.
 */
enum vx_kernel_ext_amd_cv_e
{
    /*!
     * \brief The OpenCV blur function kernel. Kernel name is "org.opencv.blur".
     */
    VX_KERNEL_OPENCV_BLUR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x100,

    /*!
     * \brief The OpenCV medianBlur function kernel. Kernel name is "org.opencv.medianblur".
     */
    VX_KERNEL_OPENCV_MEDIAN_BLUR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x101,

    /*!
     * \brief The OpenCV GaussianBlur function kernel. Kernel name is "org.opencv.gaussianblur".
     */
    VX_KERNEL_OPENCV_GAUSSIAN_BLUR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x102,

    /*!
     * \brief The OpenCV boxFilter function kernel. Kernel name is "org.opencv.boxfilter".
     */
    VX_KERNEL_OPENCV_BOXFILTER = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x103,

    /*!
     * \brief The OpenCV BilateralFilter function kernel. Kernel name is "org.opencv.bilateralfilter".
     */
    VX_KERNEL_OPENCV_BILATERAL_FILTER = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x104,

    /*!
     * \brief The OpenCV Flip function kernel. Kernel name is "org.opencv.flip".
     */
    VX_KERNEL_OPENCV_FLIP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x37,

    /*!
     * \brief The OpenCV transpose function kernel. Kernel name is "org.opencv.transpose".
     */
    VX_KERNEL_OPENCV_TRANSPOSE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x39,

    /*!
     * \brief The OpenCV absdiff function kernel. Kernel name is "org.opencv.absdiff".
     */
    VX_KERNEL_OPENCV_ABSDIFF = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x40,

    /*!
     * \brief The OpenCV add function kernel. Kernel name is "org.opencv.add".
     */
    VX_KERNEL_OPENCV_ADD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x41,

    /*!
     * \brief The OpenCV bitwise_and function kernel. Kernel name is "org.opencv.bitwise_and".
     */
    VX_KERNEL_OPENCV_BITWISE_AND = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x42,

    /*!
     * \brief The OpenCV bitwise_not function kernel. Kernel name is "org.opencv.flip".
     */
    VX_KERNEL_OPENCV_BITWISE_NOT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x43,

    /*!
    * \brief The OpenCV bitwise_or function kernel. Kernel name is "org.opencv.bitwise_or".
    */
    VX_KERNEL_OPENCV_BITWISE_OR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x44,

    /*!
    * \brief The OpenCV bitwise_xor function kernel. Kernel name is "org.opencv.bitwise_xor".
    */
    VX_KERNEL_OPENCV_BITWISE_XOR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x45,

    /*!
    * \brief The OpenCV subtract function kernel. Kernel name is "org.opencv.subtract".
    */
    VX_KERNEL_OPENCV_SUBTRACT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x46,

    /*!
    * \brief The OpenCV compare function kernel. Kernel name is "org.opencv.compare".
    */
    VX_KERNEL_OPENCV_COMPARE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x47,

    /*!
    * \brief The OpenCV SOBEL function kernel. Kernel name is "org.opencv.sobel".
    */
    VX_KERNEL_OPENCV_SOBEL = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x30,

    /*!
    * \brief The OpenCV CONVERTSCALEABS function kernel. Kernel name is "org.opencv.convertscaleabs".
    */
    VX_KERNEL_OPENCV_CONVERTSCALEABS = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x31,

    /*!
    * \brief The OpenCV ADDWEIGHTED function kernel. Kernel name is "org.opencv.addweighted".
    */
    VX_KERNEL_OPENCV_ADDWEIGHTED = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x32,

    /*!
    * \brief The OpenCV CANNY function kernel. Kernel name is "org.opencv.canny".
    */
    VX_KERNEL_OPENCV_CANNY = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x33,

    /*!
    * \brief The OpenCV LAPLACIAN function kernel. Kernel name is "org.opencv.laplacian".
    */
    VX_KERNEL_OPENCV_LAPLACIAN = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x34,

    /*!
    * \brief The OpenCV MORPHOLOGYEX function kernel. Kernel name is "org.opencv.morphologyex".
    */
    VX_KERNEL_OPENCV_MORPHOLOGYEX = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x35,

    /*!
    * \brief The OpenCV SCHARR function kernel. Kernel name is "org.opencv.scharr".
    */
    VX_KERNEL_OPENCV_SCHARR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x36,

    /*!
    * \brief The OpenCV FAST feature detector function kernel. Kernel name is "org.opencv.fast".
    */
    VX_KERNEL_OPENCV_FAST = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x12,

    /*!
    * \brief The OpenCV GoodFeaturesToTrack (GFTT) detector function kernel. Kernel name is "org.opencv.good_features_to_track".
    */
    VX_KERNEL_OPENCV_GOOD_FEATURE_TO_TRACK = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x13,

    /*!
    * \brief The OpenCV SIFT detector function kernel.Kernel name is "org.opencv.sift_detect".
    */
    VX_KERNEL_OPENCV_SIFT_DETECT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x10,

    /*!
    * \brief The OpenCV SURF detector function kernel. Kernel name is "org.opencv.surf_detect".
    */
    VX_KERNEL_OPENCV_SURF_DETECT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x11,

    /*!
    * \brief The OpenCV BRISK detector function kernel. Kernel name is "org.opencv.brisk_detect".
    */
    VX_KERNEL_OPENCV_BRISK_DETECT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x14,

    /*!
    * \brief The OpenCV MSER feature detector function kernel. Kernel name is "org.opencv.mser_detect".
    */
    VX_KERNEL_OPENCV_MSER_DETECT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x16,

    /*!
    * \brief The OpenCV ORB detector function kernel. Kernel name is "org.opencv.orb_detect".
    */
    VX_KERNEL_OPENCV_ORB_DETECT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x17,

    /*!
    * \brief The OpenCV Simple Blob detector function kernel. Kernel name is "org.opencv.simple_blob_detect".
    */
    VX_KERNEL_OPENCV_SIMPLE_BLOB_DETECT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x18,

    /*!
    * \brief The OpenCV simple_blob_detector_initialize function kernel. Kernel name is "org.opencv.simple_blob_detector_initialize".
    */
    VX_KERNEL_OPENCV_SIMPLE_BLOB_DETECT_INITIALIZE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x19,

    /*!
    * \brief The OpenCV STAR feature detector function kernel. Kernel name is "org.opencv.star_detect".
    */
    VX_KERNEL_OPENCV_STAR_FEATURE_DETECT = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x20,

    /*!
    * \brief The OpenCV SIFT descriptor function kernel. Kernel name is "org.opencv.sift_compute".
    */
    VX_KERNEL_OPENCV_SIFT_COMPUTE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x21,

    /*!
    * \brief The OpenCV SURF descriptor function kernel. Kernel name is "org.opencv.surf_compute".
    */
    VX_KERNEL_OPENCV_SURF_COMPUTE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x22,

    /*!
    * \brief The OpenCV BRISK descriptor function kernel. Kernel name is "org.opencv.brisk_compute".
    */
    VX_KERNEL_OPENCV_BRISK_COMPUTE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x23,

    /*!
    * \brief The OpenCV ORB descriptor function kernel. Kernel name is "org.opencv.orb_compute".
    */
    VX_KERNEL_OPENCV_ORB_COMPUTE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x24,

    /*!
    * \brief The OpenCV MULTIPLY function kernel. Kernel name is "org.opencv.multiply".
    */
    VX_KERNEL_OPENCV_MULTIPLY = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x51,

    /*!
    * \brief The OpenCV Divide function kernel. Kernel name is "org.opencv.divide".
    */
    VX_KERNEL_OPENCV_DIVIDE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x52,

    /*!
    * \brief The OpenCV ADAPTIVETHRESHOLD function kernel. Kernel name is "org.opencv.adaptivethreshold".
    */
    VX_KERNEL_OPENCV_ADAPTIVETHRESHOLD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x53,

    /*!
    * \brief The OpenCV DISTANCETRANSFORM function kernel. Kernel name is "org.opencv.distancetransform".
    */
    VX_KERNEL_OPENCV_DISTANCETRANSFORM = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x54,

    /*!
    * \brief The OpenCV cvtcolor function kernel. Kernel name is "org.opencv.cvtcolor".
    */
    VX_KERNEL_OPENCV_CVTCOLOR = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x55,

    /*!
    * \brief The OpenCV Threshold function kernel. Kernel name is "org.opencv.threshold".
    */
    VX_KERNEL_OPENCV_THRESHOLD = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x56,

    /*!
    * \brief The OpenCV fastNlMeansDenoising function kernel. Kernel name is "org.opencv.fastnlmeansdenoising".
    */
    VX_KERNEL_OPENCV_FAST_NL_MEANS_DENOISING = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x57,

    /*!
    * \brief The OpenCV fastNlMeansDenoising Colored function kernel. Kernel name is "org.opencv.fastnlmeansdenoisingcolored".
    */
    VX_KERNEL_OPENCV_FAST_NL_MEANS_DENOISING_COLORED = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x58,

    /*!
    * \brief The OpenCV pyrup function kernel. Kernel name is "org.opencv.pyrup".
    */
    VX_KERNEL_OPENCV_PYRUP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x59,

    /*!
    * \brief The OpenCV pyrdown function kernel. Kernel name is "org.opencv.pyrdown".
    */
    VX_KERNEL_OPENCV_PYRDOWN = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x60,

    /*!
    * \brief The OpenCV filter2D function kernel. Kernel name is "org.opencv.filter2D".
    */
    VX_KERNEL_OPENCV_FILTER_2D = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x61,

    /*!
    * \brief The OpenCV sepFilter2D function kernel. Kernel name is "org.opencv.sepFilter2D".
    */
    VX_KERNEL_OPENCV_SEPFILTER_2D = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x62,

    /*!
    * \brief The OpenCV dilate function kernel. Kernel name is "org.opencv.dilate".
    */
    VX_KERNEL_OPENCV_DILATE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x63,

    /*!
    * \brief The OpenCV erode function kernel. Kernel name is "org.opencv.erode".
    */
    VX_KERNEL_OPENCV_ERODE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x64,

    /*!
    * \brief The OpenCV warpAffine function kernel. Kernel name is "org.opencv.warpAffine".
    */
    VX_KERNEL_OPENCV_WARP_AFFINE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x65,

    /*!
    * \brief The OpenCV warpPerspective function kernel. Kernel name is "org.opencv.warpPerspective".
    */
    VX_KERNEL_OPENCV_WARP_PERSPECTIVE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x66,

    /*!
    * \brief The OpenCV resize function kernel. Kernel name is "org.opencv.resize".
    */
    VX_KERNEL_OPENCV_RESIZE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x67,

    /*!
    * \brief The OpenCV buildPyramid function kernel. Kernel name is "org.opencv.buildPyramid".
    */
    VX_KERNEL_OPENCV_BUILD_PYRAMID = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x68,

    /*!
    * \brief The OpenCV Flip function kernel. Kernel name is "org.opencv.Flip".
    */
    VX_KERNEL_OPENCV_BUILD_OPTICAL_FLOW_PYRAMID = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x69,

    /*!
    * \brief The OpenCV integral function kernel. Kernel name is "org.opencv.integral".
    */
    VX_KERNEL_OPENCV_INTEGRAL = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x70,

    /*!
    * \brief The OpenCV countNonZero function kernel. Kernel name is "org.opencv.countnonzero".
    */
    VX_KERNEL_OPENCV_COUNT_NON_ZERO = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x71,

    /*!
    * \brief The OpenCV norm function kernel. Kernel name is "org.opencv.norm".
    */
    VX_KERNEL_OPENCV_NORM = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x72,

    /*!
    * \brief The OpenCV CORNERHARRIS function kernel. Kernel name is "org.opencv.cornerharris".
    */
    VX_KERNEL_OPENCV_CORNERHARRIS = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x201,

    /*!
    * \brief The OpenCV cornerMinEigenVal function kernel. Kernel name is "org.opencv.cornermineigenVal".
    */
    VX_KERNEL_OPENCV_CORNER_MIN_EIGEN_VAL = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_OPENCV) + 0x202,

};

#ifdef  __cplusplus
}
#endif

#endif
