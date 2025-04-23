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

#include "vx_ext_opencv.h"
#include "internal_publishKernels.h"

/**********************************************************************
  PUBLIC FUNCTION for OpenVX/OpenCV user defined functions
  **********************************************************************/
extern "C"  SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
    vx_status status = VX_SUCCESS;

    STATUS_ERROR_CHECK(get_kernels_to_publish());
    STATUS_ERROR_CHECK(Kernel_List->PUBLISH(context));

    return status;
}

/************************************************************************************************************
Add All Kernels to the Kernel List
*************************************************************************************************************/
vx_status get_kernels_to_publish()
{
    vx_status status = VX_SUCCESS;

    Kernel_List = new Kernellist(MAX_KERNELS);

    STATUS_ERROR_CHECK(ADD_KERENEL(CV_absdiff_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_adaptiveThreshold_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_add_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_AddWeighted_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_bilateralFilter_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_bitwise_and_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_bitwise_not_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_bitwise_or_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_bitwise_xor_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_blur_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_Boxfilter_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_brisk_compute_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_brisk_detect_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_buildOpticalFlowPyramid_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_buildPyramid_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_Canny_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_compare_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_convertScaleAbs_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_cornerHarris_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_cornerMinEigenVal_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_countNonZero_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_cvtColor_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_dilate_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_distanceTransform_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_divide_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_erode_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_FAST_detector_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_fastNlMeansDenoising_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_fastNlMeansDenoisingColored_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_filter2D_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_flip_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_Gaussianblur_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_good_features_to_track_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_integral_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_Laplacian_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_MedianBlur_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_morphologyEx_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_MSER_detect_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_multiply_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_norm_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_ORB_compute_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_ORB_detect_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_pyrdown_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_pyrup_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_resize_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_Scharr_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_sepFilter2D_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_simple_blob_detect_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_simple_blob_detect_initialize_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_Sobel_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_subtract_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_threshold_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_transpose_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_warpAffine_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_warpPerspective_Register));

#if USE_OPENCV_CONTRIB
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_SIFT_compute_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_SIFT_detect_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_star_detect_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_SURF_compute_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CV_SURF_detect_Register));
#endif

    return status;
}

/************************************************************************************************************
Add Kernels to the Kernel List
*************************************************************************************************************/
vx_status ADD_KERENEL(std::function<vx_status(vx_context)> func)
{
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(Kernel_List->ADD(func));
    return status;
}
