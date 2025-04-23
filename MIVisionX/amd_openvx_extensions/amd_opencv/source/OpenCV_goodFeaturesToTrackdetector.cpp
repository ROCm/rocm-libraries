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

#include "internal_publishKernels.h"

/*!***********************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_good_feature_detector_InputValidator(vx_node node, vx_uint32 index)
{

    vx_status status = VX_SUCCESS;

    vx_parameter param = vxGetParameterByIndex(node, index);

    if (index == 0)
    {
        vx_image image;
        vx_df_image df_image = VX_DF_IMAGE_VIRT;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
        if (df_image != VX_DF_IMAGE_U8)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseImage(&image);
    }

    else if (index == 1)
    {
        vx_array array;
        vx_size size = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &array, sizeof(array)));
        STATUS_ERROR_CHECK(vxQueryArray(array, VX_ARRAY_ATTRIBUTE_CAPACITY, &size, sizeof(size)));
        vxReleaseArray(&array);
    }

    else if (index == 2)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_int32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_INT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 3)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_float32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_FLOAT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 4)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_float32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_FLOAT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    if (index == 5)
    {
        vx_image image;
        vx_df_image df_image = VX_DF_IMAGE_VIRT;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
        if (df_image != VX_DF_IMAGE_U8)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseImage(&image);
    }

    else if (index == 6)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_int32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_INT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 7)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_bool value = vx_true_e;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if ((value != vx_true_e && value != vx_false_e) || type != VX_TYPE_BOOL)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 8)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_float32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_FLOAT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    vxReleaseParameter(&param);
    return status;
}

/*!***********************************************************************************************************
output parameter validator.
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_good_feature_detector_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_status status = VX_SUCCESS;
    if (index == 1)
    {
        vx_parameter output_param = vxGetParameterByIndex(node, 1);
        vx_array output;
        vx_size size = 0;

        STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_array)));
        STATUS_ERROR_CHECK(vxQueryArray(output, VX_ARRAY_ATTRIBUTE_CAPACITY, &size, sizeof(size)));

        if (size <= 0)
        {
            status = VX_ERROR_INVALID_VALUE;
        }

        vxReleaseArray(&output);
        vxReleaseParameter(&output_param);
    }

    return status;
}

/*!***********************************************************************************************************
Execution Kernel
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_good_feature_detector_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_SUCCESS;

    vx_image image_in = (vx_image) parameters[0];
    vx_array array = (vx_array) parameters[1];
    vx_scalar MAXCORNERS = (vx_scalar) parameters[2];
    vx_scalar QUALITYLEVEL = (vx_scalar) parameters[3];
    vx_scalar MINDISTANCE = (vx_scalar) parameters[4];
    vx_image mask = (vx_image) parameters[5];
    vx_scalar BLOCKSIZE = (vx_scalar) parameters[6];
    vx_scalar USEHARRISDETECTOR = (vx_scalar) parameters[7];
    vx_scalar K = (vx_scalar) parameters[8];

    Mat *mat, *mask_mat, Img;
    int maxCorners, blockSize;
    float qualityLevel, minDistance, k;
    vx_float32 FloatValue = 0;
    vx_int32 value = 0;
    vx_bool useHarris, value_b;

    //Extracting Values from the Scalar
    STATUS_ERROR_CHECK(vxReadScalarValue(QUALITYLEVEL, &FloatValue));
    qualityLevel = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MINDISTANCE, &FloatValue));
    minDistance = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(K, &FloatValue));
    k = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MAXCORNERS, &value));
    maxCorners = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(BLOCKSIZE, &value));
    blockSize = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(USEHARRISDETECTOR, &value_b));
    useHarris = value_b;

    //Converting VX Image to OpenCV Mat
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mask_mat, mask));

    //Compute using OpenCV
    vector<Point2f> Points2;
    bool useHarrisDetector;
    if (useHarris == 1) {
        useHarrisDetector = true;
    }
    else {
        useHarrisDetector = false;
    }
    cv::goodFeaturesToTrack(*mat, Points2, maxCorners, qualityLevel, minDistance, *mask_mat, blockSize, useHarrisDetector, k); ////Compute using OpenCV

    //Converting OpenCV Keypoints to OpenVX Keypoints
    STATUS_ERROR_CHECK(CVPoints2f_to_VX_keypoints(Points2, array));

    return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status CV_good_features_to_track_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_kernel Kernel = vxAddKernel(context,
                                   "org.opencv.good_features_to_track",
                                   VX_KERNEL_OPENCV_GOOD_FEATURE_TO_TRACK,
                                   CV_good_feature_detector_Kernel,
                                   9,
                                   CV_good_feature_detector_InputValidator,
                                   CV_good_feature_detector_OutputValidator,
                                   nullptr,
                                   nullptr);

    if (Kernel)
    {
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 1, VX_BIDIRECTIONAL, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 5, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(Kernel));
    }
    if (status != VX_SUCCESS)
    {
exit:
        vxRemoveKernel(Kernel);
        return VX_FAILURE;
    }

    return status;
}
