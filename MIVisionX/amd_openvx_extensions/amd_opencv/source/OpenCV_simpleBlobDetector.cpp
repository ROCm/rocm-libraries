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
static vx_status VX_CALLBACK CV_simple_blob_detector_InputValidator(vx_node node, vx_uint32 index)
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

    if (index == 2)
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

    vxReleaseParameter(&param);
    return status;
}

/*!***********************************************************************************************************
output parameter validator.
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_simple_blob_detector_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
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
static vx_status VX_CALLBACK CV_simple_blob_detector_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_image image_in = (vx_image) parameters[0];
    vx_array array = (vx_array) parameters[1];
    vx_image mask = (vx_image) parameters[2];
    vector<KeyPoint> key_points;
    Mat *mat, *mask_mat, Img;

    //Converting VX Image to OpenCV Mat
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mask_mat, mask));

    //OpenCV Calls to Simple Blob Detector
    Ptr<Feature2D> simple = SimpleBlobDetector::create();
    simple->detect(*mat, key_points, *mask_mat);

    //Converting OpenCV Keypoints to OpenVX Keypoints
    STATUS_ERROR_CHECK(CV_to_VX_keypoints(key_points, array));

    return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status CV_simple_blob_detect_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_kernel Kernel = vxAddKernel(context,
                                   "org.opencv.simple_blob_detect",
                                   VX_KERNEL_OPENCV_SIMPLE_BLOB_DETECT,
                                   CV_simple_blob_detector_Kernel,
                                   3,
                                   CV_simple_blob_detector_InputValidator,
                                   CV_simple_blob_detector_OutputValidator,
                                   nullptr,
                                   nullptr);

    if (Kernel)
    {
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 1, VX_BIDIRECTIONAL, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
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

/*!***********************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_simple_blob_detector_INITIALIZE_InputValidator(vx_node node, vx_uint32 index)
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

    if (index == 2)
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

    else if (index == 5)
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

    else if (index == 6)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_size value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (type != VX_TYPE_SIZE)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }
    else if (index == 7)
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

    else if (index == 8)
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

    else if (index == 9)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_uint16 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (type != VX_TYPE_UINT16)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 10)
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

    else if (index == 11)
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

    else if (index == 12)
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

    else if (index == 13)
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

    else if (index == 14)
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

    else if (index == 15)
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

    else if (index == 16)
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

    else if (index == 17)
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

    else if (index == 18)
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

    else if (index == 19)
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

    else if (index == 20)
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

    else if (index == 21)
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
Execution Kernel
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_simple_blob_detector_INITIALIZE_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_SUCCESS;

    vx_image image_in = (vx_image) parameters[0];
    vx_array array = (vx_array) parameters[1];
    vx_image mask = (vx_image) parameters[2];

    vector<KeyPoint> key_points;
    Mat *mat, *mask_mat, Img;

    vx_scalar THRESHOLDSTEP = (vx_scalar) parameters[3];
    vx_scalar MINTHRESHOLD = (vx_scalar) parameters[4];
    vx_scalar MAXTHRESHOLD = (vx_scalar) parameters[5];
    vx_scalar MINREPEAT = (vx_scalar) parameters[6];
    vx_scalar MINDISTBTW = (vx_scalar) parameters[7];
    vx_scalar FILTERBYCOLOR = (vx_scalar) parameters[8];
    vx_scalar BLOBCOLOR = (vx_scalar) parameters[9];
    vx_scalar FILTERBYAREA = (vx_scalar) parameters[10];
    vx_scalar MINAREA = (vx_scalar) parameters[11];
    vx_scalar MAXAREA = (vx_scalar) parameters[12];
    vx_scalar FILTERBYCIR = (vx_scalar) parameters[13];
    vx_scalar MINCIR = (vx_scalar) parameters[14];
    vx_scalar MAXCIR = (vx_scalar) parameters[15];
    vx_scalar FILTERBYINER = (vx_scalar) parameters[16];
    vx_scalar MININER = (vx_scalar) parameters[17];
    vx_scalar MAXINER = (vx_scalar) parameters[18];
    vx_scalar FILTERBYCON = (vx_scalar) parameters[19];
    vx_scalar MINCON = (vx_scalar) parameters[20];
    vx_scalar MAXCON = (vx_scalar) parameters[21];

    size_t minRepeatability;
    vx_bool filterByColor, filterByArea, filterByCircularity, filterByInertia, filterByConvexity;
    vx_uint16 blobColor;
    float thresholdStep, minThreshold, maxThreshold, minDistBetweenBlobs, minArea, maxArea, minCircularity, maxCircularity, minInertiaRatio, maxInertiaRatio, minConvexity, maxConvexity;
    vx_float32 FloatValue = 0;
    vx_bool value;
    vx_size value_s = 0;

    //Extracting Values from the Scalar
    STATUS_ERROR_CHECK(vxReadScalarValue(MINREPEAT, &value_s));
    minRepeatability = value_s;
    STATUS_ERROR_CHECK(vxReadScalarValue(THRESHOLDSTEP, &FloatValue));
    thresholdStep = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MINTHRESHOLD, &FloatValue));
    minThreshold = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MAXTHRESHOLD, &FloatValue));
    maxThreshold = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MINDISTBTW, &FloatValue));
    minDistBetweenBlobs = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MINAREA, &FloatValue));
    minArea = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MAXAREA, &FloatValue));
    maxArea = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MINCIR, &FloatValue));
    minCircularity = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MAXCIR, &FloatValue));
    maxCircularity = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MININER, &FloatValue));
    minInertiaRatio = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MAXINER, &FloatValue));
    maxInertiaRatio = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MINCON, &FloatValue));
    minConvexity = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(MAXCON, &FloatValue));
    maxConvexity = FloatValue;
    STATUS_ERROR_CHECK(vxReadScalarValue(FILTERBYCOLOR, &value));
    filterByColor = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(FILTERBYAREA, &value));
    filterByArea = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(FILTERBYCIR, &value));
    filterByCircularity = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(FILTERBYINER, &value));
    filterByInertia = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(FILTERBYCON, &value));
    filterByConvexity = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(BLOBCOLOR, &blobColor));

    //Converting VX Image to OpenCV Mat
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mask_mat, mask));

    //OpenCV Calls to Simple Blob Detector
    bool filterByColor_bool, filterByArea_bool, filterByCircularity_bool, filterByConvexity_bool, filterByInertia_bool;
    if (filterByColor == vx_true_e) {
        filterByColor_bool = true;
    }
    else {
        filterByColor_bool = false;
    }
    if (filterByArea == vx_true_e) {
        filterByArea_bool = true;
    }
    else {
        filterByArea_bool = false;
    }
    if (filterByCircularity == vx_true_e) {
        filterByCircularity_bool = true;
    }
    else {
        filterByCircularity_bool = false;
    }
    if (filterByConvexity == vx_true_e)  {
        filterByConvexity_bool = true;
    }
    else  {
        filterByConvexity_bool = false;
    }
    if (filterByInertia == vx_true_e)  {
        filterByInertia_bool = true;
    }
    else  {
        filterByInertia_bool = false;
    }
    cv::SimpleBlobDetector::Params params;
    params.blobColor = (uchar)blobColor;
    params.thresholdStep = thresholdStep;
    params.filterByArea = filterByArea_bool;
    params.filterByCircularity = filterByCircularity_bool;
    params.filterByColor = filterByColor_bool;
    params.filterByConvexity = filterByCircularity_bool;
    params.filterByInertia = filterByInertia_bool;
    params.maxThreshold = maxThreshold;
    params.maxArea = maxArea;
    params.maxConvexity = maxConvexity;
    params.maxInertiaRatio = maxInertiaRatio;
    params.maxCircularity = maxCircularity;
    params.maxCircularity = minCircularity;
    params.maxInertiaRatio = minInertiaRatio;
    params.minThreshold = minThreshold;
    params.minRepeatability = minRepeatability;
    params.minDistBetweenBlobs = minDistBetweenBlobs;
    params.minArea = minArea;
    params.minConvexity = minConvexity;

    Ptr<Feature2D> simple = SimpleBlobDetector::create(params);
    simple->detect(*mat, key_points, *mask_mat);

    //Converting OpenCV Keypoints to OpenVX Keypoints
    STATUS_ERROR_CHECK(CV_to_VX_keypoints(key_points, array));

    return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status CV_simple_blob_detect_initialize_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_kernel Kernel = vxAddKernel(context,
                                   "org.opencv.simple_blob_detect_initialize",
                                   VX_KERNEL_OPENCV_SIMPLE_BLOB_DETECT_INITIALIZE,
                                   CV_simple_blob_detector_INITIALIZE_Kernel,
                                   22,
                                   CV_simple_blob_detector_INITIALIZE_InputValidator,
                                   CV_simple_blob_detector_OutputValidator,
                                   nullptr,
                                   nullptr);

    if (Kernel)
    {
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 1, VX_BIDIRECTIONAL, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 12, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 13, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 14, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 15, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 16, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 17, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 18, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 19, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 20, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 21, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
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
