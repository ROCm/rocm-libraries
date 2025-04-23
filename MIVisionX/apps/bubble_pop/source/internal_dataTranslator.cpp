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

#include "internal_opencvTunnel.h"

/************************************************************************************************************
Converting CV Pyramid into an OpenVX Pyramid
*************************************************************************************************************/
int CV_to_VX_Pyramid(vx_pyramid pyramid_vx, vector<Mat> pyramid_cv)
{
	vx_status status = VX_SUCCESS;
	vx_size Level_vx = 0;
	vx_uint32 width = 0;
	vx_uint32 height = 0;
	vx_int32 i;

	STATUS_ERROR_CHECK(vxQueryPyramid(pyramid_vx, VX_PYRAMID_ATTRIBUTE_LEVELS, &Level_vx, sizeof(Level_vx)));
	for (i = 0; i < (int)Level_vx; i++)
	{
		vx_image this_level = vxGetPyramidLevel(pyramid_vx, i);
		STATUS_ERROR_CHECK(vxQueryImage(this_level, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		STATUS_ERROR_CHECK(vxQueryImage(this_level, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		if (width != pyramid_cv[i].cols && height != pyramid_cv[i].rows)
		{
			vxAddLogEntry((vx_reference)pyramid_vx, VX_ERROR_INVALID_DIMENSION, "CV_to_VX_Pyramid ERROR: Pyramid Image Mismatch\n");
			return VX_ERROR_INVALID_DIMENSION;
		}
		Mat *pyr_level;
		pyr_level = &pyramid_cv[i];
		CV_to_VX_Image(this_level, pyr_level);
	}
	return 0;
}

/************************************************************************************************************
Converting VX matrix into an OpenCV Mat
*************************************************************************************************************/
int VX_to_CV_MATRIX(Mat **mat, vx_matrix matrix_vx)
{
	vx_status status = VX_SUCCESS;
	vx_size numRows = 0;
	vx_size numCols = 0;
	vx_enum type;
	int Type_CV = 0;

	STATUS_ERROR_CHECK(vxQueryMatrix(matrix_vx, VX_MATRIX_ATTRIBUTE_ROWS, &numRows, sizeof(numRows)));
	STATUS_ERROR_CHECK(vxQueryMatrix(matrix_vx, VX_MATRIX_ATTRIBUTE_COLUMNS, &numCols, sizeof(numCols)));
	STATUS_ERROR_CHECK(vxQueryMatrix(matrix_vx, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type)));

	if (type == VX_TYPE_INT32)
		Type_CV = CV_32S;
	if (type == VX_TYPE_FLOAT32)
		Type_CV = CV_32F;

	if (type != VX_TYPE_FLOAT32 && type != VX_TYPE_INT32)
	{
		vxAddLogEntry((vx_reference)matrix_vx, VX_ERROR_INVALID_FORMAT, "VX_to_CV_MATRIX ERROR: Matrix type not Supported in this RELEASE\n");
		return VX_ERROR_INVALID_FORMAT;
	}

	Mat *m_cv;
	m_cv = new Mat((int)numRows, (int)numCols, Type_CV);
	vx_size mat_size = numRows * numCols;
	float *dyn_matrix = new float[mat_size];
	int z = 0;

	STATUS_ERROR_CHECK(vxReadMatrix(matrix_vx, (void *)dyn_matrix));
	for (int i = 0; i < (int)numRows; i++)
		for (int j = 0; j < (int)numCols; j++)
		{
			m_cv->at<float>(i, j) = dyn_matrix[z];
			z++;
		}

	*mat = m_cv;
	return status;
}

/************************************************************************************************************
Converting VX Image into an OpenCV Mat
*************************************************************************************************************/
int VX_to_CV_Image(Mat **mat, vx_image image)
{
	vx_status status = VX_SUCCESS;
	vx_uint32 width = 0;
	vx_uint32 height = 0;
	vx_df_image format = VX_DF_IMAGE_VIRT;
	int CV_format = 0;
	vx_size planes = 0;

	STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
	STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &planes, sizeof(planes)));

	if (format == VX_DF_IMAGE_U8)
		CV_format = CV_8U;
	if (format == VX_DF_IMAGE_S16)
		CV_format = CV_16S;
	if (format == VX_DF_IMAGE_RGB)
		CV_format = CV_8UC3;

	if (format != VX_DF_IMAGE_U8 && format != VX_DF_IMAGE_S16 && format != VX_DF_IMAGE_RGB)
	{
		vxAddLogEntry((vx_reference)image, VX_ERROR_INVALID_FORMAT, "VX_to_CV_Image ERROR: Image type not Supported in this RELEASE\n");
		return VX_ERROR_INVALID_FORMAT;
	}

	Mat *m_cv;
	m_cv = new Mat(height, width, CV_format);
	Mat *pMat = (Mat *)m_cv;
	vx_rectangle_t rect;
	rect.start_x = 0;
	rect.start_y = 0;
	rect.end_x = width;
	rect.end_y = height;

	vx_uint8 *src[4] = {NULL, NULL, NULL, NULL};
	vx_uint32 p;
	void *ptr = NULL;
	vx_imagepatch_addressing_t addr[4] = {0, 0, 0, 0};
	vx_uint32 y = 0u;

	for (p = 0u; (p < (int)planes); p++)
	{
		STATUS_ERROR_CHECK(vxAccessImagePatch(image, &rect, p, &addr[p], (void **)&src[p], VX_READ_ONLY));
		size_t len = addr[p].stride_x * (addr[p].dim_x * addr[p].scale_x) / VX_SCALE_UNITY;
		for (y = 0; y < height; y += addr[p].step_y)
		{
			ptr = vxFormatImagePatchAddress2d(src[p], 0, y - rect.start_y, &addr[p]);
			memcpy(pMat->data + y * pMat->step, ptr, len);
		}
	}

	for (p = 0u; p < (int)planes; p++)
		STATUS_ERROR_CHECK(vxCommitImagePatch(image, &rect, p, &addr[p], src[p]));

	*mat = pMat;

	return status;
}

/************************************************************************************************************
Converting CV Image into an OpenVX Image
*************************************************************************************************************/
int CV_to_VX_Image(vx_image image, Mat *mat)
{
	vx_status status = VX_SUCCESS;
	vx_uint32 width = 0;
	vx_uint32 height = 0;
	vx_size planes = 0;

	STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_PLANES, &planes, sizeof(planes)));

	Mat *pMat = mat;
	vx_rectangle_t rect;
	rect.start_x = 0;
	rect.start_y = 0;
	rect.end_x = width;
	rect.end_y = height;

	vx_uint8 *src[4] = {NULL, NULL, NULL, NULL};
	vx_uint32 p;
	void *ptr = NULL;
	vx_imagepatch_addressing_t addr[4] = {0, 0, 0, 0};
	vx_uint32 y = 0u;

	for (p = 0u; (p < (int)planes); p++)
	{
		STATUS_ERROR_CHECK(vxAccessImagePatch(image, &rect, p, &addr[p], (void **)&src[p], VX_READ_ONLY));
		size_t len = addr[p].stride_x * (addr[p].dim_x * addr[p].scale_x) / VX_SCALE_UNITY;
		for (y = 0; y < height; y += addr[p].step_y)
		{
			ptr = vxFormatImagePatchAddress2d(src[p], 0, y - rect.start_y, &addr[p]);
			memcpy(ptr, pMat->data + y * pMat->step, len);
		}
	}

	for (p = 0u; p < (int)planes; p++)
		STATUS_ERROR_CHECK(vxCommitImagePatch(image, &rect, p, &addr[p], src[p]));

	return status;
}

/************************************************************************************************************
sort function.
*************************************************************************************************************/
bool sortbysize_CV(const KeyPoint &lhs, const KeyPoint &rhs)
{
	return lhs.size < rhs.size;
}

/************************************************************************************************************
OpenCV Keypoints to OpenVX Keypoints
*************************************************************************************************************/
int CV_to_VX_keypoints(vector<KeyPoint> key_points, vx_array array)
{

	vx_status status = VX_SUCCESS;
	vector<vx_keypoint_t> Keypoint_VX;

	int x, y, j = 0;
	void *ptr = NULL;
	vx_size size = 0;

	STATUS_ERROR_CHECK(vxQueryArray(array, VX_ARRAY_ATTRIBUTE_CAPACITY, &size, sizeof(size)));

	size_t S = key_points.size();
	Keypoint_VX.resize(S);
	sort(key_points.begin(), key_points.end(), sortbysize_CV);
	vx_size stride = 0;
	void *base = NULL;
	vx_size L = 0;

	for (vector<KeyPoint>::const_iterator i = key_points.begin(); i != key_points.end(); ++i)
	{
		float X, Y, K_Size, K_Angle, K_Response;

		X = key_points[j].pt.x;
		Y = key_points[j].pt.y;
		K_Size = key_points[j].size;
		K_Angle = key_points[j].angle;
		K_Response = key_points[j].response;

		if (fmod(X, 1) >= 0.5)
			x = (int)ceil(X);
		else
			x = (int)floor(X);
		if (fmod(Y, 1) >= 0.5)
			y = (int)ceil(Y);
		else
			y = (int)floor(Y);

		Keypoint_VX[j].x = x;
		Keypoint_VX[j].y = y;
		Keypoint_VX[j].strength = K_Size;
		Keypoint_VX[j].orientation = K_Angle;
		Keypoint_VX[j].scale = K_Response;
		Keypoint_VX[j].tracking_status = 1;
		Keypoint_VX[j].error = 0;
		j++;
	}

	vx_keypoint_t *keypoint_ptr = &Keypoint_VX[0];
	size = min(size, S);

	status = vxTruncateArray(array, 0);
	if (status)
	{
		vxAddLogEntry((vx_reference)array, status, "CV_to_VX_keypoints ERROR: vxTruncateArray failed\n");
		return status;
	}

	status = vxAddArrayItems(array, size, keypoint_ptr, sizeof(vx_keypoint_t));
	if (status)
	{
		vxAddLogEntry((vx_reference)array, status, "CV_to_VX_keypoints ERROR: vxAddArrayItems failed\n");
		return status;
	}

	return status;
}

/************************************************************************************************************
OpenCV Points to OpenVX Keypoints
*************************************************************************************************************/
int CVPoints2f_to_VX_keypoints(vector<Point2f> key_points, vx_array array)
{
	vx_status status = VX_SUCCESS;
	vector<vx_keypoint_t> Keypoint_VX;

	int x, y, j = 0;
	void *ptr = NULL;
	vx_size size = 0;

	STATUS_ERROR_CHECK(vxQueryArray(array, VX_ARRAY_ATTRIBUTE_CAPACITY, &size, sizeof(size)));

	size_t S = key_points.size();
	Keypoint_VX.resize(S);

	for (int i = 0; i < (int)key_points.size(); ++i)
	{
		float X, Y;
		X = key_points[j].x;
		Y = key_points[j].y;

		if (fmod(X, 1) >= 0.5)
			x = (int)ceil(X);
		else
			x = (int)floor(X);
		if (fmod(Y, 1) >= 0.5)
			y = (int)ceil(Y);
		else
			y = (int)floor(Y);

		Keypoint_VX[j].x = x;
		Keypoint_VX[j].y = y;
		Keypoint_VX[j].strength = 0;
		Keypoint_VX[j].orientation = 0;
		Keypoint_VX[j].scale = 0;
		Keypoint_VX[j].tracking_status = 0;
		Keypoint_VX[j].error = 0;

		j++;
	}

	vx_keypoint_t *keypoint_ptr = &Keypoint_VX[0];
	size = min(size, S);

	status = vxTruncateArray(array, 0);
	if (status)
	{
		vxAddLogEntry((vx_reference)array, status, "CVPoints2f_to_VX_keypoints ERROR: vxTruncateArray failed\n");
		return status;
	}

	status = vxAddArrayItems(array, size, keypoint_ptr, sizeof(vx_keypoint_t));
	if (status)
	{
		vxAddLogEntry((vx_reference)array, status, "CVPoints2f_to_VX_keypoints ERROR: vxAddArrayItems failed\n");
		return status;
	}

	return status;
}

/************************************************************************************************************
OpenCV Descriptors to OpenVX Descriptors
*************************************************************************************************************/
int CV_DESP_to_VX_DESP(Mat mat, vx_array array, int stride)
{
	vx_status status = VX_SUCCESS;
	vx_size size = 0;

	STATUS_ERROR_CHECK(vxQueryArray(array, VX_ARRAY_ATTRIBUTE_CAPACITY, &size, sizeof(size)));

	uchar *p = mat.data;

	status = vxTruncateArray(array, 0);
	if (status)
	{
		vxAddLogEntry((vx_reference)array, status, "CV_DESP_to_VX_DESP ERROR: vxTruncateArray failed\n");
		return status;
	}

	status = vxAddArrayItems(array, size, p, stride);
	if (status)
	{
		vxAddLogEntry((vx_reference)array, status, "CV_DESP_to_VX_DESP ERROR: vxAddArrayItems failed\n");
		return status;
	}

	return status;
}

/************************************************************************************************************
Match VX in and out image size
*************************************************************************************************************/
int match_vx_image_parameters(vx_image image1, vx_image image2)
{
	vx_status status = VX_SUCCESS;
	vx_uint32 W1 = 0;
	vx_uint32 H1 = 0;
	STATUS_ERROR_CHECK(vxQueryImage(image1, VX_IMAGE_ATTRIBUTE_WIDTH, &W1, sizeof(W1)));
	STATUS_ERROR_CHECK(vxQueryImage(image1, VX_IMAGE_ATTRIBUTE_HEIGHT, &H1, sizeof(H1)));

	vx_uint32 W2 = 0;
	vx_uint32 H2 = 0;
	STATUS_ERROR_CHECK(vxQueryImage(image2, VX_IMAGE_ATTRIBUTE_WIDTH, &W2, sizeof(W2)));
	STATUS_ERROR_CHECK(vxQueryImage(image2, VX_IMAGE_ATTRIBUTE_HEIGHT, &H2, sizeof(H2)));

	// Input and Output image size match check
	if (W1 != W2 || H1 != H2)
	{
		status = VX_ERROR_INVALID_DIMENSION;
		vxAddLogEntry((vx_reference)image1, status, "match_vx_image_parameters ERROR: Image1 Height or Width Not Equal to Image2\n");
		return status;
	}

	return status;
}
