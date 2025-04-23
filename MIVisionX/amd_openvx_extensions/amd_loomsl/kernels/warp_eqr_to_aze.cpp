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

#define _CRT_SECURE_NO_WARNINGS
#include "warp_eqr_to_aze.h"

//! \brief The input validator callback.
static vx_status VX_CALLBACK warp_eqr_to_aze_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;

	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0)
	{ // image of format RGB
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGB) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: warp_to_sphere doesn't support input image format: %4.4s\n", &input_format);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 1)
	{ // array object of float32 type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		if (itemtype != VX_TYPE_FLOAT32) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: warp_to_sphere rad_lat_map array should be of FLOAT32 type\n");
		}
		else {
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 3)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
		if (itemtype != VX_TYPE_FLOAT32) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: warp_to_sphere parameter 'a' must be a scalar type of FLOAT32\n");
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 4)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
		if (itemtype != VX_TYPE_FLOAT32) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: warp_to_sphere parameter 'b' must be a scalar type of FLOAT32\n");
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 5)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
		if (itemtype != VX_TYPE_UINT8) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: warp_to_sphere parameter flags must be a scalar type of UINT8\n");
		}
		else {
			status = VX_SUCCESS;
		}
	}

	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK warp_eqr_to_aze_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;

	if (index == 2)
	{ // output image of format RGB or RGBX
		vx_uint32 output_width = 0, output_height = 0;
		// get image configuration
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_df_image output_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		if (!((output_format == VX_DF_IMAGE_RGB) || (output_format == VX_DF_IMAGE_RGBX))) {
			// pick RGB as default
			output_format = VX_DF_IMAGE_RGB;
		}
		// set output image meta data
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
		status = VX_SUCCESS;
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK warp_eqr_to_aze_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK warp_eqr_to_aze_opencl_codegen(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
	char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
	std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
	std::string& opencl_build_options,             // [output] options for clBuildProgram()
	vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
	vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
	vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
	)
{
	// get source image dimensions
	vx_uint32 src_width = 0, src_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &src_width, sizeof(src_width)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &src_height, sizeof(src_height)));
	vx_float32 src_width_f = (vx_float32)src_width;
	vx_float32 src_height_f = (vx_float32)src_height;
	// get destination image dimensions
	vx_uint32 dst_width = 0, dst_height = 0;
	vx_df_image dst_format = VX_DF_IMAGE_VIRT;
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[2], VX_IMAGE_WIDTH, &dst_width, sizeof(dst_width)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[2], VX_IMAGE_HEIGHT, &dst_height, sizeof(dst_height)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[2], VX_IMAGE_FORMAT, &dst_format, sizeof(dst_format)));
	vx_float32 dst_width_f = (vx_float32)dst_width;
	vx_float32 dst_height_f = (vx_float32)dst_height;
	// get the size of the radius to lat array
	vx_size arr_capacity = 0;
	ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
	// get interpolation flags
	vx_uint8 flags = 0;
	if (parameters[5]) {
		ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[5], &flags));
	}
	bool useBilinearInterpolation = (flags & 1) ? true : false;

	// set kernel configuration
	strcpy(opencl_kernel_function_name, "warp_eqr_to_aze");
	vx_uint32 work_items[2] = { (vx_uint32)((dst_width + 3) >> 2), (vx_uint32)dst_height };
	opencl_work_dim = 2;
	opencl_local_work[0] = 8;
	opencl_local_work[1] = 8;
	opencl_global_work[0] = (work_items[0] + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
	opencl_global_work[1] = (work_items[1] + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);

	char item[8192];

	if(useBilinearInterpolation)
	{
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n" // opencl_local_work[0], opencl_local_work[1]
			"void %s(\n" // opencl_kernel_function_name
			"        uint src_width, uint src_height, __global uchar * src_buf, uint src_stride, uint src_offset,\n"
			"        __global char * rad2lat_map_arr, uint rad2lat_map_arr_offset, uint rad2lat_map_arr_num_items,\n"
			"        uint dst_width, uint dst_height, __global uchar * dst_buf, uint dst_stride, uint dst_offset"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name);
		opencl_kernel_code = item;

		if (parameters[3]) {
			opencl_kernel_code +=
				",\n"
				"        float a\n";
		}
		if (parameters[4]) {
			opencl_kernel_code +=
				",\n"
				"        float b";
		}
		if (parameters[5]) {
			opencl_kernel_code +=
				",\n"
				"        uint flags";
		}

		snprintf(item, sizeof(item),
			")\n"
			"{\n"
			"  int gx = (int) get_global_id(0);\n"
			"  int gy = (int) get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d))\n" // work_items[0], work_items[1]
			"  {\n"
			"    rad2lat_map_arr += rad2lat_map_arr_offset;\n"
			"    __global float * map = (__global float *) rad2lat_map_arr;\n"
			"    // Remap generation\n"
			"    gx <<= 2;\n"
			"    float4 dx = (float4)((float)gx, (float)gx + 1.0f, (float)gx + 2.0f, (float)gx + 3.0f);\n"
			"    float4 dy = (float4)((float)gy);\n"
			"    dx -= (float)%f; dy -= (float)%f;\n" // dst_width_f/2 , dst_height_f/2
			"    float4 theta = atan2(dy, dx);\n"
			"    float4 radius = sqrt(dx*dx + dy*dy) / (float)%f;\n" // dr = min(dst_width_f,dst_height_f)/2
			"    int4 idx = convert_int4(radius * (float)%f);\n" // arr_capacity
			"    idx = min(idx, (int4)%d);\n" // arr_capacity
			, work_items[0], work_items[1], dst_width_f / 2.0f, dst_height_f / 2.0f, fmin(dst_width_f, dst_height_f) / 2.0f, (float)arr_capacity, (int)arr_capacity);
		opencl_kernel_code += item;

		if (!parameters[3]) {
			opencl_kernel_code +=
				"    float a = 0.0f;\n";
		}
		if (!parameters[4]) {
			opencl_kernel_code +=
				"    float b = 1.0f;\n";
		}
		if (dst_format == VX_DF_IMAGE_RGB) {
			snprintf(item, sizeof(item),
				"    theta *= b;\n"
				"    // Remap\n"
				"    src_buf += src_offset;"
				"    dst_buf += dst_offset + gy * dst_stride + gx * 3;"
				"    uint offset; uint3 px0, px1; __global uchar * pt; float4 f, mf; uint3 outpix; float sx, sy; int isValidRemap;\n"
				"    // pixel[0]\n"
				"    isValidRemap = (radius.s0 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s0]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s0 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)sy * src_stride + (int)sx * 3; pt = src_buf + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + src_stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = sx - (int)sx; mf.s1 = sy - (int)sy; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
				"    f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
				"    f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
				"    // pixel[1]\n"
				"    isValidRemap = (radius.s1 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s1]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s1 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)sy * src_stride + (int)sx * 3; pt = src_buf + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + src_stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = sx - (int)sx; mf.s1 = sy - (int)sy; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
				"    f.s3 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;"
				"    outpix.s0 = amd_pack(f);\n"
				"    f.s0 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s1 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
				"    // pixel[2]\n"
				"    isValidRemap = (radius.s2 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s2]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s2 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)sy * src_stride + (int)sx * 3; pt = src_buf + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + src_stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = sx - (int)sx; mf.s1 = sy - (int)sy; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
				"    f.s2 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
				"    f.s3 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
				"    outpix.s1 = amd_pack(f);\n"
				"    f.s0 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
				"    // pixel[3]\n"
				"    isValidRemap = (radius.s3 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s3]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s3 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)sy * src_stride + (int)sx * 3; pt = src_buf + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + src_stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = sx - (int)sx; mf.s1 = sy - (int)sy; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
				"    f.s1 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
				"    f.s2 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s3 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
				"    outpix.s2 = amd_pack(f);\n\n"
				"    *(__global uint2 *)dst_buf = outpix.s01; *(__global uint *)(dst_buf + 8) = outpix.s2;\n"
				"  }\n"
				"}\n"
				, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f);
		}
		else if (dst_format == VX_DF_IMAGE_RGBX) {
			snprintf(item, sizeof(item),
				"    theta *= b;\n"
				"    // Remap\n"
				"    src_buf += src_offset;"
				"    dst_buf += dst_offset + gy * dst_stride + gx * 4;"
				"    uint offset; uint3 px0, px1; __global uchar * pt; float4 f, mf; uint4 outpix; float sx, sy; int isValidRemap;\n"
				"    // pixel[0]\n"
				"    isValidRemap = (radius.s0 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s0]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s0 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)sy * src_stride + (int)sx * 3; pt = src_buf + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + src_stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = sx - (int)sx; mf.s1 = sy - (int)sy; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
				"    f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
				"    f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s3 = 255.0f; outpix.s0 = amd_pack(f);\n"
				"    // pixel[1]\n"
				"    isValidRemap = (radius.s1 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s1]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s1 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)sy * src_stride + (int)sx * 3; pt = src_buf + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + src_stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = sx - (int)sx; mf.s1 = sy - (int)sy; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
				"    f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;"
				"    f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s3 = 255.0f; outpix.s1 = amd_pack(f);\n"
				"    // pixel[2]\n"
				"    isValidRemap = (radius.s2 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s2]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s2 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)sy * src_stride + (int)sx * 3; pt = src_buf + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + src_stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = sx - (int)sx; mf.s1 = sy - (int)sy; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
				"    f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
				"    f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s3 = 255.0f; outpix.s2 = amd_pack(f);\n"
				"    // pixel[3]\n"
				"    isValidRemap = (radius.s3 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s3]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s3 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)sy * src_stride + (int)sx * 3; pt = src_buf + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + src_stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = sx - (int)sx; mf.s1 = sy - (int)sy; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
				"    f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
				"    f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
				"    f.s3 = 255.0f; outpix.s3 = amd_pack(f);\n"
				"    *(__global uint4 *)dst_buf = outpix;\n"
				"  }\n"
				"}\n"
				, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f);
		}

		opencl_kernel_code += item;
	}
	else // Bicubic
	{
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n\n"
			"float4 amd_unpack(uint src)\n"
			"{\n"
			"  return (float4)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src), amd_unpack3(src));\n"
			"}\n"
			"float4 compute_bicubic_coeffs(float x)\n"
			"{\n"
			"  float4 mf;\n"
			"  mf.s0 = -0.5f*x + x*x - 0.5f*x*x*x;\n"
			"  mf.s1 = 1.0f - 2.5f*x*x + 1.5f*x*x*x;\n"
			"  mf.s2 = 0.5f*x + 2.0f*x*x - 1.5f*x*x*x;\n"
			"  mf.s3 = 0.5f*(-x*x + x*x*x);\n"
			"  return(mf);\n"
			"}\n"
			"float3 interpolate_cubic_rgb(uint4 pix, float4 mf)\n"
			"{\n"
			"  float3 res;\n"
			"  res  = ((float3)(amd_unpack0(pix.s0), amd_unpack1(pix.s0), amd_unpack2(pix.s0))) * mf.s0;\n"
			"  res += ((float3)(amd_unpack3(pix.s0), amd_unpack0(pix.s1), amd_unpack1(pix.s1))) * mf.s1;\n"
			"  res += ((float3)(amd_unpack2(pix.s1), amd_unpack3(pix.s1), amd_unpack0(pix.s2))) * mf.s2;\n"
			"  res += ((float3)(amd_unpack1(pix.s2), amd_unpack2(pix.s2), amd_unpack3(pix.s2))) * mf.s3;\n"
			"  return(res);\n"
			"}\n"
			"\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n" // opencl_local_work[0], opencl_local_work[1]
			"void %s(\n" // opencl_kernel_function_name
			"        uint src_width, uint src_height, __global uchar * src_buf, uint src_stride, uint src_offset,\n"
			"        __global char * rad2lat_map_arr, uint rad2lat_map_arr_offset, uint rad2lat_map_arr_num_items,\n"
			"        uint dst_width, uint dst_height, __global uchar * dst_buf, uint dst_stride, uint dst_offset"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name);
		opencl_kernel_code = item;

		if (parameters[3]) {
			opencl_kernel_code +=
				",\n"
				"        float a\n";
		}
		if (parameters[4]) {
			opencl_kernel_code +=
				",\n"
				"        float b";
		}
		if (parameters[5]) {
			opencl_kernel_code +=
				",\n"
				"        uint flags";
		}

		snprintf(item, sizeof(item),
			")\n"
			"{\n"
			"  int gx = (int) get_global_id(0);\n"
			"  int gy = (int) get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d))\n" // work_items[0], work_items[1]
			"  {\n"
			"    rad2lat_map_arr += rad2lat_map_arr_offset;\n"
			"    __global float * map = (__global float *) rad2lat_map_arr;\n"
			"    // Remap generation\n"
			"    gx <<= 2;\n"
			"    float4 dx = (float4)((float)gx, (float)gx + 1.0f, (float)gx + 2.0f, (float)gx + 3.0f);\n"
			"    float4 dy = (float4)((float)gy);\n"
			"    dx -= (float)%f; dy -= (float)%f;\n" // dst_width_f/2 , dst_height_f/2
			"    float4 theta = atan2(dy, dx);\n"
			"    float4 radius = sqrt(dx*dx + dy*dy) / (float)%f;\n" // dr = min(dst_width_f,dst_height_f)/2
			"    int4 idx = convert_int4(radius * (float)%f);\n" // arr_capacity
			"    idx = min(idx, (int4)%d);\n" // arr_capacity
			, work_items[0], work_items[1], dst_width_f / 2.0f, dst_height_f / 2.0f, fmin(dst_width_f, dst_height_f) / 2.0f, (float)arr_capacity, (int)arr_capacity);
		opencl_kernel_code += item;

		if (!parameters[3]) {
			opencl_kernel_code +=
				"    float a = 0.0f;\n";
		}
		if (!parameters[4]) {
			opencl_kernel_code +=
				"    float b = 1.0f;\n";
		}
		if (dst_format == VX_DF_IMAGE_RGB) {
			snprintf(item, sizeof(item),
				"    theta *= b;\n"
				"    // Remap\n"
				"    src_buf += src_offset;"
				"    dst_buf += dst_offset + gy * dst_stride + gx * 3;"
				"    uint offset; uint4 px; __global uchar * pt; float4 f, mf; float3 tf; uint3 outpix; float sx, sy, y; int isValidRemap;\n"
				"    // pixel[0]\n"
				"    isValidRemap = (radius.s0 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s0]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s0 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)(sy - 1.0f) * src_stride + (int)(sx - 1.0f) * 3; pt = src_buf + (offset & ~3);\n"
				"    mf = compute_bicubic_coeffs(sx - floor(sx)); y = sy - floor(sy);\n"
				"    px = vload4(0, (__global uint *)pt);                  px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf = interpolate_cubic_rgb(px, mf) * (-0.5f*y + y*y - 0.5f*y*y*y);\n"
				"    px = vload4(0, (__global uint *)(pt +   src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (1.0f - 2.5f*y*y + 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 2*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (0.5f*y + 2.0f*y*y - 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 3*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (-0.5f*y*y + 0.5f*y*y*y));\n"
				"    f.s012 = tf;\n"
				"    // pixel[1]\n"
				"    isValidRemap = (radius.s1 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s1]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s1 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)(sy - 1.0f) * src_stride + (int)(sx - 1.0f) * 3; pt = src_buf + (offset & ~3);\n"
				"    mf = compute_bicubic_coeffs(sx - floor(sx)); y = sy - floor(sy);\n"
				"    px = vload4(0, (__global uint *)pt);                  px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf = interpolate_cubic_rgb(px, mf) * (-0.5f*y + y*y - 0.5f*y*y*y);\n"
				"    px = vload4(0, (__global uint *)(pt +   src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (1.0f - 2.5f*y*y + 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 2*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (0.5f*y + 2.0f*y*y - 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 3*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (-0.5f*y*y + 0.5f*y*y*y));\n"
				"    f.s3 = tf.s0;\n"
				"    outpix.s0 = amd_pack(f);\n"
				"    f.s01 = tf.s12;\n"
				"    // pixel[2]\n"
				"    isValidRemap = (radius.s2 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s2]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s2 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)(sy - 1.0f) * src_stride + (int)(sx - 1.0f) * 3; pt = src_buf + (offset & ~3);\n"
				"    mf = compute_bicubic_coeffs(sx - floor(sx)); y = sy - floor(sy);\n"
				"    px = vload4(0, (__global uint *)pt);                  px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf = interpolate_cubic_rgb(px, mf) * (-0.5f*y + y*y - 0.5f*y*y*y);\n"
				"    px = vload4(0, (__global uint *)(pt +   src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (1.0f - 2.5f*y*y + 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 2*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (0.5f*y + 2.0f*y*y - 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 3*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (-0.5f*y*y + 0.5f*y*y*y));\n"
				"    f.s23 = tf.s01;\n"
				"    outpix.s1 = amd_pack(f);\n"
				"    f.s0 = tf.s2;\n"
				"    // pixel[3]\n"
				"    isValidRemap = (radius.s3 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s3]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s3 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)(sy - 1.0f) * src_stride + (int)(sx - 1.0f) * 3; pt = src_buf + (offset & ~3);\n"
				"    mf = compute_bicubic_coeffs(sx - floor(sx)); y = sy - floor(sy);\n"
				"    px = vload4(0, (__global uint *)pt);                  px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf = interpolate_cubic_rgb(px, mf) * (-0.5f*y + y*y - 0.5f*y*y*y);\n"
				"    px = vload4(0, (__global uint *)(pt +   src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (1.0f - 2.5f*y*y + 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 2*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (0.5f*y + 2.0f*y*y - 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 3*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    tf += (interpolate_cubic_rgb(px, mf) * (-0.5f*y*y + 0.5f*y*y*y));\n"
				"    f.s123 = tf;\n"
				"    outpix.s2 = amd_pack(f);\n\n"
				"    *(__global uint2 *)dst_buf = outpix.s01; *(__global uint *)(dst_buf + 8) = outpix.s2;\n"
				"  }\n"
				"}\n"
				, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f);
		}
		else if (dst_format == VX_DF_IMAGE_RGBX) {
			snprintf(item, sizeof(item),
				"    theta *= b;\n"
				"    // Remap\n"
				"    src_buf += src_offset;"
				"    dst_buf += dst_offset + gy * dst_stride + gx * 4;"
				"    uint offset; uint4 px; __global uchar * pt; float4 f, mf; uint4 outpix; float sx, sy; int isValidRemap;\n"
				"    // pixel[0]\n"
				"    isValidRemap = (radius.s0 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s0]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s0 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)(sy - 1.0f) * src_stride + (int)(sx - 1.0f) * 3; pt = src_buf + (offset & ~3);\n"
				"    mf = compute_bicubic_coeffs(sx - floor(sx)); y = sy - floor(sy);\n"
				"    px = vload4(0, (__global uint *)pt);                  px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 = interpolate_cubic_rgb(px, mf) * (-0.5f*y + y*y - 0.5f*y*y*y);\n"
				"    px = vload4(0, (__global uint *)(pt +   src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (1.0f - 2.5f*y*y + 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 2*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (0.5f*y + 2.0f*y*y - 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 3*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (-0.5f*y*y + 0.5f*y*y*y));\n"
				"    f.s3 = 255.0f; outpix.s0 = amd_pack(f);\n"
				"    // pixel[1]\n"
				"    isValidRemap = (radius.s1 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s1]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s1 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)(sy - 1.0f) * src_stride + (int)(sx - 1.0f) * 3; pt = src_buf + (offset & ~3);\n"
				"    mf = compute_bicubic_coeffs(sx - floor(sx)); y = sy - floor(sy);\n"
				"    px = vload4(0, (__global uint *)pt);                  px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 = interpolate_cubic_rgb(px, mf) * (-0.5f*y + y*y - 0.5f*y*y*y);\n"
				"    px = vload4(0, (__global uint *)(pt +   src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (1.0f - 2.5f*y*y + 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 2*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (0.5f*y + 2.0f*y*y - 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 3*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (-0.5f*y*y + 0.5f*y*y*y));\n"
				"    f.s3 = 255.0f; outpix.s1 = amd_pack(f);\n"
				"    // pixel[2]\n"
				"    isValidRemap = (radius.s2 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s2]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s2 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)(sy - 1.0f) * src_stride + (int)(sx - 1.0f) * 3; pt = src_buf + (offset & ~3);\n"
				"    mf = compute_bicubic_coeffs(sx - floor(sx)); y = sy - floor(sy);\n"
				"    px = vload4(0, (__global uint *)pt);                  px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 = interpolate_cubic_rgb(px, mf) * (-0.5f*y + y*y - 0.5f*y*y*y);\n"
				"    px = vload4(0, (__global uint *)(pt +   src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (1.0f - 2.5f*y*y + 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 2*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (0.5f*y + 2.0f*y*y - 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 3*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (-0.5f*y*y + 0.5f*y*y*y));\n"
				"    f.s3 = 255.0f; outpix.s2 = amd_pack(f);\n"
				"    // pixel[3]\n"
				"    isValidRemap = (radius.s3 <= 1.0f) ? 1 : 0; sy = isValidRemap ? (90 - map[idx.s3]) * (float)%f : 0; sx = isValidRemap ? ((float)%f * ((float)%f - theta.s3 - a)) / (float)%f : 0;\n" // src_height_f/180, src_width_f, pi, 2pi
				"    offset = (int)(sy - 1.0f) * src_stride + (int)(sx - 1.0f) * 3; pt = src_buf + (offset & ~3);\n"
				"    mf = compute_bicubic_coeffs(sx - floor(sx)); y = sy - floor(sy);\n"
				"    px = vload4(0, (__global uint *)pt);                  px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 = interpolate_cubic_rgb(px, mf) * (-0.5f*y + y*y - 0.5f*y*y*y);\n"
				"    px = vload4(0, (__global uint *)(pt +   src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (1.0f - 2.5f*y*y + 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 2*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (0.5f*y + 2.0f*y*y - 1.5f*y*y*y));\n"
				"    px = vload4(0, (__global uint *)(pt + 3*src_stride)); px.s0 = amd_bytealign(px.s1, px.s0, offset); px.s1 = amd_bytealign(px.s2, px.s1, offset); px.s2 = amd_bytealign(px.s3, px.s2, offset);\n"
				"    f.s012 += (interpolate_cubic_rgb(px, mf) * (-0.5f*y*y + 0.5f*y*y*y));\n"
				"    f.s3 = 255.0f; outpix.s3 = amd_pack(f);\n"
				"    *(__global uint4 *)dst_buf = outpix;\n"
				"  }\n"
				"}\n"
				, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f, src_height_f / 180.0f, src_width_f, (float)M_PI, (float)M_PI * 2.0f);
		}

		opencl_kernel_code += item;
	}
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK warp_eqr_to_aze_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}

//! \brief The kernel publisher.
vx_status warp_eqr_to_aze_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.warp_eqr_to_aze",
		AMDOVX_KERNEL_STITCHING_WARP_EQR_TO_AZE,
		warp_eqr_to_aze_kernel,
		5,
		warp_eqr_to_aze_input_validator,
		warp_eqr_to_aze_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = warp_eqr_to_aze_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = warp_eqr_to_aze_opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}
