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
#include "color_convert.h"

//! \brief The input validator callback.
static vx_status VX_CALLBACK color_convert_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0)
	{ // image of format UYVY or Y210 or Y216
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		if (format == VX_DF_IMAGE_UYVY || format == VX_DF_IMAGE_YUYV || format == VX_DF_IMAGE_Y210_AMD || format == VX_DF_IMAGE_Y216_AMD || format == VX_DF_IMAGE_RGB) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: color_convert doesn't support input image format: %4.4s\n", &format);
		}
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK color_convert_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 1)
	{ // image of format RGB or RGBX
		// get image configuration
		vx_image image = (vx_image)avxGetNodeParamRef(node, 0);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 output_width = 0, output_height = 0;
		vx_df_image output_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		if (input_width != output_width || input_height != output_height)
		{
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: color_convert doesn't support input & output image with different dimensions\n");
			return status;
		}
		if ((input_format == VX_DF_IMAGE_UYVY || input_format == VX_DF_IMAGE_YUYV || input_format == VX_DF_IMAGE_Y210_AMD || input_format == VX_DF_IMAGE_Y216_AMD) && output_format != VX_DF_IMAGE_RGB && output_format != VX_DF_IMAGE_RGBX) {
			// pick RGBX as default
			output_format = VX_DF_IMAGE_RGBX;
		}
		else if ((input_format == VX_DF_IMAGE_RGB) && (output_format != VX_DF_IMAGE_UYVY) && (output_format != VX_DF_IMAGE_YUYV)) {
			// pick UYVY as default
			output_format = VX_DF_IMAGE_UYVY;
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
static vx_status VX_CALLBACK color_convert_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK color_convert_opencl_codegen(
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
	// get input and output image configurations
	vx_uint32 input_width = 0, input_height = 0, output_width = 0, output_height = 0;
	vx_df_image input_format = VX_DF_IMAGE_VIRT, output_format = VX_DF_IMAGE_VIRT;
	vx_channel_range_e input_channel_range, output_channel_range;
	vx_color_space_e input_color_space, output_color_space;
	vx_image image = (vx_image)avxGetNodeParamRef(node, 0);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_RANGE, &input_channel_range, sizeof(input_channel_range)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_SPACE, &input_color_space, sizeof(input_color_space)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	image = (vx_image)avxGetNodeParamRef(node, 1);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_RANGE, &output_channel_range, sizeof(output_channel_range)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_SPACE, &output_color_space, sizeof(output_color_space)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));

	// set kernel configuration
	vx_uint32 work_items[2] = { (input_width + 7) / 8, (input_height + 1) / 2 };
	strcpy(opencl_kernel_function_name, "color_convert");
	opencl_work_dim = 2;
	opencl_local_work[0] = 16;
	opencl_local_work[1] = 4;
	opencl_global_work[0] = (work_items[0] + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
	opencl_global_work[1] = (work_items[1] + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n" // opencl_local_work[0], opencl_local_work[1]
		"void %s(uint p422_width, uint p422_height, __global uchar * p422_buf, uint p422_stride, uint p422_offset,\n" // opencl_kernel_function_name
		"        uint pRGB_width, uint pRGB_height, __global uchar * pRGB_buf, uint pRGB_stride, uint pRGB_offset)\n"
		"{\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  if ((gx < %d) && (gy < %d)) {\n" // work_items[0], work_items[1]
		"    uint8 pRGB0, pRGB1;\n"
		, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, work_items[0], work_items[1]);
	opencl_kernel_code = item;

	if (input_format == VX_DF_IMAGE_UYVY || input_format == VX_DF_IMAGE_YUYV) {
		if (input_color_space == VX_COLOR_SPACE_BT601_525 || input_color_space == VX_COLOR_SPACE_BT601_625) {
			opencl_kernel_code +=
				"    float2 cR = (float2)( 0.0000f,  1.4030f);\n"
				"    float2 cG = (float2)(-0.3440f, -0.7140f);\n"
				"    float2 cB = (float2)( 1.7730f,  0.0000f);\n";
		}
		else { // VX_COLOR_SPACE_BT709
			opencl_kernel_code +=
				"    float2 cR = (float2)( 0.0000f,  1.5748f);\n"
				"    float2 cG = (float2)(-0.1873f, -0.4681f);\n"
				"    float2 cB = (float2)( 1.8556f,  0.0000f);\n";
		}
		if (input_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
			opencl_kernel_code +=
				"    float4 r2f = (float4)(256f/219f, -16f*256f/219f, 256f/224f, -128f*256f/224f);\n";
		}
		else { // VX_CHANNEL_RANGE_FULL
			opencl_kernel_code +=
				"    float4 r2f = (float4)(1.0f, 0.0f, 1.0f, -128.0f);\n";
		}
		if (output_format == VX_DF_IMAGE_RGBX) {
			if (input_format == VX_DF_IMAGE_UYVY) {
				opencl_kernel_code +=
					"    uint4 L0, L1;\n"
					"    p422_buf += p422_offset + (gy * p422_stride * 2) + (gx << 4);\n"
					"    L0 = *(__global uint4 *) p422_buf;\n"
					"    L1 = *(__global uint4 *)&p422_buf[p422_stride];\n"
					"    float4 f; float y0, y1, u, v;\n"
					"    u = mad(amd_unpack0(L0.s0),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L0.s0),r2f.s0,r2f.s1); v = mad(amd_unpack2(L0.s0),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L0.s0),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s0 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s1 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L0.s1),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L0.s1),r2f.s0,r2f.s1); v = mad(amd_unpack2(L0.s1),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L0.s1),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s2 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s3 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L0.s2),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L0.s2),r2f.s0,r2f.s1); v = mad(amd_unpack2(L0.s2),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L0.s2),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s4 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s5 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L0.s3),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L0.s3),r2f.s0,r2f.s1); v = mad(amd_unpack2(L0.s3),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L0.s3),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s6 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s7 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L1.s0),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L1.s0),r2f.s0,r2f.s1); v = mad(amd_unpack2(L1.s0),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L1.s0),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s0 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s1 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L1.s1),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L1.s1),r2f.s0,r2f.s1); v = mad(amd_unpack2(L1.s1),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L1.s1),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s2 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s3 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L1.s2),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L1.s2),r2f.s0,r2f.s1); v = mad(amd_unpack2(L1.s2),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L1.s2),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s4 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s5 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L1.s3),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L1.s3),r2f.s0,r2f.s1); v = mad(amd_unpack2(L1.s3),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L1.s3),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s6 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s7 = amd_pack(f);\n";
			}
			else if (input_format == VX_DF_IMAGE_YUYV) {
				opencl_kernel_code +=
					"    uint4 L0, L1;\n"
					"    p422_buf += p422_offset + (gy * p422_stride * 2) + (gx << 4);\n"
					"    L0 = *(__global uint4 *) p422_buf;\n"
					"    L1 = *(__global uint4 *)&p422_buf[p422_stride];\n"
					"    float4 f; float y0, y1, u, v;\n"
					"    u = mad(amd_unpack1(L0.s0),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L0.s0),r2f.s0,r2f.s1); v = mad(amd_unpack3(L0.s0),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L0.s0),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s0 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s1 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L0.s1),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L0.s1),r2f.s0,r2f.s1); v = mad(amd_unpack3(L0.s1),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L0.s1),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s2 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s3 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L0.s2),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L0.s2),r2f.s0,r2f.s1); v = mad(amd_unpack3(L0.s2),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L0.s2),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s4 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s5 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L0.s3),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L0.s3),r2f.s0,r2f.s1); v = mad(amd_unpack3(L0.s3),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L0.s3),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s6 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s7 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L1.s0),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L1.s0),r2f.s0,r2f.s1); v = mad(amd_unpack3(L1.s0),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L1.s0),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s0 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s1 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L1.s1),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L1.s1),r2f.s0,r2f.s1); v = mad(amd_unpack3(L1.s1),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L1.s1),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s2 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s3 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L1.s2),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L1.s2),r2f.s0,r2f.s1); v = mad(amd_unpack3(L1.s2),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L1.s2),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s4 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s5 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L1.s3),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L1.s3),r2f.s0,r2f.s1); v = mad(amd_unpack3(L1.s3),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L1.s3),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s6 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s7 = amd_pack(f);\n";
			}
		}
		else {
			if (input_format == VX_DF_IMAGE_UYVY) {
				opencl_kernel_code +=
					"    uint4 L0, L1;\n"
					"    p422_buf += p422_offset + (gy * p422_stride * 2) + (gx << 4);\n"
					"    L0 = *(__global uint4 *) p422_buf;\n"
					"    L1 = *(__global uint4 *)&p422_buf[p422_stride];\n"
					"    float4 f; float y0, y1, u, v; f.s3 = 0.0f;\n"
					"    u = mad(amd_unpack0(L0.s0),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L0.s0),r2f.s0,r2f.s1); v = mad(amd_unpack2(L0.s0),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L0.s0),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s0 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s1 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L0.s1),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L0.s1),r2f.s0,r2f.s1); v = mad(amd_unpack2(L0.s1),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L0.s1),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s2 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s3 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L0.s2),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L0.s2),r2f.s0,r2f.s1); v = mad(amd_unpack2(L0.s2),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L0.s2),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s4 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s5 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L0.s3),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L0.s3),r2f.s0,r2f.s1); v = mad(amd_unpack2(L0.s3),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L0.s3),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s6 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s7 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L1.s0),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L1.s0),r2f.s0,r2f.s1); v = mad(amd_unpack2(L1.s0),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L1.s0),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s0 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s1 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L1.s1),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L1.s1),r2f.s0,r2f.s1); v = mad(amd_unpack2(L1.s1),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L1.s1),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s2 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s3 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L1.s2),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L1.s2),r2f.s0,r2f.s1); v = mad(amd_unpack2(L1.s2),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L1.s2),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s4 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s5 = amd_pack(f);\n"
					"    u = mad(amd_unpack0(L1.s3),r2f.s2,r2f.s3); y0 = mad(amd_unpack1(L1.s3),r2f.s0,r2f.s1); v = mad(amd_unpack2(L1.s3),r2f.s2,r2f.s3); y1 = mad(amd_unpack3(L1.s3),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s6 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s7 = amd_pack(f);\n";
			}
			else if (input_format == VX_DF_IMAGE_YUYV) {
				opencl_kernel_code +=
					"    uint4 L0, L1;\n"
					"    p422_buf += p422_offset + (gy * p422_stride * 2) + (gx << 4);\n"
					"    L0 = *(__global uint4 *) p422_buf;\n"
					"    L1 = *(__global uint4 *)&p422_buf[p422_stride];\n"
					"    float4 f; float y0, y1, u, v; f.s3 = 0.0f;\n"
					"    u = mad(amd_unpack1(L0.s0),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L0.s0),r2f.s0,r2f.s1); v = mad(amd_unpack3(L0.s0),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L0.s0),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s0 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s1 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L0.s1),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L0.s1),r2f.s0,r2f.s1); v = mad(amd_unpack3(L0.s1),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L0.s1),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s2 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s3 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L0.s2),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L0.s2),r2f.s0,r2f.s1); v = mad(amd_unpack3(L0.s2),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L0.s2),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s4 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s5 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L0.s3),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L0.s3),r2f.s0,r2f.s1); v = mad(amd_unpack3(L0.s3),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L0.s3),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s6 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s7 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L1.s0),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L1.s0),r2f.s0,r2f.s1); v = mad(amd_unpack3(L1.s0),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L1.s0),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s0 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s1 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L1.s1),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L1.s1),r2f.s0,r2f.s1); v = mad(amd_unpack3(L1.s1),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L1.s1),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s2 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s3 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L1.s2),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L1.s2),r2f.s0,r2f.s1); v = mad(amd_unpack3(L1.s2),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L1.s2),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s4 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s5 = amd_pack(f);\n"
					"    u = mad(amd_unpack1(L1.s3),r2f.s2,r2f.s3); y0 = mad(amd_unpack0(L1.s3),r2f.s0,r2f.s1); v = mad(amd_unpack3(L1.s3),r2f.s2,r2f.s3); y1 = mad(amd_unpack2(L1.s3),r2f.s0,r2f.s1);\n"
					"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s6 = amd_pack(f);\n"
					"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s7 = amd_pack(f);\n";
			}
		}
	}
	else if (input_format == VX_DF_IMAGE_Y210_AMD || input_format == VX_DF_IMAGE_Y216_AMD)
	{
		if (input_format == VX_DF_IMAGE_Y210_AMD) {
			opencl_kernel_code +=
				"    float2 cR = (float2)( 0.00000000f,  1.57943176f);\n"
				"    float2 cG = (float2)(-0.18785088f, -0.46947676f);\n"
				"    float2 cB = (float2)( 1.86105765f,  0.00000000f);\n";
		}
		else {
			opencl_kernel_code +=
				"    float2 cR = (float2)( 0.00000000f,  1.5809516f);\n"
				"    float2 cG = (float2)(-0.18803164f, -0.46992852f);\n"
				"    float2 cB = (float2)( 1.86284844f,  0.00000000f);\n";
		}
		if (output_format == VX_DF_IMAGE_RGBX) {
			opencl_kernel_code +=
				"    uint8 L0, L1;\n"
				"    p422_buf += p422_offset + (gy * p422_stride * 2) + (gx << 5);\n"
				"    L0 = *(__global uint8 *) p422_buf;\n"
				"    L1 = *(__global uint8 *)&p422_buf[p422_stride];\n"
				"    float4 f; float y0, y1, u, v;\n"
				"    u = mad(0.00390625f, amd_unpack1(L0.s0), amd_unpack0(L0.s0)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L0.s0), amd_unpack2(L0.s0));\n"
				"    v = mad(0.00390625f, amd_unpack1(L0.s1), amd_unpack0(L0.s1)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L0.s1), amd_unpack2(L0.s1));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s0 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s1 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L0.s2), amd_unpack0(L0.s2)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L0.s2), amd_unpack2(L0.s2));\n"
				"    v = mad(0.00390625f, amd_unpack1(L0.s3), amd_unpack0(L0.s3)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L0.s3), amd_unpack2(L0.s3));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s2 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s3 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L0.s4), amd_unpack0(L0.s4)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L0.s4), amd_unpack2(L0.s4));\n"
				"    v = mad(0.00390625f, amd_unpack1(L0.s5), amd_unpack0(L0.s5)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L0.s5), amd_unpack2(L0.s5));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s4 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s5 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L0.s6), amd_unpack0(L0.s6)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L0.s6), amd_unpack2(L0.s6));\n"
				"    v = mad(0.00390625f, amd_unpack1(L0.s7), amd_unpack0(L0.s7)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L0.s7), amd_unpack2(L0.s7));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB0.s6 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB0.s7 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L1.s0), amd_unpack0(L1.s0)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L1.s0), amd_unpack2(L1.s0));\n"
				"    v = mad(0.00390625f, amd_unpack1(L1.s1), amd_unpack0(L1.s1)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L1.s1), amd_unpack2(L1.s1));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s0 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s1 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L1.s2), amd_unpack0(L1.s2)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L1.s2), amd_unpack2(L1.s2));\n"
				"    v = mad(0.00390625f, amd_unpack1(L1.s3), amd_unpack0(L1.s3)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L1.s3), amd_unpack2(L1.s3));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s2 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s3 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L1.s4), amd_unpack0(L1.s4)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L1.s4), amd_unpack2(L1.s4));\n"
				"    v = mad(0.00390625f, amd_unpack1(L1.s5), amd_unpack0(L1.s5)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L1.s5), amd_unpack2(L1.s5));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s4 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s5 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L1.s6), amd_unpack0(L1.s6)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L1.s6), amd_unpack2(L1.s6));\n"
				"    v = mad(0.00390625f, amd_unpack1(L1.s7), amd_unpack0(L1.s7)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L1.s7), amd_unpack2(L1.s7));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); f.s3 = y0; pRGB1.s6 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); f.s3 = y1; pRGB1.s7 = amd_pack(f);\n";
		}
		else {
			opencl_kernel_code +=
				"    uint8 L0, L1;\n"
				"    p422_buf += p422_offset + (gy * p422_stride * 2) + (gx << 5);\n"
				"    L0 = *(__global uint8 *) p422_buf;\n"
				"    L1 = *(__global uint8 *)&p422_buf[p422_stride];\n"
				"    float4 f; float y0, y1, u, v; f.s3 = 0.0f;\n"
				"    u = mad(0.00390625f, amd_unpack1(L0.s0), amd_unpack0(L0.s0)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L0.s0), amd_unpack2(L0.s0));\n"
				"    v = mad(0.00390625f, amd_unpack1(L0.s1), amd_unpack0(L0.s1)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L0.s1), amd_unpack2(L0.s1));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s0 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s1 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L0.s2), amd_unpack0(L0.s2)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L0.s2), amd_unpack2(L0.s2));\n"
				"    v = mad(0.00390625f, amd_unpack1(L0.s3), amd_unpack0(L0.s3)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L0.s3), amd_unpack2(L0.s3));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s2 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s3 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L0.s4), amd_unpack0(L0.s4)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L0.s4), amd_unpack2(L0.s4));\n"
				"    v = mad(0.00390625f, amd_unpack1(L0.s5), amd_unpack0(L0.s5)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L0.s5), amd_unpack2(L0.s5));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s4 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s5 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L0.s6), amd_unpack0(L0.s6)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L0.s6), amd_unpack2(L0.s6));\n"
				"    v = mad(0.00390625f, amd_unpack1(L0.s7), amd_unpack0(L0.s7)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L0.s7), amd_unpack2(L0.s7));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB0.s6 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB0.s7 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L1.s0), amd_unpack0(L1.s0)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L1.s0), amd_unpack2(L1.s0));\n"
				"    v = mad(0.00390625f, amd_unpack1(L1.s1), amd_unpack0(L1.s1)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L1.s1), amd_unpack2(L1.s1));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s0 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s1 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L1.s2), amd_unpack0(L1.s2)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L1.s2), amd_unpack2(L1.s2));\n"
				"    v = mad(0.00390625f, amd_unpack1(L1.s3), amd_unpack0(L1.s3)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L1.s3), amd_unpack2(L1.s3));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s2 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s3 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L1.s4), amd_unpack0(L1.s4)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L1.s4), amd_unpack2(L1.s4));\n"
				"    v = mad(0.00390625f, amd_unpack1(L1.s5), amd_unpack0(L1.s5)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L1.s5), amd_unpack2(L1.s5));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s4 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s5 = amd_pack(f);\n"
				"    u = mad(0.00390625f, amd_unpack1(L1.s6), amd_unpack0(L1.s6)-128.0f); y0 = mad(0.00390625f, amd_unpack3(L1.s6), amd_unpack2(L1.s6));\n"
				"    v = mad(0.00390625f, amd_unpack1(L1.s7), amd_unpack0(L1.s7)-128.0f); y1 = mad(0.00390625f, amd_unpack3(L1.s7), amd_unpack2(L1.s7));\n"
				"    f.s0 = mad(cR.s1, v, y0); f.s1 = mad(cG.s0, u, y0); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y0); pRGB1.s6 = amd_pack(f);\n"
				"    f.s0 = mad(cR.s1, v, y1); f.s1 = mad(cG.s0, u, y1); f.s1 = mad(cG.s1, v, f.s1); f.s2 = mad(cB.s0, u, y1); pRGB1.s7 = amd_pack(f);\n";
		}
	}
	else // input_format RGB
	{
		if (output_format == VX_DF_IMAGE_UYVY) {
			opencl_kernel_code +=
				"    float3 cY = (float3)(0.2126f, 0.7152f, 0.0722f);\n"
				"    float3 cU = (float3)(-0.1146f, -0.3854f, 0.5f);\n"
				"    float3 cV = (float3)(0.5f, -0.4542f, -0.0458f);\n"
				"    uint8 L0, L1;\n"
				"    p422_buf += p422_offset + (gy * p422_stride * 2) + (gx * 24);\n"
				"    L0 = *(__global uint8 *) p422_buf;\n"
				"    L1 = *(__global uint8 *)&p422_buf[p422_stride];\n"
				"    float4 f; float3 rgb;\n"
				"    rgb = (float3)(amd_unpack0(L0.s0), amd_unpack1(L0.s0), amd_unpack2(L0.s0));\n"
				"    f.s0 = dot(cU, rgb) + 128.0f; f.s1 = dot(cY, rgb); f.s2 = dot(cV, rgb) + 128.0f; f.s3 = dot(cY, (float3)(amd_unpack3(L0.s0), amd_unpack0(L0.s1), amd_unpack1(L0.s1))); pRGB0.s0 = amd_pack(f);\n" // TBD: BT601 U and V are at even pixel location(0,2,..), so ignoring odd pixels for chroma conversion
				"    rgb = (float3)(amd_unpack2(L0.s1), amd_unpack3(L0.s1), amd_unpack0(L0.s2));\n"
				"    f.s0 = dot(cU, rgb) + 128.0f; f.s1 = dot(cY, rgb); f.s2 = dot(cV, rgb) + 128.0f; f.s3 = dot(cY, (float3)(amd_unpack1(L0.s2), amd_unpack2(L0.s2), amd_unpack3(L0.s2))); pRGB0.s1 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack0(L0.s3), amd_unpack1(L0.s3), amd_unpack2(L0.s3));\n"
				"    f.s0 = dot(cU, rgb) + 128.0f; f.s1 = dot(cY, rgb); f.s2 = dot(cV, rgb) + 128.0f; f.s3 = dot(cY, (float3)(amd_unpack3(L0.s3), amd_unpack0(L0.s4), amd_unpack1(L0.s4))); pRGB0.s2 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack2(L0.s4), amd_unpack3(L0.s4), amd_unpack0(L0.s5));\n"
				"    f.s0 = dot(cU, rgb) + 128.0f; f.s1 = dot(cY, rgb); f.s2 = dot(cV, rgb) + 128.0f; f.s3 = dot(cY, (float3)(amd_unpack1(L0.s5), amd_unpack2(L0.s5), amd_unpack3(L0.s5))); pRGB0.s3 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack0(L1.s0), amd_unpack1(L1.s0), amd_unpack2(L1.s0));\n"
				"    f.s0 = dot(cU, rgb) + 128.0f; f.s1 = dot(cY, rgb); f.s2 = dot(cV, rgb) + 128.0f; f.s3 = dot(cY, (float3)(amd_unpack3(L1.s0), amd_unpack0(L1.s1), amd_unpack1(L1.s1))); pRGB0.s4 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack2(L1.s1), amd_unpack3(L1.s1), amd_unpack0(L1.s2));\n"
				"    f.s0 = dot(cU, rgb) + 128.0f; f.s1 = dot(cY, rgb); f.s2 = dot(cV, rgb) + 128.0f; f.s3 = dot(cY, (float3)(amd_unpack1(L1.s2), amd_unpack2(L1.s2), amd_unpack3(L1.s2))); pRGB0.s5 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack0(L1.s3), amd_unpack1(L1.s3), amd_unpack2(L1.s3));\n"
				"    f.s0 = dot(cU, rgb) + 128.0f; f.s1 = dot(cY, rgb); f.s2 = dot(cV, rgb) + 128.0f; f.s3 = dot(cY, (float3)(amd_unpack3(L1.s3), amd_unpack0(L1.s4), amd_unpack1(L1.s4))); pRGB0.s6 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack2(L1.s4), amd_unpack3(L1.s4), amd_unpack0(L1.s5));\n"
				"    f.s0 = dot(cU, rgb) + 128.0f; f.s1 = dot(cY, rgb); f.s2 = dot(cV, rgb) + 128.0f; f.s3 = dot(cY, (float3)(amd_unpack1(L1.s5), amd_unpack2(L1.s5), amd_unpack3(L1.s5))); pRGB0.s7 = amd_pack(f);\n";
		}
		else if (output_format == VX_DF_IMAGE_YUYV) {
			opencl_kernel_code +=
				"    float3 cY = (float3)(0.2126f, 0.7152f, 0.0722f);\n"
				"    float3 cU = (float3)(-0.1146f, -0.3854f, 0.5f);\n"
				"    float3 cV = (float3)(0.5f, -0.4542f, -0.0458f);\n"
				"    uint8 L0, L1;\n"
				"    p422_buf += p422_offset + (gy * p422_stride * 2) + (gx * 24);\n"
				"    L0 = *(__global uint8 *) p422_buf;\n"
				"    L1 = *(__global uint8 *)&p422_buf[p422_stride];\n"
				"    float4 f; float3 rgb;\n"
				"    rgb = (float3)(amd_unpack0(L0.s0), amd_unpack1(L0.s0), amd_unpack2(L0.s0));\n"
				"    f.s1 = dot(cU, rgb) + 128.0f; f.s0 = dot(cY, rgb); f.s3 = dot(cV, rgb) + 128.0f; f.s2 = dot(cY, (float3)(amd_unpack3(L0.s0), amd_unpack0(L0.s1), amd_unpack1(L0.s1))); pRGB0.s0 = amd_pack(f);\n" // TBD: BT601 U and V are at even pixel location(0,2,..), so ignoring odd pixels for chroma conversion
				"    rgb = (float3)(amd_unpack2(L0.s1), amd_unpack3(L0.s1), amd_unpack0(L0.s2));\n"
				"    f.s1 = dot(cU, rgb) + 128.0f; f.s0 = dot(cY, rgb); f.s3 = dot(cV, rgb) + 128.0f; f.s2 = dot(cY, (float3)(amd_unpack1(L0.s2), amd_unpack2(L0.s2), amd_unpack3(L0.s2))); pRGB0.s1 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack0(L0.s3), amd_unpack1(L0.s3), amd_unpack2(L0.s3));\n"
				"    f.s1 = dot(cU, rgb) + 128.0f; f.s0 = dot(cY, rgb); f.s3 = dot(cV, rgb) + 128.0f; f.s2 = dot(cY, (float3)(amd_unpack3(L0.s3), amd_unpack0(L0.s4), amd_unpack1(L0.s4))); pRGB0.s2 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack2(L0.s4), amd_unpack3(L0.s4), amd_unpack0(L0.s5));\n"
				"    f.s1 = dot(cU, rgb) + 128.0f; f.s0 = dot(cY, rgb); f.s3 = dot(cV, rgb) + 128.0f; f.s2 = dot(cY, (float3)(amd_unpack1(L0.s5), amd_unpack2(L0.s5), amd_unpack3(L0.s5))); pRGB0.s3 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack0(L1.s0), amd_unpack1(L1.s0), amd_unpack2(L1.s0));\n"
				"    f.s1 = dot(cU, rgb) + 128.0f; f.s0 = dot(cY, rgb); f.s3 = dot(cV, rgb) + 128.0f; f.s2 = dot(cY, (float3)(amd_unpack3(L1.s0), amd_unpack0(L1.s1), amd_unpack1(L1.s1))); pRGB0.s4 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack2(L1.s1), amd_unpack3(L1.s1), amd_unpack0(L1.s2));\n"
				"    f.s1 = dot(cU, rgb) + 128.0f; f.s0 = dot(cY, rgb); f.s3 = dot(cV, rgb) + 128.0f; f.s2 = dot(cY, (float3)(amd_unpack1(L1.s2), amd_unpack2(L1.s2), amd_unpack3(L1.s2))); pRGB0.s5 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack0(L1.s3), amd_unpack1(L1.s3), amd_unpack2(L1.s3));\n"
				"    f.s1 = dot(cU, rgb) + 128.0f; f.s0 = dot(cY, rgb); f.s3 = dot(cV, rgb) + 128.0f; f.s2 = dot(cY, (float3)(amd_unpack3(L1.s3), amd_unpack0(L1.s4), amd_unpack1(L1.s4))); pRGB0.s6 = amd_pack(f);\n"
				"    rgb = (float3)(amd_unpack2(L1.s4), amd_unpack3(L1.s4), amd_unpack0(L1.s5));\n"
				"    f.s1 = dot(cU, rgb) + 128.0f; f.s0 = dot(cY, rgb); f.s3 = dot(cV, rgb) + 128.0f; f.s2 = dot(cY, (float3)(amd_unpack1(L1.s5), amd_unpack2(L1.s5), amd_unpack3(L1.s5))); pRGB0.s7 = amd_pack(f);\n";
		}
	}
	if (output_format == VX_DF_IMAGE_RGBX) {
		opencl_kernel_code +=
			"    pRGB_buf += pRGB_offset + (gy * pRGB_stride * 2) + (gx << 5);\n"
			"    *(__global uint8 *) pRGB_buf = pRGB0;\n"
			"    *(__global uint8 *)&pRGB_buf[pRGB_stride] = pRGB1;\n";

	}
	else if (output_format == VX_DF_IMAGE_RGB)
	{
		opencl_kernel_code +=
			"    pRGB0.s0 = ((pRGB0.s0 & 0x00ffffff)      ) + (pRGB0.s1 << 24);\n"
			"    pRGB0.s1 = ((pRGB0.s1 & 0x00ffff00) >>  8) + (pRGB0.s2 << 16);\n"
			"    pRGB0.s2 = ((pRGB0.s2 & 0x00ff0000) >> 16) + (pRGB0.s3 <<  8);\n"
			"    pRGB0.s4 = ((pRGB0.s4 & 0x00ffffff)      ) + (pRGB0.s5 << 24);\n"
			"    pRGB0.s5 = ((pRGB0.s5 & 0x00ffff00) >>  8) + (pRGB0.s6 << 16);\n"
			"    pRGB0.s6 = ((pRGB0.s6 & 0x00ff0000) >> 16) + (pRGB0.s7 <<  8);\n"
			"    pRGB1.s0 = ((pRGB1.s0 & 0x00ffffff)      ) + (pRGB1.s1 << 24);\n"
			"    pRGB1.s1 = ((pRGB1.s1 & 0x00ffff00) >>  8) + (pRGB1.s2 << 16);\n"
			"    pRGB1.s2 = ((pRGB1.s2 & 0x00ff0000) >> 16) + (pRGB1.s3 <<  8);\n"
			"    pRGB1.s4 = ((pRGB1.s4 & 0x00ffffff)      ) + (pRGB1.s5 << 24);\n"
			"    pRGB1.s5 = ((pRGB1.s5 & 0x00ffff00) >>  8) + (pRGB1.s6 << 16);\n"
			"    pRGB1.s6 = ((pRGB1.s6 & 0x00ff0000) >> 16) + (pRGB1.s7 <<  8);\n"
			"    pRGB_buf += pRGB_offset + (gy * pRGB_stride * 2) + (gx * 24);\n"
			"    *(__global uint3 *) pRGB_buf = pRGB0.s012;\n"
			"    *(__global uint3 *)&pRGB_buf[12] = pRGB0.s456;\n"
			"    *(__global uint3 *)&pRGB_buf[pRGB_stride] = pRGB1.s012;\n"
			"    *(__global uint3 *)&pRGB_buf[pRGB_stride+12] = pRGB1.s456;\n";
	}
	else // output format UYVY
	{
		opencl_kernel_code +=
			"    pRGB_buf += pRGB_offset + (gy * pRGB_stride * 2) + (gx << 4);\n"
			"    *(__global uint4 *) pRGB_buf = pRGB0.s0123;\n"
			"    *(__global uint4 *)&pRGB_buf[pRGB_stride] = pRGB0.s4567;\n";
	}
	opencl_kernel_code +=
		"  }\n"
		"}\n";

	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK color_convert_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}

//! \brief The kernel publisher.
vx_status color_convert_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.color_convert",
		AMDOVX_KERNEL_STITCHING_COLOR_CONVERT,
		color_convert_kernel,
		2,
		color_convert_input_validator,
		color_convert_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = color_convert_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = color_convert_opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}
