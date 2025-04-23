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
#include "merge.h"

//! \brief The input validator callback.
static vx_status VX_CALLBACK merge_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0)
	{ // image object of U008 type
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		if (format == VX_DF_IMAGE_U8) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: merge camera id selection for image 0 should be an image of U008 type\n");
		}
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		return status;
	}
	else if (index == 1 || index == 2)
	{ // image of format S016
		vx_image image = (vx_image)avxGetNodeParamRef(node, 0);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_U16) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: merge camera id selection for image %d should be an image of U016 type\n", index);
		}
		else if ((input_width != width) || (input_height != height)) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: merge invalid input camera id selection image %d dimensions %dx%d do not match camera id selection image 0 dimensions %dx%d\n", index, input_width, input_height, width, height);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 3)
	{ // input image of format RGBX
		vx_image image = (vx_image)avxGetNodeParamRef(node, 0);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGBX) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: merge input image should be of RGBX type\n");
		}
		else if ((input_width != width * 8)) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: merge invalid input RGBX image dimensions %dx%d do not match camera id selection image 0 dimensions %dx%d\n", input_width, input_height, width, height);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 4)
	{ // input weight image of format U008
		vx_image image = (vx_image)avxGetNodeParamRef(node, 3);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_U8) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: merge input weight image should be of U008 type\n");
		}
		else if ((input_width != width) || (input_height != height)) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: merge invalid input weight image dimensions %dx%d do not match input image dimensions %dx%d\n", input_width, input_height, width, height);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK merge_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 5)
	{ // image of format RGB2
		// get image configuration
		vx_image image = (vx_image)avxGetNodeParamRef(node, 0);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 input_width = 0, input_height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		// set output image meta data
		image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 output_width = 0, output_height = 0;
		vx_df_image output_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		if (input_width != (output_width >> 3))
		{ // pick default output width as input map width * 8
			output_width = input_width << 3;
		}
		if (input_height != output_height)
		{ // pick default output height as the input map height
			output_height = input_height;
		}
		if ((output_format != VX_DF_IMAGE_RGB) && (output_format != VX_DF_IMAGE_RGBX))
		{ // pick default output format RGB
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
static vx_status VX_CALLBACK merge_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK merge_opencl_codegen(
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
	vx_uint32 width = 0, height = 0;
	vx_df_image output_format = VX_DF_IMAGE_VIRT;
	vx_image image = (vx_image)avxGetNodeParamRef(node, 5);				// output image
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	// set kernel configuration
	strcpy(opencl_kernel_function_name, "merge");
	vx_uint32 work_items[2] = { (width + 3) / 4, height };
	opencl_work_dim = 2;
	opencl_local_work[0] = 8;
	opencl_local_work[1] = 8;
	opencl_global_work[0] = (work_items[0] + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
	opencl_global_work[1] = (work_items[1] + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);

	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

	vx_float32 wt_mul_factor = 1.0f / 255.0f;
	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"float4 amd_unpack(uint src)\n"
		"{\n"
		"  return (float4)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src), amd_unpack3(src));\n"
		"}\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n" // opencl_local_work[0], opencl_local_work[1]
		"void %s(uint camID0_img_width, uint camID0_img_height, __global uchar * camID0_img_buf, uint camID0_img_stride, uint camID0_img_offset,\n" // opencl_kernel_function_name
		"        uint camID1_img_width, uint camID1_img_height, __global uchar * camID1_img_buf, uint camID1_img_stride, uint camID1_img_offset,\n"
		"        uint camID2_img_width, uint camID2_img_height, __global uchar * camID2_img_buf, uint camID2_img_stride, uint camID2_img_offset,\n"
		"        uint ip_width, uint ip_height, __global uchar * ip_buf, uint ip_stride, uint ip_offset,\n"
		"        uint wt_width, uint wt_height, __global uchar * wt_buf, uint wt_stride, uint wt_offset,\n"
		"        uint op_width, uint op_height, __global uchar * op_buf, uint op_stride, uint op_offset)\n"
		"{\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  float weight_mul_factor = %f;\n" // wt_mul_factor
		"  if ((gx < %d) && (gy < %d)) {\n" // work_items[0], work_items[1]
		, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, wt_mul_factor, work_items[0], work_items[1]);
	opencl_kernel_code = item;

	opencl_kernel_code +=
		"  uint4 pRGB_out;\n"
		"  camID0_img_buf += camID0_img_offset + gy * camID0_img_stride + (gx >> 1);\n"
		"  uchar camIdSelect = *(__global uchar *)camID0_img_buf;\n"
		"  uint4 pRGBX_in; float4 weights;\n"
		"  float16 fa = 0;\n"
		"  if(camIdSelect < 31) {\n"
		"    pRGBX_in = *(__global uint4 *) (ip_buf + ip_offset + ((gy + op_height * camIdSelect) * ip_stride) + (gx << 4));\n"
		"    fa.s0123 += amd_unpack(pRGBX_in.s0);\n"
		"    fa.s4567 += amd_unpack(pRGBX_in.s1);\n"
		"    fa.s89AB += amd_unpack(pRGBX_in.s2);\n"
		"    fa.sCDEF += amd_unpack(pRGBX_in.s3);\n"
		"  }\n"
		"  else if(camIdSelect > 31) {\n"
		"    camID1_img_buf += camID1_img_offset + gy * camID1_img_stride + ((gx >> 1) << 1);\n"
		"    ushort camID_struct = *(__global ushort *)camID1_img_buf;\n"
		"    ushort camId = camID_struct & 0x1f;\n"
		"    pRGBX_in = (uint4)0;  weights = (float4)0;\n"
		"    if (camId < 31) {\n"
		"      pRGBX_in = *(__global uint4 *) (ip_buf + ip_offset + ((gy + op_height * camId) * ip_stride) + (gx << 4));\n"
		"      weights = convert_float4(*(__global uchar4 *) (wt_buf + wt_offset + ((gy + op_height * camId) * wt_stride) + (gx << 2)));\n"
		"      weights *= weight_mul_factor;\n"
		"    }\n"
		"    fa.s0123 += weights.s0 * amd_unpack(pRGBX_in.s0); fa.s4567 += weights.s1 * amd_unpack(pRGBX_in.s1); fa.s89AB += weights.s2 * amd_unpack(pRGBX_in.s2); fa.sCDEF += weights.s3 * amd_unpack(pRGBX_in.s3);\n"
		"    \n"
		"    camId = (camID_struct >> 5) & 0x1f;\n"
		"    weights = (float4)0;\n"
		"    if (camId < 31) {\n"
		"      pRGBX_in = *(__global uint4 *) (ip_buf + ip_offset + ((gy + op_height * camId) * ip_stride) + (gx << 4));\n"
		"      weights = convert_float4(*(__global uchar4 *) (wt_buf + wt_offset + ((gy + op_height * camId) * wt_stride) + (gx << 2)));\n"
		"      weights *= weight_mul_factor;\n"
		"    }\n"
		"    fa.s0123 = mad((float4)weights.s0, amd_unpack(pRGBX_in.s0), fa.s0123); fa.s4567 = mad((float4)weights.s1, amd_unpack(pRGBX_in.s1), fa.s4567); fa.s89AB = mad((float4)weights.s2, amd_unpack(pRGBX_in.s2), fa.s89AB); fa.sCDEF = mad((float4)weights.s3, amd_unpack(pRGBX_in.s3), fa.sCDEF);\n"
		"    if(camIdSelect > 128) {\n"
		"      camId = (camID_struct >> 10) & 0x1f;\n"
		"      weights = (float4)0;\n"
		"      if (camId < 31) {\n"
		"        pRGBX_in = *(__global uint4 *) (ip_buf + ip_offset + ((gy + op_height * camId) * ip_stride) + (gx << 4));\n"
		"        weights = convert_float4(*(__global uchar4 *) (wt_buf + wt_offset + ((gy + op_height * camId) * wt_stride) + (gx << 2)));\n"
		"        weights *= weight_mul_factor;\n"
		"      }\n"
		"      fa.s0123 = mad((float4)weights.s0, amd_unpack(pRGBX_in.s0), fa.s0123); fa.s4567 = mad((float4)weights.s1, amd_unpack(pRGBX_in.s1), fa.s4567); fa.s89AB = mad((float4)weights.s2, amd_unpack(pRGBX_in.s2), fa.s89AB); fa.sCDEF = mad((float4)weights.s3, amd_unpack(pRGBX_in.s3), fa.sCDEF);\n"
		"    }\n\n"
		"    if(camIdSelect > 129) {\n"
		"      camID2_img_buf += camID2_img_offset + gy * camID2_img_stride + ((gx >> 1) << 1);\n"
		"      camID_struct = *(__global ushort *)camID2_img_buf;\n"
		"      camId = camID_struct & 0x1f;\n"
		"      weights = (float4)0;\n"
		"      if (camId < 31) {\n"
		"        pRGBX_in = *(__global uint4 *) (ip_buf + ip_offset + ((gy + op_height * camId) * ip_stride) + (gx << 4));\n"
		"        weights = convert_float4(*(__global uchar4 *) (wt_buf + wt_offset + ((gy + op_height * camId) * wt_stride) + (gx << 2)));\n"
		"        weights *= weight_mul_factor;\n"
		"      }\n"
		"      fa.s0123 = mad((float4)weights.s0, amd_unpack(pRGBX_in.s0), fa.s0123); fa.s4567 = mad((float4)weights.s1, amd_unpack(pRGBX_in.s1), fa.s4567); fa.s89AB = mad((float4)weights.s2, amd_unpack(pRGBX_in.s2), fa.s89AB); fa.sCDEF = mad((float4)weights.s3, amd_unpack(pRGBX_in.s3), fa.sCDEF);\n"
		"    }\n\n"
		"    if(camIdSelect > 130) {\n"
		"      camId = (camID_struct >> 5) & 0x1f;\n"
		"       weights = (float4)0;\n"
		"      if (camId < 31) {\n"
		"        pRGBX_in = *(__global uint4 *) (ip_buf + ip_offset + ((gy + op_height * camId) * ip_stride) + (gx << 4));\n"
		"        weights = convert_float4(*(__global uchar4 *) (wt_buf + wt_offset + ((gy + op_height * camId) * wt_stride) + (gx << 2)));\n"
		"        weights *= weight_mul_factor;\n"
		"      }\n"
		"      fa.s0123 = mad((float4)weights.s0, amd_unpack(pRGBX_in.s0), fa.s0123); fa.s4567 = mad((float4)weights.s1, amd_unpack(pRGBX_in.s1), fa.s4567); fa.s89AB = mad((float4)weights.s2, amd_unpack(pRGBX_in.s2), fa.s89AB); fa.sCDEF = mad((float4)weights.s3, amd_unpack(pRGBX_in.s3), fa.sCDEF);\n"
		"    }\n\n"
		"    if(camIdSelect > 131) {\n"
		"      camId = (camID_struct >> 10) & 0x1f;\n"
		"      weights = (float4)0;\n"
		"      if (camId < 31) {\n"
		"        pRGBX_in = *(__global uint4 *) (ip_buf + ip_offset + ((gy + op_height * camId) * ip_stride) + (gx << 4));\n"
		"        weights = convert_float4(*(__global uchar4 *) (wt_buf + wt_offset + ((gy + op_height * camId) * wt_stride) + (gx << 2)));\n"
		"        weights *= weight_mul_factor;\n"
		"      }\n"
		"      fa.s0123 = mad((float4)weights.s0, amd_unpack(pRGBX_in.s0), fa.s0123); fa.s4567 = mad((float4)weights.s1, amd_unpack(pRGBX_in.s1), fa.s4567); fa.s89AB = mad((float4)weights.s2, amd_unpack(pRGBX_in.s2), fa.s89AB); fa.sCDEF = mad((float4)weights.s3, amd_unpack(pRGBX_in.s3), fa.sCDEF);\n"
		"    }\n"
		"  }\n\n";
	if (output_format == VX_DF_IMAGE_RGB) {
		opencl_kernel_code +=
			"  pRGB_out.s0 = amd_pack(fa.s0124); pRGB_out.s1 = amd_pack(fa.s5689); pRGB_out.s2 = amd_pack(fa.sACDE);\n"
			"  if(camIdSelect != 31) {\n"
			"    op_buf += op_offset + gy * op_stride + gx * 12;\n"
			"    vstore3(pRGB_out.s012, 0, (__global uint *) op_buf);\n"
			"    }\n"
			"  }\n"
			"}";
	}
	else { // RGBX out
		opencl_kernel_code +=
			"  uint Xmask = 0xff000000;\n"
			"  pRGB_out.s0 = amd_pack(fa.s0123); pRGB_out.s0 |= Xmask;\n"
			"  pRGB_out.s1 = amd_pack(fa.s4567); pRGB_out.s1 |= Xmask;\n"
			"  pRGB_out.s2 = amd_pack(fa.s89AB); pRGB_out.s2 |= Xmask;\n"
			"  pRGB_out.s3 = amd_pack(fa.sCDEF); pRGB_out.s3 |= Xmask;\n"
			"  if(camIdSelect != 31) {\n"
			"    op_buf += op_offset + gy * op_stride + gx << 4;\n"
			"    *(__global uint4 *) op_buf = pRGB_out;\n"
			"    }\n"
			"  }\n"
			"}";
	}
	return VX_SUCCESS;
}

//! \brief The kernel initialize.
static vx_status VX_CALLBACK merge_initialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_SUCCESS;
}

//! \brief The kernel deinitialize.
static vx_status VX_CALLBACK merge_deinitialize(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK merge_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status merge_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.merge",
		AMDOVX_KERNEL_STITCHING_MERGE,
		merge_kernel,
		6,
		merge_input_validator,
		merge_output_validator,
		merge_initialize,
		merge_deinitialize);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = merge_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = merge_opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

//////////////////////////////////////////////////////////////////////
// Generate data in buffers for merge
vx_status GenerateMergeBuffers(
	vx_uint32 numCamera,                  // [in] number of cameras
	vx_uint32 eqrWidth,                   // [in] output equirectangular image width
	vx_uint32 eqrHeight,                  // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,   // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_uint32 * paddedPixelCamMap,  // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight] (optional)
	vx_uint32  camIdStride,               // [in] stride (in bytes) of camId table (image)
	vx_uint32  camGroup1Stride,           // [in] stride (in bytes) of camGroup1 table (image)
	vx_uint32  camGroup2Stride,           // [in] stride (in bytes) of camGroup2 table (image)
	vx_uint8 * camIdBuf,                  // [out] camId table (image)
	StitchMergeCamIdEntry * camGroup1Buf, // [out] camId Group1 table (image)
	StitchMergeCamIdEntry * camGroup2Buf  // [out] camId Group2 table (image)
	)
{
	for (vx_uint32 y = 0; y < eqrHeight; y++) {
		for (vx_uint32 x = 0, xi = 0; x < eqrWidth; x += 8, xi++) {
			vx_uint32 validMaskFor8Pixels =
				validPixelCamMap[x + 0] | validPixelCamMap[x + 1] |
				validPixelCamMap[x + 2] | validPixelCamMap[x + 3] |
				validPixelCamMap[x + 4] | validPixelCamMap[x + 5] |
				validPixelCamMap[x + 6] | validPixelCamMap[x + 7];
			if (paddedPixelCamMap) {
				validMaskFor8Pixels |=
					paddedPixelCamMap[x + 0] | paddedPixelCamMap[x + 1] |
					paddedPixelCamMap[x + 2] | paddedPixelCamMap[x + 3] |
					paddedPixelCamMap[x + 4] | paddedPixelCamMap[x + 5] |
					paddedPixelCamMap[x + 6] | paddedPixelCamMap[x + 7];
			}
			vx_uint32 count = GetOneBitCount(validMaskFor8Pixels);
			vx_uint8 camId = 31;
			vx_uint8 id[6] = { 31, 31, 31, 31, 31, 31 };
			if (count == 1) {
				// use two pixel mode with second cam as 31 if not all 8 pixels are from same camera
				vx_uint32 commonValidMaskFor8Pixels =
					validPixelCamMap[x + 0] & validPixelCamMap[x + 1] &
					validPixelCamMap[x + 2] & validPixelCamMap[x + 3] &
					validPixelCamMap[x + 4] & validPixelCamMap[x + 5] &
					validPixelCamMap[x + 6] & validPixelCamMap[x + 7];
				if (paddedPixelCamMap) {
					commonValidMaskFor8Pixels =
						(validPixelCamMap[x + 0] | paddedPixelCamMap[x + 0]) &
						(validPixelCamMap[x + 1] | paddedPixelCamMap[x + 1]) &
						(validPixelCamMap[x + 2] | paddedPixelCamMap[x + 2]) &
						(validPixelCamMap[x + 3] | paddedPixelCamMap[x + 3]) &
						(validPixelCamMap[x + 4] | paddedPixelCamMap[x + 4]) &
						(validPixelCamMap[x + 5] | paddedPixelCamMap[x + 5]) &
						(validPixelCamMap[x + 6] | paddedPixelCamMap[x + 6]) &
						(validPixelCamMap[x + 7] | paddedPixelCamMap[x + 7]);
				}

				if (commonValidMaskFor8Pixels == 0) {
					count = 2;
				}
			}
			if (count == 1) {
				camId = (vx_uint8)GetOneBitPosition(validMaskFor8Pixels);
			}
			else if (count >= 2) {
				camId = (vx_uint8)(126 + count);
				for (vx_uint32 j = 0, k = 0; j < 32 && k < 6; j++) {
					if (validMaskFor8Pixels & (1 << j)) {
						id[k++] = j;
					}
				}
			}
			StitchMergeCamIdEntry group1 = { id[0], id[1], id[2], 0 };
			StitchMergeCamIdEntry group2 = { id[3], id[4], id[5], 0 };
			camIdBuf[xi] = camId;
			camGroup1Buf[xi] = group1;
			camGroup2Buf[xi] = group2;
		}
		validPixelCamMap += eqrWidth;
		if (paddedPixelCamMap) paddedPixelCamMap += eqrWidth;
		camIdBuf += camIdStride;
		camGroup1Buf += (camGroup1Stride >> 1);
		camGroup2Buf += (camGroup2Stride >> 1);
	}
	return VX_SUCCESS;
}

//////////////////////////////////////////////////////////////////////
// Generate default merge mask image
vx_status GenerateDefaultMergeMaskImage(
	vx_uint32 numCamera,                  // [in] number of cameras
	vx_uint32 eqrWidth,                   // [in] output equirectangular image width
	vx_uint32 eqrHeight,                  // [in] output equirectangular image height
	const vx_uint8 * defaultCamIndex,     // [in] default camera index (255 refers to no camera): size: [eqrWidth * eqrHeight] (optional)
	vx_uint32  maskStride,                // [in] stride (in bytes) of mask image
	vx_uint8 * maskBuf                    // [out] mask image buffer: size: [eqrWidth * eqrHeight * numCamera]
	)
{
	vx_uint32 maskPosition = 0;
	for (vx_uint32 camId = 0; camId < numCamera; camId++) {
		for (vx_uint32 y = 0, pixelPosition = 0; y < eqrHeight; y++) {
			for (vx_uint32 x = 0; x < eqrWidth; x++, pixelPosition++) {
				maskBuf[maskPosition + x] = (defaultCamIndex[pixelPosition] == camId) ? 255 : 0;
			}
			maskPosition += maskStride;
		}
	}
	return VX_SUCCESS;
}
