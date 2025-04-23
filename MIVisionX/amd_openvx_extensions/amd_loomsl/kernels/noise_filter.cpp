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
#include "noise_filter.h"

//! \brief The input validator callback.
static vx_status VX_CALLBACK noise_filter_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;

	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
		if (itemtype == VX_TYPE_FLOAT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: noise filter lambda scalar type should be a vx_float32\n");
		}
	}
	if (index == 1)
	{ // image of format RGB
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		if (format == VX_DF_IMAGE_RGB) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: noise_filter doesn't support input image format: %4.4s\n", &format);
		}
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
	}
	if (index == 2)
	{ // image of format RGB
		vx_df_image format = VX_DF_IMAGE_VIRT;
		vx_uint32 width1 = 0, width2 = 0, height1 = 0, height2 = 0;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width2, sizeof(width2)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height2, sizeof(height2)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		ref = avxGetNodeParamRef(node, 1);
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width1, sizeof(width1)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height1, sizeof(height1)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (format == VX_DF_IMAGE_RGB) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: noise_filter doesn't support input image format: %4.4s\n", &format);
		}
		if (!((width1 == width2) && (height1 == height2))) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: noise_filter input images must be of same dimensions\n");
		}
		else {
			status = VX_SUCCESS;
		}
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK noise_filter_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;

	if (index == 3)
	{ // image of format RGB
		// get image configuration
		vx_image image = (vx_image)avxGetNodeParamRef(node, 1);
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
		if (!(output_width == input_width && output_height == input_height))
		{
			// pick input dimensions as default
			output_width = input_width;
			output_height = input_height;
		}
		if (output_format != VX_DF_IMAGE_RGB) {
			// pick RGB as default
			output_format = VX_DF_IMAGE_RGBX;
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
static vx_status VX_CALLBACK noise_filter_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK noise_filter_opencl_codegen(
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
	// get image configurations
	vx_uint32 width = 0, height = 0;
	vx_float32 lambda = 0.0f;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &lambda));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 1);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));

	// set kernel configuration
	vx_uint32 work_items[2] = { (width + 3) >> 2, height };
	strcpy(opencl_kernel_function_name, "noise_filter");
	opencl_work_dim = 2;
	opencl_local_work[0] = 16;
	opencl_local_work[1] = 16;
	opencl_global_work[0] = (work_items[0] + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
	opencl_global_work[1] = (work_items[1] + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);

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
		"void %s(float lambda,\n" // opencl_kernel_function_name
		"        uint ip0_width, uint ip0_height, __global uchar * ip0_buf, uint ip0_stride, uint ip0_offset,\n"
		"        uint ip1_width, uint ip1_height, __global uchar * ip1_buf, uint ip1_stride, uint ip1_offset,\n"
		"        uint op_width, uint op_height, __global uchar * op_buf, uint op_stride, uint op_offset)\n"
		"{\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  if ((gx < %d) && (gy < %d)) {\n" // work_items[0], work_items[1]
		"    uint3 pix0 = *(__global uint3 *) (ip0_buf + ip0_offset + (gy * ip0_stride) + (gx * 12));\n"
		"    uint3 pix1 = *(__global uint3 *) (ip1_buf + ip1_offset + (gy * ip1_stride) + (gx * 12));\n"
		"    uint3 outpix;\n"
		"    float4 f;\n"
		"    float oneMinusLambda = 1.0f - lambda;"
		"    f = mad(amd_unpack(pix0.s0), (float4)lambda, amd_unpack(pix1.s0) * (float4)oneMinusLambda);  outpix.s0 = amd_pack(f);\n"
		"    f = mad(amd_unpack(pix0.s1), (float4)lambda, amd_unpack(pix1.s1) * (float4)oneMinusLambda);  outpix.s1 = amd_pack(f);\n"
		"    f = mad(amd_unpack(pix0.s2), (float4)lambda, amd_unpack(pix1.s2) * (float4)oneMinusLambda);  outpix.s2 = amd_pack(f);\n"
		"    *(__global uint3 *) (op_buf + op_offset + (gy * op_stride) + (gx * 12)) = outpix;\n"
		"  }\n"
		"}\n"
		, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, work_items[0], work_items[1]);
	opencl_kernel_code = item;
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK noise_filter_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}

//! \brief The kernel publisher.
vx_status noise_filter_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.noise_filter",
		AMDOVX_KERNEL_STITCHING_NOISE_FILTER,
		noise_filter_kernel,
		4,
		noise_filter_input_validator,
		noise_filter_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = noise_filter_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = noise_filter_opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}
