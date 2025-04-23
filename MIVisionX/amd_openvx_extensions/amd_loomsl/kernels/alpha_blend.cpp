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
#include "alpha_blend.h"

//! \brief The input validator callback.
static vx_status VX_CALLBACK validate(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
	if (num != 3)
		return VX_ERROR_INVALID_PARAMETERS;
	vx_uint32 width, height, width2, height2;
	vx_df_image format, format2;
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width2, sizeof(width2)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_HEIGHT, &height2, sizeof(height2)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_FORMAT, &format2, sizeof(format2)));
	if (format != VX_DF_IMAGE_RGB || format2 != VX_DF_IMAGE_RGBX)
		return VX_ERROR_INVALID_FORMAT;
	if (width != width2 || height != height2)
		return VX_ERROR_INVALID_DIMENSION;
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_FORMAT, &format, sizeof(format)));
	return VX_SUCCESS;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK opencl_codegen(
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
	// get image dimensions
	vx_uint32 width, height;
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[2], VX_IMAGE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[2], VX_IMAGE_HEIGHT, &height, sizeof(height)));
	// set kernel configuration
	strcpy(opencl_kernel_function_name, "alpha_blend");
	vx_uint32 work_items[2] = { (vx_uint32)((width + 3) >> 2), (vx_uint32)height };
	opencl_work_dim = 2;
	opencl_local_work[0] = 8;
	opencl_local_work[1] = 8;
	opencl_global_work[0] = (work_items[0] + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
	opencl_global_work[1] = (work_items[1] + opencl_local_work[1] - 1) & ~(opencl_local_work[1] - 1);
	
	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

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
		"void %s(uint i0_width, uint i0_height, __global uchar * i0_buf, uint i0_stride, uint i0_offset,\n"
		"        uint i1_width, uint i1_height, __global uchar * i1_buf, uint i1_stride, uint i1_offset,\n"
		"        uint o0_width, uint o0_height, __global uchar * o0_buf, uint o0_stride, uint o0_offset)\n"
		"{\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  if ((gx < %d) && (gy < %d)) {\n" // work_items[0], work_items[1]
		"    uint3 i0 = *(__global uint3 *) (i0_buf + i0_offset + (gy * i0_stride) + (gx * 12));\n"
		"    uint4 i1 = *(__global uint4 *) (i1_buf + i1_offset + (gy * i1_stride) + (gx * 16));\n"
		"    uint3 o0;\n"
		"    float4 f; float alpha0, alpha1, alpha_normalizer = 0.0039215686274509803921568627451f;\n"
		"    alpha1 = amd_unpack3(i1.s0)*alpha_normalizer; alpha0 = 1.0f - alpha1;\n"
		"    f.s0 = mad(amd_unpack0(i0.s0), alpha0, amd_unpack0(i1.s0)*alpha1);\n"
		"    f.s1 = mad(amd_unpack1(i0.s0), alpha0, amd_unpack1(i1.s0)*alpha1);\n"
		"    f.s2 = mad(amd_unpack2(i0.s0), alpha0, amd_unpack2(i1.s0)*alpha1);\n"
		"    alpha1 = amd_unpack3(i1.s1)*alpha_normalizer; alpha0 = 1.0f - alpha1;\n"
		"    f.s3 = mad(amd_unpack3(i0.s0), alpha0, amd_unpack0(i1.s1)*alpha1);\n"
		"    o0.s0 = amd_pack(f);\n"
		"    f.s0 = mad(amd_unpack0(i0.s1), alpha0, amd_unpack1(i1.s1)*alpha1);\n"
		"    f.s1 = mad(amd_unpack1(i0.s1), alpha0, amd_unpack2(i1.s1)*alpha1);\n"
		"    alpha1 = amd_unpack3(i1.s2)*alpha_normalizer; alpha0 = 1.0f - alpha1;\n"
		"    f.s2 = mad(amd_unpack2(i0.s1), alpha0, amd_unpack0(i1.s2)*alpha1);\n"
		"    f.s3 = mad(amd_unpack3(i0.s1), alpha0, amd_unpack1(i1.s2)*alpha1);\n"
		"    o0.s1 = amd_pack(f);\n"
		"    f.s0 = mad(amd_unpack0(i0.s2), alpha0, amd_unpack2(i1.s2)*alpha1);\n"
		"    alpha1 = amd_unpack3(i1.s3)*alpha_normalizer; alpha0 = 1.0f - alpha1;\n"
		"    f.s1 = mad(amd_unpack1(i0.s2), alpha0, amd_unpack0(i1.s3)*alpha1);\n"
		"    f.s2 = mad(amd_unpack2(i0.s2), alpha0, amd_unpack1(i1.s3)*alpha1);\n"
		"    f.s3 = mad(amd_unpack3(i0.s2), alpha0, amd_unpack2(i1.s3)*alpha1);\n"
		"    o0.s2 = amd_pack(f);\n"
		"    *(__global uint3 *) (o0_buf + o0_offset + (gy * o0_stride) + (gx * 12)) = o0;\n"
		"  }\n"
		"}\n"
		, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, work_items[0], work_items[1]);
	opencl_kernel_code = item;

	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status alpha_blend_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddUserKernel(context, "com.amd.loomsl.alpha_blend", AMDOVX_KERNEL_STITCHING_ALPHA_BLEND, host_kernel, 2, validate, nullptr, nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}
