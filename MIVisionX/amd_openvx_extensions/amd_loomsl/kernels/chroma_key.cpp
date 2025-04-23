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
#include "chroma_key.h"


/***********************************************************************************************************************************
Chroma Key Mask Generation -- CPU/GPU - Mask
************************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK chroma_key_mask_generation_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0 || index == 1)
	{//->Chroma Key
		vx_enum type = 0;	vx_uint32 value = 0;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)ref, &value));
		if (type == VX_TYPE_UINT32)
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar*)&ref));
	}
	else if (index == 2)
	{ // Image object	
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_RGB)
			status = VX_ERROR_INVALID_FORMAT;
		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;
		else if (height_img != (width_img >> 1))
			status = VX_ERROR_INVALID_DIMENSION;
		else
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image*)&ref));
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK chroma_key_mask_generation_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	if (index == 3)
	{ // Image object	
		//Query Weight Image
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_FORMAT;

		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;

		else if (height_img != ( width_img >> 1))
			status = VX_ERROR_INVALID_DIMENSION;
		else
		{
			// set output image data
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK chroma_key_mask_generation_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	char textBuffer[256];
	int CHROMAKEY_MASK = 0;
	if (StitchGetEnvironmentVariable("CHROMAKEY_MASK", textBuffer, sizeof(textBuffer))) { CHROMAKEY_MASK = atoi(textBuffer); }

	if (!CHROMAKEY_MASK)
		supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	else
		supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK chroma_key_mask_generation_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	// Get the number of work items
	vx_image input_image = (vx_image)parameters[2];
	vx_uint32 input_width = 0, input_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	vx_uint32 work_items = (vx_uint32)(input_width * input_height);
	opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK chroma_key_mask_generation_opencl_codegen(
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
	vx_image input_image = (vx_image)parameters[2];
	vx_uint32 input_width = 0, input_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));

	// set kernel configuration
	vx_uint32 work_items = (vx_uint32)(input_width*input_height);
	strcpy(opencl_kernel_function_name, "chromaKey_mask_generator");
	opencl_work_dim = 1;
	opencl_local_work[0] = 256;
	opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n" // opencl_local_work[0]
		"\n"
		"void %s(uint chroma_key, uint tol,\n"				 // opencl_kernel_function_name
		"			uint ip_rgb_width, uint ip_rgb_height, __global uchar * ip_rgb_buf, uint ip_rgb_stride, uint ip_rgb_offset,\n"
		"			uint op_u8_width, uint op_u8_height, __global uchar * op_u8_buf, uint op_u8_stride, uint op_u8_offset)\n"
		, (int)opencl_local_work[0], opencl_kernel_function_name);
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"{\n"
		"\n"
		"	int gid = get_global_id(0);\n"
		"\n"
		"	if (gid < (ip_rgb_height * ip_rgb_width))\n"
		"	{\n"
		"\n"
		"		ip_rgb_buf =  ip_rgb_buf + ip_rgb_offset;\n"
		"		op_u8_buf =  op_u8_buf + op_u8_offset;\n"
		"\n"
		"		int tola = 0, tolb = tol;\n"
		"		uchar Red_g = 0, Green_g = 0, Blue_g = 0;\n"
		"\n"
		"		// get RGB values from the key\n"
		"		Red_g = (uchar)(chroma_key & 0x000000FF);\n"
		"		Green_g = (uchar)( (chroma_key & 0x0000FF00) >> 8);\n"
		"		Blue_g = (uchar)( (chroma_key & 0x00FF0000) >> 16);\n"
		"\n"
		"		// convert RGB to yuv space key\n"
		"		int cb_key = (int)round(128 + -0.168736*Red_g - 0.331264*Green_g + 0.5*Blue_g); ;\n"
		"		int cr_key = (int)round(128 + 0.5*Red_g - 0.418688*Green_g - 0.081312*Blue_g);;\n"
		"\n"
		"		uchar3 RGB_pixel = 0;\n"
		"		uint RGB_img = *(__global uint *)&ip_rgb_buf[gid * 3];\n"
		"		// get RGB values from the pixel\n"
		"		RGB_pixel.s0 = (uchar)(RGB_img & 0x000000FF);RGB_pixel.s1 = (uchar)((RGB_img & 0x0000FF00)>> 8); RGB_pixel.s2 = (uchar)((RGB_img & 0x00FF0000)>> 16);\n"
		"\n"
		"		// convert RGB to yuv space pixel\n"
		"		int cb_p = (int)round(128 + -0.168736*RGB_pixel.s0 - 0.331264*RGB_pixel.s1 + 0.5*RGB_pixel.s2); ;\n"
		"		int cr_p = (int)round(128 + 0.5*RGB_pixel.s0 - 0.418688*RGB_pixel.s1 - 0.081312*RGB_pixel.s2);;\n"
		"\n"
		"		// check for chroma key and set mask\n"
		"		float mask = 0;\n"
		"		float temp = (float)sqrt((float)((cb_key - cb_p)*(cb_key - cb_p) + (cr_key - cr_p)*(cr_key - cr_p)));\n"
		"		if (temp < tola) { mask = 0.0; }\n"
		"		if (temp < tolb) { mask = ((temp - tola) / (tolb - tola)); }\n"
		"		else{ mask = 1.0; }\n"
		"		mask = 1 - mask;\n"
		"\n"
		"		uchar MASK_IMAGE = 0;\n"
		"		if (mask) { MASK_IMAGE = 0xFF; }\n"
		"\n"
		"		*(__global uchar *)&op_u8_buf[gid] = MASK_IMAGE;\n"
		"\n"
		"	}\n"
		"}\n";

	return VX_SUCCESS;
}

//! \brief The kernel execution on the CPU.
static vx_status VX_CALLBACK chroma_key_mask_generation_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	//Chroma Key - Variable 0
	vx_uint32 ChromaKey = 0;
	ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[0], &ChromaKey));

	//Tol - Variable 1
	vx_uint32 Tolerance = 0;
	ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)parameters[1], &Tolerance));

	//Input image - Variable 1
	vx_image input_image = (vx_image)parameters[2];
	void *input_image_ptr = nullptr; vx_rectangle_t input_rect;	vx_imagepatch_addressing_t input_addr;
	vx_uint32 input_width = 0, input_height = 0, plane = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	input_rect.start_x = input_rect.start_y = 0; input_rect.end_x = input_width; input_rect.end_y = input_height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(input_image, &input_rect, plane, &input_addr, &input_image_ptr, VX_READ_ONLY));
	vx_uint8 *input_ptr = (vx_uint8*)input_image_ptr;

	//Output Mask image - Variable 2
	vx_image output_mask_image = (vx_image)parameters[3];
	void *output_mask_image_ptr = nullptr; vx_rectangle_t output_mask_rect;	vx_imagepatch_addressing_t output_mask_addr;
	vx_uint32 output_mask_width = 0, output_mask_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage(output_mask_image, VX_IMAGE_ATTRIBUTE_WIDTH, &output_mask_width, sizeof(output_mask_width)));
	ERROR_CHECK_STATUS(vxQueryImage(output_mask_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_mask_height, sizeof(output_mask_height)));
	output_mask_rect.start_x = output_mask_rect.start_y = 0; output_mask_rect.end_x = output_mask_width; output_mask_rect.end_y = output_mask_height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(output_mask_image, &output_mask_rect, plane, &output_mask_addr, &output_mask_image_ptr, VX_WRITE_ONLY));
	vx_uint8 *output_mask_ptr = (vx_uint8*)output_mask_image_ptr;

	int tola = 0, tolb = Tolerance;
	vx_uint8 Red_g = 0, Green_g = 0, Blue_g = 0;

	Red_g = (vx_uint8)(ChromaKey & 0x000000FF);
	Green_g = (vx_uint8)((ChromaKey & 0x0000FF00) >> 8);
	Blue_g = (vx_uint8)((ChromaKey & 0x00FF0000) >> 16);

	int cb_key = (int)round(128 + -0.168736*Red_g - 0.331264*Green_g + 0.5*Blue_g);
	int cr_key = (int)round(128 + 0.5*Red_g - 0.418688*Green_g - 0.081312*Blue_g);

	for (vx_uint32 y = 0, pixelPosition = 0; y < input_height; y++){
		for (vx_uint32 x = 0; x < input_width; x++, pixelPosition++){

			double mask = 0;
			vx_uint8 Red = 0, Green = 0, Blue = 0;
			// get rgb values from the pixel
			Red = input_ptr[(pixelPosition * 3) + 0];
			Green = input_ptr[(pixelPosition * 3) + 1];
			Blue = input_ptr[(pixelPosition * 3) + 2];

			// convert rgb to yuv space
			int cb_p = (int)round(128 + -0.168736*Red - 0.331264*Green + 0.5*Blue);
			int cr_p = (int)round(128 + 0.5*Red - 0.418688*Green - 0.081312*Blue);
				
			// check for chroma key and set mask
			double temp = sqrt((cb_key - cb_p)*(cb_key - cb_p) + (cr_key - cr_p)*(cr_key - cr_p));
			if (temp < tola) { mask = 0.0; }
			if (temp < tolb) { mask = ((temp - tola) / (tolb - tola)); }
			else{ mask = 1.0; }
			mask = 1 - mask;

			if (mask){output_mask_ptr[pixelPosition] = 255;}
			else{output_mask_ptr[pixelPosition] = 0;}
		}
	}

	ERROR_CHECK_STATUS(vxCommitImagePatch(input_image, &input_rect, 0, &input_addr, input_image_ptr));
	ERROR_CHECK_STATUS(vxCommitImagePatch(output_mask_image, &output_mask_rect, 0, &output_mask_addr, output_mask_image_ptr));

	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status chroma_key_mask_generation_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.chroma_key_mask_generation",
		AMDOVX_KERNEL_STITCHING_CHROMA_KEY_MASK_GENERATION,
		chroma_key_mask_generation_kernel,
		4,
		chroma_key_mask_generation_input_validator,
		chroma_key_mask_generation_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = chroma_key_mask_generation_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = chroma_key_mask_generation_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = chroma_key_mask_generation_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));


	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

/***********************************************************************************************************************************

Chroma Key Merge -- CPU/GPU - Merge

************************************************************************************************************************************/
//! \brief The input validator callback.
static vx_status VX_CALLBACK chroma_key_merge_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0 || index == 1)
	{ // Image object	
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_RGB)
			status = VX_ERROR_INVALID_FORMAT;
		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;
		else if (height_img != (width_img >> 1))
			status = VX_ERROR_INVALID_DIMENSION;
		else
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image*)&ref));
	}
	else if (index == 2)
	{ // Image object	
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_FORMAT;
		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;
		else if (height_img != (width_img >> 1))
			status = VX_ERROR_INVALID_DIMENSION;
		else
			status = VX_SUCCESS;
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image*)&ref));
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK chroma_key_merge_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	if (index == 3)
	{ // Image object	
		//Query Weight Image
		vx_int32 width_img = 0, height_img = 0;
		vx_df_image format = VX_DF_IMAGE_VIRT;
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
		if (format != VX_DF_IMAGE_RGB)
			status = VX_ERROR_INVALID_FORMAT;

		else if (width_img < 0)
			status = VX_ERROR_INVALID_DIMENSION;

		else if (height_img != (width_img >> 1))
			status = VX_ERROR_INVALID_DIMENSION;
		else
		{
			// set output image data
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width_img, sizeof(width_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height_img, sizeof(height_img)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK chroma_key_merge_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	char textBuffer[256];
	int CHROMAKEY_MERGE = 0;
	if (StitchGetEnvironmentVariable("CHROMAKEY_MERGE", textBuffer, sizeof(textBuffer))) { CHROMAKEY_MERGE = atoi(textBuffer); }

	if (!CHROMAKEY_MERGE)
		supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	else
		supported_target_affinity = AGO_TARGET_AFFINITY_CPU;

	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK chroma_key_merge_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	// Get the number of work items
	vx_image input_image = (vx_image)parameters[1];
	vx_uint32 input_width = 0, input_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	vx_uint32 work_items = (vx_uint32)(input_width * input_height);
	opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK chroma_key_merge_opencl_codegen(
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
	vx_image input_image = (vx_image)parameters[1];
	vx_uint32 input_width = 0, input_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));

	// set kernel configuration
	vx_uint32 work_items = (vx_uint32)(input_width*input_height);
	strcpy(opencl_kernel_function_name, "chromaKey_merge");
	opencl_work_dim = 1;
	opencl_local_work[0] = 256;
	opencl_global_work[0] = (work_items + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

	// Setting variables required by the interface
	opencl_local_buffer_usage_mask = 0;
	opencl_local_buffer_size_in_bytes = 0;

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n" // opencl_local_work[0]
		"\n"
		"void %s(\n"				 // opencl_kernel_function_name
		"			uint ip_rgb_width, uint ip_rgb_height, __global uchar * ip_rgb_buf, uint ip_rgb_stride, uint ip_rgb_offset,\n"
		"			uint ip_chr_width, uint ip_chr_height, __global uchar * ip_chr_buf, uint ip_chr_stride, uint ip_chr_offset,\n"
		"			uint ip_u8_width, uint ip_u8_height, __global uchar * ip_u8_buf, uint op_u8_stride, uint ip_u8_offset,\n"
		"			uint op_width, uint op_height, __global uchar * op_buf, uint op_stride, uint op_offset)\n"
		, (int)opencl_local_work[0], opencl_kernel_function_name);
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"{\n"
		"\n"
		"	int gid = get_global_id(0);\n"
		"\n"
		"	if (gid < (ip_rgb_height * ip_rgb_width))\n"
		"	{\n"
		"\n"
		"		ip_rgb_buf =  ip_rgb_buf + ip_rgb_offset;\n"
		"		ip_chr_buf =  ip_chr_buf + ip_chr_offset;\n"
		"		ip_u8_buf =  ip_u8_buf + ip_u8_offset;\n"
		"		op_buf =  op_buf + op_offset;\n"
		"\n"
		"		uchar mask_img  = *(__global uchar *)&ip_u8_buf[gid];\n"
		"\n"
		"		if(!(mask_img))\n"
		"		{\n"
		"			uchar3 RGB_pixel = 0;\n"
		"			uint RGB_img = *(__global uint *)&ip_rgb_buf[gid * 3];\n"
		"			// get RGB values from the pixel\n"
		"			RGB_pixel.s0 = (uchar)(RGB_img & 0x000000FF);RGB_pixel.s1 = (uchar)((RGB_img & 0x0000FF00)>> 8); RGB_pixel.s2 = (uchar)((RGB_img & 0x00FF0000)>> 16);\n"
		"			*(__global uchar2 *)&op_buf[gid * 3] = RGB_pixel.s01; *(__global uchar *)&op_buf[(gid * 3) + 2] = RGB_pixel.s2;\n"
		"		}\n"
		"		else\n"
		"		{\n"
		"			uchar3 RGB_pixel = 0;\n"
		"			uint RGB_img = *(__global uint *)&ip_chr_buf[gid * 3];\n"
		"			// get RGB values from the pixel\n"
		"			RGB_pixel.s0 = (uchar)(RGB_img & 0x000000FF); RGB_pixel.s1 = (uchar)((RGB_img & 0x0000FF00)>> 8); RGB_pixel.s2 = (uchar)((RGB_img & 0x00FF0000)>> 16);\n"
		"			*(__global uchar2 *)&op_buf[gid * 3] = RGB_pixel.s01; *(__global uchar *)&op_buf[(gid * 3) + 2] = RGB_pixel.s2;\n"
		"		}\n"
		"\n"
		"	}\n"
		"}\n";

	return VX_SUCCESS;
}

//! \brief The kernel execution on the CPU.
static vx_status VX_CALLBACK chroma_key_merge_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	// Input stitched image - Variable 1
	vx_image input_RGB_image = (vx_image)parameters[0];
	void *input_RGB_ptr = nullptr; vx_rectangle_t input_RGB_rect;	vx_imagepatch_addressing_t input_RGB_addr;
	vx_uint32 input_RGB_width = 0, input_RGB_height = 0, plane = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_RGB_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_RGB_width, sizeof(input_RGB_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_RGB_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_RGB_height, sizeof(input_RGB_height)));
	input_RGB_rect.start_x = input_RGB_rect.start_y = 0; input_RGB_rect.end_x = input_RGB_width; input_RGB_rect.end_y = input_RGB_height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(input_RGB_image, &input_RGB_rect, plane, &input_RGB_addr, &input_RGB_ptr, VX_READ_ONLY));
	vx_uint8 *input_RGB_image_ptr = (vx_uint8*)input_RGB_ptr;

	// Input chroma image - Variable 1
	vx_image input_chroma_image = (vx_image)parameters[1];
	void *input_chroma_ptr = nullptr; vx_rectangle_t input_chroma_rect;	vx_imagepatch_addressing_t input_chroma_addr;
	vx_uint32 input_chroma_width = 0, input_chroma_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_chroma_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_chroma_width, sizeof(input_chroma_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_chroma_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_chroma_height, sizeof(input_chroma_height)));
	input_chroma_rect.start_x = input_chroma_rect.start_y = 0; input_chroma_rect.end_x = input_chroma_width; input_chroma_rect.end_y = input_chroma_height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(input_chroma_image, &input_chroma_rect, plane, &input_chroma_addr, &input_chroma_ptr, VX_READ_ONLY));
	vx_uint8 *input_chroma_image_ptr = (vx_uint8*)input_chroma_ptr;

	// Input Mask image - Variable 2
	vx_image input_mask_image = (vx_image)parameters[2];
	void *input_mask_image_ptr = nullptr; vx_rectangle_t input_mask_rect;	vx_imagepatch_addressing_t input_mask_addr;
	vx_uint32 input_mask_width = 0, input_mask_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage(input_mask_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_mask_width, sizeof(input_mask_width)));
	ERROR_CHECK_STATUS(vxQueryImage(input_mask_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_mask_height, sizeof(input_mask_height)));
	input_mask_rect.start_x = input_mask_rect.start_y = 0; input_mask_rect.end_x = input_mask_width; input_mask_rect.end_y = input_mask_height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(input_mask_image, &input_mask_rect, plane, &input_mask_addr, &input_mask_image_ptr, VX_READ_ONLY));
	vx_uint8 *input_mask_ptr = (vx_uint8*)input_mask_image_ptr;

	// Output merged image - Variable 3
	vx_image output_image = (vx_image)parameters[3];
	void * output_image_ptr = nullptr; vx_rectangle_t  output_rect;	vx_imagepatch_addressing_t  output_addr;
	vx_uint32  output_width = 0, output_height = 0;
	ERROR_CHECK_STATUS(vxQueryImage(output_image, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
	ERROR_CHECK_STATUS(vxQueryImage(output_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
	output_rect.start_x = output_rect.start_y = 0;  output_rect.end_x = output_width;  output_rect.end_y = output_height;
	ERROR_CHECK_STATUS(vxAccessImagePatch(output_image, &output_rect, plane, &output_addr, &output_image_ptr, VX_WRITE_ONLY));
	vx_uint8 * output_ptr = (vx_uint8*)output_image_ptr;


	for (vx_uint32 y = 0, pixelPosition = 0; y < input_RGB_height; y++){
		for (vx_uint32 x = 0; x < input_RGB_width; x++, pixelPosition++){
			if (input_mask_ptr[pixelPosition] == 255){
				output_ptr[pixelPosition * 3 + 0] = input_chroma_image_ptr[pixelPosition * 3 + 0];
				output_ptr[pixelPosition * 3 + 1] = input_chroma_image_ptr[pixelPosition * 3 + 1];
				output_ptr[pixelPosition * 3 + 2] = input_chroma_image_ptr[pixelPosition * 3 + 2];
			}
			else{
				output_ptr[pixelPosition * 3 + 0] = input_RGB_image_ptr[pixelPosition * 3 + 0];
				output_ptr[pixelPosition * 3 + 1] = input_RGB_image_ptr[pixelPosition * 3 + 1];
				output_ptr[pixelPosition * 3 + 2] = input_RGB_image_ptr[pixelPosition * 3 + 2];
			}
		}
	}

	ERROR_CHECK_STATUS(vxCommitImagePatch(input_RGB_image, &input_RGB_rect, 0, &input_RGB_addr, input_RGB_ptr));
	ERROR_CHECK_STATUS(vxCommitImagePatch(input_chroma_image, &input_chroma_rect, 0, &input_chroma_addr, input_chroma_ptr));
	ERROR_CHECK_STATUS(vxCommitImagePatch(input_mask_image, &input_mask_rect, 0, &input_mask_addr, input_mask_image_ptr));
	ERROR_CHECK_STATUS(vxCommitImagePatch(output_image, &output_rect, 0, &output_addr, output_image_ptr));


	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status chroma_key_merge_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.chroma_key_merge",
		AMDOVX_KERNEL_STITCHING_CHROMA_KEY_MERGE,
		chroma_key_merge_kernel,
		4,
		chroma_key_merge_input_validator,
		chroma_key_merge_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = chroma_key_merge_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = chroma_key_merge_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = chroma_key_merge_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));


	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}
