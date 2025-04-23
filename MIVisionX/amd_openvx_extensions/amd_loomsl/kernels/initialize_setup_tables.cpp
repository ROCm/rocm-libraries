/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include "kernels.h"
#include "lens_distortion_remap.h"

//! \brief The input validator callback.
static vx_status VX_CALLBACK calc_lens_distortionwarp_map_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	if (index < 5)
	{
		// scalar of VX_TYPE_UINT32
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: calc_lens_distortionwarp_map par%d should be UINT32 type\n", index);
		}
	}
	/*
	else if ((index > 6) && (index < 12))
	{
		// scalar of VX_TYPE_UINT32
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		if (itemtype == VX_TYPE_FLOAT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: calc_lens_distortionwarp_map par%d should be UINT32 type\n", index);
		}
	}
	*/
	else if (index == 5)
	{
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(vx_float32)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: calc_lens_distortionwarp_map array element size should be 4 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return status;
}


//! \brief The output validator callback.
static vx_status VX_CALLBACK calc_lens_distortionwarp_map_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if ((index == 6) || (index == 7) || (index == 8))
	{
		vx_reference ref = avxGetNodeParamRef(node, index);
		if (ref){
			// check input image format and dimensions
			vx_uint32 width = 0, height = 0;
			vx_df_image format = VX_DF_IMAGE_VIRT;
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(height)));
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
			// set output image meta data
			if (format != VX_DF_IMAGE_U32)
				format = VX_DF_IMAGE_U32;
            ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
			ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		}
		status = VX_SUCCESS;
	}
	if (index > 8)
	{ 
		vx_enum itemtype = VX_TYPE_INVALID;
		vx_size capacity = 0; vx_size itemsize = 0;
		vx_array arr = (vx_array)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(arr);
		ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
		ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(vx_float32)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: calc_lens_distortionwarp_map array element size should be float\n");
		}
		// set output image meta data
		vx_uint32 nCam = 0;
		vx_uint32 width, height;
		vx_image image = (vx_image)avxGetNodeParamRef(node, 6);
		ERROR_CHECK_OBJECT(image);
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);
		ERROR_CHECK_OBJECT(scalar);
		ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &nCam));
		ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));

		if (capacity < ((width + 3)&~3)*height*nCam)
		{
			capacity = ((width + 3)&~3)*height*nCam;
		}
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
		status = VX_SUCCESS;
	}
	return status;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK calc_lens_distortionwarp_map_opencl_codegen(
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
	vx_uint32 nCam = 0; vx_uint32 lens_type = 0;
	vx_uint32 in_width = 0, in_height = 0;
	vx_uint32 out_width = 0, out_height = 0;
	vx_df_image format = VX_DF_IMAGE_VIRT;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);		
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &nCam));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	scalar = (vx_scalar)avxGetNodeParamRef(node, 1);
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &lens_type));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	scalar = (vx_scalar)avxGetNodeParamRef(node, 2);
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &in_width));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	scalar = (vx_scalar)avxGetNodeParamRef(node, 3);	
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &in_height));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 6);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &out_width, sizeof(out_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &out_height, sizeof(out_height)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	vx_image pad_img = (vx_image)avxGetNodeParamRef(node, 7);
	if (pad_img){
		vx_uint32 pad_width = 0, pad_height = 0;
		ERROR_CHECK_OBJECT(pad_img);
		ERROR_CHECK_STATUS(vxQueryImage(pad_img, VX_IMAGE_ATTRIBUTE_WIDTH, &pad_width, sizeof(pad_width)));
		ERROR_CHECK_STATUS(vxQueryImage(pad_img, VX_IMAGE_ATTRIBUTE_HEIGHT, &pad_height, sizeof(pad_height)));
	}

	double pibyH = (double)M_PI / (float)out_height;
	float halfW = (float)in_width * 0.5f;
	float halfH = (float)in_height * 0.5f;
    // set kernel configuration
	strcpy(opencl_kernel_function_name, "calc_lens_distortion_and_warp_map");
	opencl_work_dim = 2;
	opencl_local_work[0] = 8;
	opencl_local_work[1] = 8;
	opencl_global_work[0] = (((out_width + 3) >> 2) + 7) & ~7;
	opencl_global_work[1] = (out_height + 7) & ~7;

	char item[8192];
	if (pad_img){
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"\n"
			"float4 lens_model_function(float4 th, float fr, float4 abcd, float lens_type)\n"
			"{\n"
			"	float4 r;\n"
			"	if (!lens_type){\n"
			"		r = tan(th) * (float4)fr;\n"
			"		return (r * ((float4)abcd.s3 + r * ((float4)abcd.s2 + r * ((float4)abcd.s1 + r * (float4)abcd.s0))));\n"
			"	}\n"
			"	else if (lens_type < 3){\n"
			"		r = th * (float4)fr;\n"
			"		return (r * ((float4)abcd.s3 + r * ((float4)abcd.s2 + r * ((float4)abcd.s1 + r * (float4)abcd.s0))));		\n"
			"	}\n"
			"	else if (lens_type == 3)\n"
			"	{\n"
			"		r = tan(th) * (float4)fr; \n"
			"		float4 r2 = r*r;\n"
			"		return (r * ((float4)1.f + r2* ((float4)abcd.s0 + r2 * ((float4)abcd.s1 + r2 * (float4)abcd.s2))));\n"
			"	}else\n"
			"	{\n"
			"		float4 r = th * (float4)fr;\n"
			"		float4 r2 = r*r;\n"
			" 		return (r * ((float4)1.f + r2 * ((float4)abcd.s0 + r2 * (float4)abcd.s1)));\n"
			"	}\n"
			"}\n"
			"\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n" // opencl_local_work[0]
			"\n"
			" void %s(      \n"
			"			uint ncam, int l_type,\n"
			"			uint camWidth, uint camHeight, uint paddingPixelCount,\n"
			"		    __global uchar * cam_params, uint camera_params_offs, uint camera_params_num,\n"
			"		    uint vm_width, uint	vm_height, __global uchar *valid_pix_map, uint vm_stride, uint	vm_offs,\n"
			"		    uint pm_width, uint	pm_height, __global uchar * padded_pix_map, uint pm_stride, uint padded_pix_map_offset,\n"
			"		    uint sc_width, uint	sc_height, __global uchar * camera_src_coord_map, uint sc_stride, uint camera_src_coord_map_offs,\n"
			"		    __global uchar * camera_z_value_buf, uint camera_z_value_buf_offs, uint zbuf_num)\n"
			"{\n"
			"	int gx = get_global_id(0);\n"
			"	int gy = get_global_id(1);\n"
			"	float pibyH = %.15le; \n"
			"	cam_params += camera_params_offs; camera_src_coord_map += camera_src_coord_map_offs; \n"
			"	gx <<= 2;\n"
			"	if( gx < vm_width && gy < vm_height){\n"
			"	camera_z_value_buf += camera_z_value_buf_offs + ((gy*vm_width + gx)<<2);\n"
			"	valid_pix_map += vm_offs + gy*vm_stride + (gx << 2);\n"
			"	uint4 valid_pix_out = (uint4) 0;\n"
			"	padded_pix_map += padded_pix_map_offset + (gy*pm_stride) + (gx<<2);\n"
			"	uint4 padded_pix_out = (uint4)0;\n"
			"	for (int camId=0; camId < %d; camId++) {\n"
			"		__global float * cam_params_cur = (__global float *)(cam_params + camId*128);\n"
			"		float4 cam_ltrb = *(__global float4*)(cam_params_cur);\n"
			"		float4 cam_k1k2k3k0 = *(__global float4*)(cam_params_cur+4);\n"
			"		float2 cam_du0dv0 = *(__global float2*)(cam_params_cur+8);\n"
			"		float r_crop = *(__global float*)(cam_params_cur + 10); \n"
			"		float F0 = *(__global float*)(cam_params_cur+11);\n"
			"		float4 F1T0T1T2 = *(__global float4*)(cam_params_cur+12);\n"
			"		__global float * Mcam = (__global float*)(cam_params_cur+16);\n"
			"		float lens_type = *(__global float*)(cam_params_cur+25);\n"
			"		float2 center = cam_du0dv0 +  (float2)(%f, %f);\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, pibyH, nCam, halfW, halfH);
	}
	else
	{
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"\n"
			"inline float4 lens_model_function(float4 th, float fr, float4 abcd, float lens_type)\n"
			"{\n"
			"	float4 r;\n"
			"	if (!lens_type){\n"
			"		r = tan(th) * (float4)fr;\n"
			"		return (r * ((float4)abcd.s3 + r * ((float4)abcd.s2 + r * ((float4)abcd.s1 + r * (float4)abcd.s0))));\n"
			"	}\n"
			"	else if (lens_type < 3){\n"
			"		r = th * (float4)fr;\n"
			"		return (r * ((float4)abcd.s3 + r * ((float4)abcd.s2 + r * ((float4)abcd.s1 + r * (float4)abcd.s0))));		\n"
			"	}\n"
			"	else if (lens_type == 3)\n"
			"	{\n"
			"		r = tan(th) * (float4)fr; \n"
			"		float4 r2 = r*r;\n"
			"		return (r * ((float4)1.f + r2* ((float4)abcd.s0 + r2 * ((float4)abcd.s1 + r2 * (float4)abcd.s2))));\n"
			"	}else\n"
			"	{\n"
			"		float4 r = th * (float4)fr;\n"
			"		float4 r2 = r*r;\n"
			" 		return (r * ((float4)1.f + r2 * ((float4)abcd.s0 + r2 * (float4)abcd.s1)));\n"
			"	}\n"
			"}\n"
			"\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n" // opencl_local_work[0]
			"\n"
			" void %s(      \n"
			"			uint ncam, int l_type,\n"
			"			uint camWidth, uint camHeight, uint paddingPixelCount,\n"
			"		    __global uchar * cam_params, uint camera_params_offs, uint camera_params_num,\n"
			"		    uint vm_width, uint	vm_height, __global uchar *valid_pix_map, uint vm_stride, uint	vm_offs,\n"
			"		    uint sc_width, uint	sc_height, __global uchar * camera_src_coord_map, uint sc_stride, uint camera_src_coord_map_offs,\n"
			"		    __global uchar * camera_z_value_buf, uint camera_z_value_buf_offs, uint zbuf_num)\n"
			"{\n"
			"	int gx = get_global_id(0);\n"
			"	int gy = get_global_id(1);\n"
			"	float pibyH = %.15le; \n"
			"	cam_params += camera_params_offs; camera_src_coord_map += camera_src_coord_map_offs; \n"
			"	gx <<= 2;\n"
			"	if( gx < vm_width && gy < vm_height){\n"
			"	camera_z_value_buf += camera_z_value_buf_offs + ((gy*vm_width + gx)<<2);\n"
			"	valid_pix_map += vm_offs + gy*vm_stride + (gx << 2);\n"
			"	uint4 valid_pix_out = (uint4) 0;\n"
			"	for (int camId=0; camId < %d; camId++) {\n"
			"		__global float * cam_params_cur = (__global float *)(cam_params + camId*128);\n"
			"		float4 cam_ltrb = *(__global float4*)(cam_params_cur);\n"
			"		float4 cam_k1k2k3k0 = *(__global float4*)(cam_params_cur+4);\n"
			"		float2 cam_du0dv0 = *(__global float2*)(cam_params_cur+8);\n"
			"		float r_crop = *(__global float*)(cam_params_cur+10);\n"
			"		float F0 = *(__global float*)(cam_params_cur+11);\n"
			"		float4 F1T0T1T2 = *(__global float4*)(cam_params_cur+12);\n"
			"		__global float * Mcam = (__global float*)(cam_params_cur+16);\n"
			"		float lens_type = *(__global float*)(cam_params_cur+25);\n"
			"		float2 center = cam_du0dv0 +  (float2)(%f, %f);\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, pibyH, nCam, halfW, halfH);
	}
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"\n"
		"		uint2 size = (uint2)((uint)(camId*vm_height*vm_width), (uint)(camId*sc_stride*vm_height));\n"
		"		__global float *camera_z_value_ptr = (__global float *)(camera_z_value_buf + size.x*4);\n"
		"		__global float *camera_src_coord = (__global float *)(camera_src_coord_map + size.y + gy*sc_stride);\n"
		"		float4 x0, x1, x2, x_src, y_src;\n"
		"		float4 y0, y1, y2, te;\n"
		"		te = (float4)((float)gx, (float)(gx+1), (float)(gx+2), (float)(gx+3));\n"
		"		float pe = gy*pibyH - 1.57079632679489661923f /* M_PI_2 */;\n"
		"		te = te*(float4)pibyH - (float4)3.14159265358979323846f /* M_PI */;\n"
		"		float sin_pe = sin(pe), cos_pe = cos(pe);\n"
		"		x0 = sin(te)*(float4)cos_pe; x0 -= (float4) F1T0T1T2.s1;\n"
		"		x2 = cos(te)*(float4)cos_pe; x2 -= (float4) F1T0T1T2.s3;\n"
		"		x1 = (float4)(sin_pe - F1T0T1T2.s2);    \n"
		"		y0 = (float4)1.f / (float4)sqrt(x0*x0 + x1*x1 + x2*x2);		// mul_factor\n"
		"		x0 *= y0; x1 *= y0; x2 *= y0;\n"
		"		// compute mat_mult output\n"
		"		y0 = x0*(float4)Mcam[0] + x1*(float4)Mcam[1] + x2*(float4)Mcam[2];\n"
		"		y1 = x0*(float4)Mcam[3] + x1*(float4)Mcam[4] + x2*(float4)Mcam[5];\n"
		"		y2 = x0*(float4)Mcam[6] + x1*(float4)Mcam[7] + x2*(float4)Mcam[8];\n"
		"		// calculate src coordinates\n"
		"		te = asin(sqrt(min(max((y0*y0 + y1*y1), (float4)0.f), (float4)1.0f)));\n"
		"		y1 = atan2(y1, y0);\n"
		"		x0 = lens_model_function(te, F0, cam_k1k2k3k0, lens_type);\n"
		"		x_src = (float4)F1T0T1T2.s0*x0*cos(y1);\n"
		"		y_src = (float4)F1T0T1T2.s0*x0*sin(y1);\n"
		"		te = sqrt(x_src*x_src + y_src*y_src);			// rr\n"
		"		x_src += (float4) center.x; y_src += (float4) center.y;\n"
		"		float2 rbminus1 = cam_ltrb.s23 - (float2) 1.f;\n"
		"		int4 isValidCamMap = select((int4)0, (int4)1,  ((y2 > (float4)0.0f) && (x_src >= (float4)cam_ltrb.s0) && (x_src <= (float4)rbminus1.s0)));\n"
		"		isValidCamMap &=  select((int4)0, (int4)1, ((y_src >= (float4)cam_ltrb.s1) && (y_src <= (float4)rbminus1.s1)));\n"
		"		isValidCamMap &= select((int4)0, (int4)1, (((float4)r_crop <= (float4)0.0f) || (te <= (float4)r_crop)));\n"
		"		valid_pix_out |= convert_uint4(isValidCamMap << camId);\n"
		"		// update zbuffer\n"
		"		*(__global float4 *)camera_z_value_ptr = select((float4)0.f, fabs(y2), (isValidCamMap != (int4)0) );\n"
		"		int4 isPaddingCamMap = (int4)0;\n"
		"		if (paddingPixelCount){\n"
		"			float4 cam_padltrb;\n"
		"			cam_padltrb.s01 = cam_ltrb.s01 - (float2)paddingPixelCount;\n"
		"			cam_padltrb.s23 = rbminus1 + (float2)paddingPixelCount;\n"
		"			isPaddingCamMap = select(isPaddingCamMap, (int4)1, ((y2 > (float4)0.0f) && (x_src >= (float4)cam_padltrb.s0) && (x_src <= (float4)cam_padltrb.s2)));\n"
		"			isPaddingCamMap &= select((int4)0, (int4)1, ((y_src >= (float4)cam_padltrb.s1) && (y_src <= (float4)cam_padltrb.s3)));\n"
		"			isPaddingCamMap &= select((int4)0, (int4)1, ((float4)r_crop <= (float4)0.0f) || (te <= (float4)(r_crop + paddingPixelCount)));\n"
		"		}\n";
		if (pad_img){
			opencl_kernel_code +=
				"		isPaddingCamMap &= ((~isValidCamMap) & (int4)(lens_type != 2));\n"
				"		padded_pix_out |= convert_uint4(isPaddingCamMap << camId);\n";
		}
		opencl_kernel_code +=
			"		x2 = convert_float4(isValidCamMap|isPaddingCamMap);\n"
			"		x_src = select((float4)-1.f, x_src, (x2 != (float4)0.0f) );\n"
			"		y_src = select((float4)-1.f, y_src, (x2 != (float4)0.0f) );\n"
			"		float2 rbminus2 = rbminus1 * (float2)2.f;\n"
			"		x_src = select(x_src, (float4)cam_ltrb.s0 - x_src, (x_src < (float4)cam_ltrb.s0));\n"
			"		x_src = select(x_src, (float4)rbminus2.s0 - x_src, (x_src >= (float4)rbminus1.s0));\n"
			"		y_src = select(y_src, (float4)cam_ltrb.s1 - y_src, (y_src < (float4)cam_ltrb.s1));\n"
			"		y_src = select(y_src, (float4)rbminus2.s1 - y_src, (y_src >= (float4)rbminus1.s1));\n"
			"		camera_src_coord += (gx<<1);\n"
			"		*(__global float4 *)camera_src_coord = (float4) (x_src.s0, y_src.s0, x_src.s1, y_src.s1);\n"
			"		*(__global float4 *)(camera_src_coord + 4) = (float4) (x_src.s2, y_src.s2, x_src.s3, y_src.s3);\n"
			"	}\n"
			"	*(__global uint4 *)valid_pix_map = valid_pix_out;\n";
		if (pad_img){
			opencl_kernel_code +=
			"	*(__global uint4 *)padded_pix_map = padded_pix_out;\n";
		}
		opencl_kernel_code +=
		" }\n"
		"}\n";
	if (pad_img){
		ERROR_CHECK_STATUS(vxReleaseImage(&pad_img));
	}

	return VX_SUCCESS;
}


//! \brief The kernel execution.
static vx_status VX_CALLBACK calc_lens_distortionwarp_map_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK calc_lens_distortionwarp_map_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}


//! \brief The kernel publisher.
vx_status calc_lens_distortionwarp_map_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.calc_lens_distortionwarp_map",
		AMDOVX_KERNEL_STITCHING_INIT_CALC_CAMERA_VALID_MAP,
		calc_lens_distortionwarp_map_kernel,
		10,
		calc_lens_distortionwarp_map_input_validator,
		calc_lens_distortionwarp_map_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = calc_lens_distortionwarp_map_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = calc_lens_distortionwarp_map_opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));

	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 7, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_OPTIONAL));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 8, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 9, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK compute_default_camIdx_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	if (index < 3)
	{
		// scalar of VX_TYPE_UINT32
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: calc_lens_distortionwarp_map par%d should be UINT32 type\n", index);
		}
	}
	else if (index == 3)
	{
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(vx_float32)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: calc_lens_distortionwarp_map array element size should be 4 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return VX_SUCCESS;
}



//! \brief The output validator callback.
static vx_status VX_CALLBACK compute_default_camIdx_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 4)
	{ // image of format U8
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		// set output image meta data
		vx_df_image format = VX_DF_IMAGE_U8;
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		status = VX_SUCCESS;
	}
	return status;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK compute_default_camIdx_opencl_codegen(
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
	int numCam = 0;
	vx_uint32 out_width = 0, out_height = 0;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &numCam));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	scalar = (vx_scalar)avxGetNodeParamRef(node, 1);
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &out_width));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	scalar = (vx_scalar)avxGetNodeParamRef(node, 2);
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &out_height));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	// set kernel configuration
	strcpy(opencl_kernel_function_name, "compute_default_camIdx");
	opencl_work_dim = 2;
	opencl_local_work[0] = 8;
	opencl_local_work[1] = 8;
	opencl_global_work[0] = (((out_width + 15) >> 4) + 7) & ~7;
	opencl_global_work[1] = (out_height + 7) & ~7;
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(	uint numCam,      \n"
		"			uint eqrWidth, uint eqrHeight,\n"
		"			__global char * camera_z_value_buf, uint camera_z_value_buf_offs, uint zbuf_num, \n"
		"		    uint dc_width, uint	dc_height, __global uchar *default_camIdx_map, uint dc_stride, uint	dc_offs)\n"
		"{\n"
		"	int gx = get_global_id(0);\n"
		"	int gy = get_global_id(1);\n"
		"	gx <<= 4; \n"
		"	if ( (gx < dc_width) && (gy < dc_height))\n"
		"	{\n"
		"		camera_z_value_buf += camera_z_value_buf_offs + (((gy * eqrWidth) + gx) << 2);\n"
		"		int buf_offs = (dc_width*dc_height*4);\n"
		"		default_camIdx_map += dc_offs + (gy * dc_stride) + gx;\n"
		"		float8 in_val = vload8(0, (__global float *) camera_z_value_buf);\n"
		"		float8 in_val1 = vload8(0, (__global float *) (camera_z_value_buf+32));\n"
		"		int8 cam_idx = (int8)0, cam_idx1 = (int8)0;\n"
		"		int cam_id = 1; \n"
		"		float8 in_val2, in_val3;\n"
		"		while(cam_id < %d){\n"
		"			camera_z_value_buf += buf_offs;\n"
		"			in_val2 = vload8(0, (__global float *) (camera_z_value_buf));\n"
		"			in_val3 = vload8(0, (__global float *) (camera_z_value_buf + 32));\n"
		"			cam_idx = select(cam_idx, (int8)(cam_id), (in_val2 > in_val));\n"
		"			cam_idx1 = select(cam_idx1, (int8)(cam_id), (in_val3 > in_val1));\n"
		"			cam_idx = select(cam_idx, (int8)(0xFF), (in_val2 == in_val));\n"
		"			cam_idx1 = select(cam_idx1, (int8)(0xFF), (in_val3 == in_val1));\n"
		"			in_val = select(in_val, in_val2, (in_val2 > in_val));\n"
		"			in_val1 = select(in_val1, in_val3, (in_val3 > in_val1));\n"
		"			cam_id++;\n"
		"		}\n"
		"		*(__global uchar8 *)default_camIdx_map = convert_uchar8_sat(cam_idx);\n"
		"		*(__global uchar8 *)(default_camIdx_map+8) = convert_uchar8_sat(cam_idx1);\n"
		"	}\n"
		"}\n"
        , (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, numCam);
	opencl_kernel_code = item;
	return VX_SUCCESS;
}


//! \brief The kernel execution.
static vx_status VX_CALLBACK compute_default_camIdx_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK compute_default_camIdx_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status compute_default_camIdx_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.compute_default_camIdx",
		AMDOVX_KERNEL_STITCHING_INIT_COMPUTE_DEFAULT_CAMERA_IDX,
		compute_default_camIdx_kernel,
		5,
		compute_default_camIdx_input_validator,
		compute_default_camIdx_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = compute_default_camIdx_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = compute_default_camIdx_opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK extend_padding_dilate_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	if (!index)
	{
		// scalar of VX_TYPE_UINT32
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: calc_lens_distortionwarp_map par%d should be UINT32 type\n", index);
		}
	}
	else //if (index == 3)
	{
		if (ref){
			// check input image format and dimensions
			vx_uint32 input_width = 0, input_height = 0;
			vx_df_image input_format = VX_DF_IMAGE_VIRT;
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
			if (input_format != VX_DF_IMAGE_U32) {
				status = VX_ERROR_INVALID_TYPE;
				vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation mask image should be of format U008\n");
			}
			ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		}
		status = VX_SUCCESS;
	}
	return VX_SUCCESS;
}



//! \brief The output validator callback.
static vx_status VX_CALLBACK extend_padding_dilate_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 2)
	{ // image of format VX_DF_IMAGE_U32
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		// set output image meta data
		vx_df_image format = VX_DF_IMAGE_U32;
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		status = VX_SUCCESS;
	}
	return status;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK extend_padding_dilate_opencl_codegen(
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
	int numCam = 0;
	vx_uint32 out_width = 0, out_height = 0;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &numCam));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 2);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &out_width, sizeof(out_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &out_height, sizeof(out_height)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	// set kernel configuration
	strcpy(opencl_kernel_function_name, "extend_padding_dilate");
	opencl_work_dim = 2;
	opencl_local_work[0] = 8;
	opencl_local_work[1] = 8;
	opencl_global_work[0] = (((out_width + 7) >> 3) + 7) & ~7;
	opencl_global_work[1] = (out_height + 7) & ~7;
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint padding_pixels,\n"
		"		uint vm_width, uint	vm_height, __global uchar *valid_pixel_map, uint vm_stride, uint vm_offs,\n"
		"		uint pm_width, uint	pm_height, __global uchar * padded_pixel_map, uint pm_stride, uint padded_pix_map_offset)\n"
		"{\n"
		"	int gx = get_global_id(0);\n"
		"	int gy = get_global_id(1);\n"
		"	gx <<= 3;	// process 8 pixels\n"
		"	if ((gx < %d) && (gy < %d))\n"
		, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, out_width, out_height);
	opencl_kernel_code = item;
	opencl_kernel_code +=
		"	{\n"
		"		valid_pixel_map += vm_offs; \n"
		"		padded_pixel_map += padded_pix_map_offset;\n"
		"		__global uchar *gbuff = (__global uchar *)(valid_pixel_map + gy*vm_stride);\n"
		"		__global uchar *dbuff = (__global uchar *)(padded_pixel_map + gy*pm_stride + (gx<<2));\n"
		"		uint8 L0 = (uint8)0; \n"
		"		// do horizontal filtering\n"
		"		int start_x = max((int)(gx - padding_pixels+8), (int)0);\n"
		"		int end_x = min((int)(gx + padding_pixels), (int)vm_width);\n"
		"		int num_pixels = end_x-start_x;\n"
		"		int goffs = max(start_x-8, (int)0);\n"
		"		// load left 8 extra pixels\n"
		"		uint8 Lt = vload8(0, (__global uint *)(gbuff + (goffs<<2)));\n"
		"		goffs = min(end_x, (int)(vm_width-8)); uint8 Rt = vload8(0, (__global uint *)(gbuff + (goffs<<2)));\n"
		"		goffs = start_x;\n"
		"		while (num_pixels >= 8){\n"
		"			L0 |= vload8(0, (__global uint *) (gbuff + (goffs<<2))); num_pixels-=8; goffs += 8;\n"
		"		}\n"
		"		L0.lo = L0.lo | L0.hi;\n"
		"		L0.s0 = L0.s0 | L0.s1 | L0.s2 | L0.s3;\n"
		"		for (int i=0; i < num_pixels; i++){\n"
		"			L0.s0 |= *(__global uint *)(gbuff + ((goffs+i)<<2));\n"
		"		}\n"
		"		uint8 D = (uint8)L0.s0;\n"
		"		// compute D.s0\n"
		"		Lt.lo = Lt.lo | Lt.hi;\n"
		"		Lt.s0 = Lt.s0 | Lt.s1 | Lt.s2 | Lt.s3;\n"
		"		D.s0  |= Lt.s0;		\n"
		"		// compute D.s1\n"
		"		Lt.s0 = 0; Lt.lo = Lt.lo | Lt.hi;\n"
		"		Lt.s1 = Lt.s0 | Lt.s1 | Lt.s2 | Lt.s3;\n"
		"		D.s1  |= (Lt.s1|Rt.s0);		\n"
		"		// compute D.s2\n"
		"		Lt.s2 |= Lt.s3 | Lt.s4 | Lt.s5 |Lt.s6 | Lt.s7;\n"
		"		Rt.s0 |= Rt.s1; D.s2  |= Lt.s2 | Rt.s0;		\n"
		"		// compute D.s3\n"
		"		Lt.s3 |= Lt.s4 | Lt.s5 | Lt.s6 | Lt.s7;\n"
		"		Rt.s0 |= Rt.s2; D.s2  |= Lt.s3 | Rt.s0;		\n"
		"		// compute D.s4\n"
		"		Lt.s4 |= Lt.s5 | Lt.s6 | Lt.s7;\n"
		"		Rt.s0 |= Rt.s3; D.s3  |= Lt.s4 | Rt.s0;		\n"
		"		// compute D.s4\n"
		"		Rt.s0 |= Rt.s4; D.s4  |= Lt.s5 | Lt.s6| Lt.s7 | Rt.s0;		\n"
		"		// compute D.s5\n"
		"		Rt.s0 |= Rt.s5; D.s5  |= Lt.s6| Lt.s7 | Rt.s0;		\n"
		"		// compute D.s6\n"
		"		Rt.s0 |= Rt.s6; D.s6  |= Lt.s7 | Rt.s0;		\n"
		"		// compute D.s7\n"
		"		Rt.s0 |= Rt.s7; D.s7  |= Rt.s0;	\n"
		"		// do vertical filtering\n"
		"		gbuff = valid_pixel_map + (gx<<2);\n"
		"		uint8 p0 = vload8(0, (__global uint *) (gbuff + gy*vm_stride));\n"
		"		int start_y = max((int)(gy - padding_pixels), (int)0);\n"
		"		int end_y = min((int)(gy + padding_pixels), (int)vm_height);\n"
		"		int num_items = end_y-start_y;\n"
		"		goffs = start_y*vm_stride;\n"
		"		L0 = vload8(0, (__global uint *) (gbuff + goffs));\n"
		"		goffs += vm_stride;\n"
		"		for (int i=1; i<num_items; i++){\n"
		"			L0 |= vload8(0, (__global uint *) (gbuff + goffs));\n"
		"			goffs += vm_stride;\n"
		"		}\n"
		"		L0 |= D;	// or with horizontal filter output\n"
		"		L0 &= (~p0);\n"
		"		*(__global uint8 *)dbuff = L0;\n"
		"	}\n"
		" }\n";
	return VX_SUCCESS;
}


//! \brief The kernel execution.
static vx_status VX_CALLBACK extend_padding_dilate_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK extend_padding_dilate_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The kernel publisher.
vx_status extend_padding_dilate_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.extend_padding_dilate",
		AMDOVX_KERNEL_STITCHING_INIT_EXTEND_PAD_DILATE,
		extend_padding_dilate_kernel,
		3,
		extend_padding_dilate_input_validator,
		extend_padding_dilate_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = extend_padding_dilate_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = extend_padding_dilate_opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}
