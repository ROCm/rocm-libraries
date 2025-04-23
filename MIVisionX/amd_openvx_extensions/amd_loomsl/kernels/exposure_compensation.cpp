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
#include "exposure_compensation.h"

//! \brief The input validator callback.
static vx_status VX_CALLBACK exposure_comp_calcErrorFn_input_validator(vx_node node, vx_uint32 index)
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
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exp_comp num_cameras scalar type should be a UINT32\n");
		}
	}
	else if (index == 1)
	{ // image of format RGBX
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGBX) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation doesn't support input image format: %4.4s\n", &input_format);
		}
		status = VX_SUCCESS;
	}
	else if (index == 2)
	{ // array object for offsets
		vx_size itemsize = 0;
		vx_size capacity = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
		if (itemsize != sizeof(StitchOverlapPixelEntry)) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation gains array type should be float32\n");
		}
		else if (capacity == 0) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation gains array capacity should be positive\n");
		}
		else {
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 3)
	{ // image of format U008
		if (ref){
			// check input image format and dimensions
			vx_uint32 input_width = 0, input_height = 0;
			vx_df_image input_format = VX_DF_IMAGE_VIRT;
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
			ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
			if (input_format != VX_DF_IMAGE_U8) {
				status = VX_ERROR_INVALID_TYPE;
				vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation mask image should be of format U008\n");
			}
		}
		status = VX_SUCCESS;
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK exposure_comp_calcErrorFn_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	if (index == 4)
	{ // matrix of type VX_TYPE_INT32
		vx_enum type = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_TYPE, &type, sizeof(type)));
		if (type == VX_TYPE_INT32) {
			// check matrix dimensions
			vx_size columns = 0, rows = 0;
			ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
			ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_TYPE, &type, sizeof(type)));
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: lens distortion matrix type should be an float32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseMatrix((vx_matrix *)&ref));
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK exposure_comp_calcErrorFn_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK exposure_comp_calcErrorFn_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK exposure_comp_calcErrorFn_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	// Get the number of elements in the stitchWarpRemapEntry array
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 2);				// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_numitems, sizeof(arr_numitems)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	opencl_global_work[0] = arr_numitems*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1];
	
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK exposure_comp_calcErrorFn_opencl_codegen(
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
	vx_size		arr_size;
	vx_uint32 num_cameras = 0;
	vx_uint32 input_width = 0, input_height = 0, output_width = 0, output_height = 0;
	vx_df_image input_format = VX_DF_IMAGE_VIRT, output_format = VX_DF_IMAGE_VIRT;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);			// input scalar - num cameras
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &num_cameras));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 1);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	vx_array exp_data = (vx_array)avxGetNodeParamRef(node, 2);
	ERROR_CHECK_STATUS(vxQueryArray(exp_data, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_size, sizeof(arr_size)));
	ERROR_CHECK_STATUS(vxReleaseArray(&exp_data));
	vx_image mask_image = (vx_image)avxGetNodeParamRef(node, 3);
	if (mask_image != NULL){
		ERROR_CHECK_STATUS(vxQueryImage(mask_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage(mask_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage(mask_image, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
	}

	// set kernel configuration
	vx_uint32 height_one = (vx_uint32)(input_height / num_cameras);
	strcpy(opencl_kernel_function_name, "exposure_comp_calc_errorfn_mask");
	opencl_work_dim = 2;
	opencl_local_work[0] = 16;
	opencl_local_work[1] = 16;
	opencl_global_work[0] = arr_size*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1];
	// kernel header and reading
	char item[8192];
	if (mask_image){
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"__attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"__kernel void %s(uint num_cameras,\n" // opencl_kernel_function_name
			"			uint	pIn_width, uint	pIn_height, __global uchar *pIn_buf, uint pIn_stride, uint	pIn_offs,\n"
			"			__global uchar * exp_data, uint	exp_data_offs, uint exp_data_num,\n"
			"			uint	pWt_width, uint	pWt_height, __global uchar *pWt_buf, uint pWt_stride, uint	pWt_offs,\n"
			"			__global int * pAMat, uint cols, uint rows)\n"
			"{\n"
			"	int grp_id = get_global_id(0)>>4;\n"
			"	__local uint  sumI[256], sumJ[256];\n"
			"   if (grp_id < exp_data_num) {\n"
			"	uint2 offs = ((__global uint2 *)(exp_data+exp_data_offs))[grp_id];\n"
			"	uint size = (uint)(pIn_stride*%d);\n"
			"	uint wt_size = (uint)(pWt_stride*%d);\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height_one, height_one);
		opencl_kernel_code = item;
		opencl_kernel_code +=
			"	int lx = get_local_id(0);\n"
			"	int ly = get_local_id(1);\n"
			"	int lid = mad24(ly, (int)get_local_size(0), lx);\n"
			"   sumI[lid] = 0; sumJ[lid] = 0;\n"
			"	bool isValid = ((lx<<3) < (int)(offs.s1&0x7f)) && (ly*2 < (int)((offs.s1>>7)&0x1f));\n"
			"	if (isValid) {\n"
			"		global uint *pI, *pJ;\n"
			"		uint4 maskSrc, I, J, mask; \n"
			"		uint4 Isum4, Jsum4;\n"
			"		int   gx = (lx<<3) + ((offs.s0 >> 5) & 0x3FFF);\n"
			"		int   gy = (ly<<1) + (offs.s0 >> 19);\n"
			"		uint2 cam_id = (uint2)((offs.s0 & 0x1f), ((offs.s1>>12) & 0x1f));\n"
			"		pIn_buf += pIn_offs + mad24(gy, (int)pIn_stride, (gx<<2));\n"
			"		pWt_buf += pWt_offs + mad24(gy, (int)pWt_stride, gx);\n"
			"		pI	   =  (global uint *)(pIn_buf + size*cam_id.x);\n"
			"		pJ	   =  (global uint *)(pIn_buf + size*cam_id.y);\n"
			"		maskSrc.s01	   =  *(global uint2 *)(pWt_buf + wt_size*cam_id.x);\n"
			"		maskSrc.s01	   &=  *(global uint2 *)(pWt_buf + wt_size*cam_id.y); pWt_buf += pWt_stride;\n"
			"		maskSrc.s23	   =  *(global uint2 *)(pWt_buf + wt_size*cam_id.x);\n"
			"		maskSrc.s23	   &=  *(global uint2 *)(pWt_buf + wt_size*cam_id.y);\n"
			"		char4 maskIJ = as_char4(maskSrc.s0);\n"
			"		I = vload4(0, pI);\n"
			"		J = vload4(0, pJ); \n"
			"		mask.s0	= select(0xff000000, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xff000000, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xff000000, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xff000000, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		Isum4	= (I&mask)>>24; Jsum4 = (J & mask)>>24;\n"
			"		I = vload4(1, pI);\n"
			"		J = vload4(1, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s1);\n"
			"		mask.s0	= select(0xff000000, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xff000000, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xff000000, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xff000000, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		Isum4	+= (I&mask)>>24; Jsum4 += (J & mask)>>24;\n"
			"		pI += (pIn_stride>>2); pJ += (pIn_stride>>2);\n"
			"		I = vload4(0, pI);\n"
			"		J = vload4(0, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s2);\n"
			"		mask.s0	= select(0xff000000, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xff000000, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xff000000, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xff000000, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		Isum4	+= (I&mask)>>24; Jsum4 += (J & mask)>>24;\n"
			"		I = vload4(1, pI); \n"
			"		J = vload4(1, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s3);\n"
			"		mask.s0	= select(0xff000000, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xff000000, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xff000000, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xff000000, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		Isum4 += ((I&mask) >> 24); Jsum4 += ((J & mask) >> 24); \n"
			"		sumI[lid] = mad24(Isum4.s3, (uint)1, mad24(Isum4.s2, (uint)1, mad24(Isum4.s1, (uint)1, Isum4.s0)));\n"
			"		sumJ[lid] = mad24(Jsum4.s3, (uint)1, mad24(Jsum4.s2, (uint)1, mad24(Jsum4.s1, (uint)1, Jsum4.s0)));\n"
			"		barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	else
	{
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"__attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"__kernel void %s(uint num_cameras,\n" // opencl_kernel_function_name
			"			uint	pIn_width, uint	pIn_height, __global uchar *pIn_buf, uint pIn_stride, uint	pIn_offs,\n"
			"			__global uchar * exp_data, uint	exp_data_offs, uint exp_data_num,\n"
			"			__global int * pAMat, uint cols, uint rows)\n"
			"{\n"
			"	int grp_id = get_global_id(0)>>4;\n"
			"   if (grp_id < exp_data_num) {\n"
			"	__local uint  sumI[256], sumJ[256];\n"
			"	uint2 offs = ((__global uint2 *)(exp_data+exp_data_offs))[grp_id];\n"
			"	uint size = (uint)(pIn_stride*%d);\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height_one);
		opencl_kernel_code = item;
		opencl_kernel_code +=
			"	int lx = get_local_id(0);\n"
			"	int ly = get_local_id(1);\n"
			"	int lid = mad24(ly, (int)get_local_size(0), lx);\n"
			"   sumI[lid] = 0; sumJ[lid] = 0;\n"
			"	bool isValid = ((lx<<3) < (int)(offs.s1&0x7f)) && (ly*2 < (int)((offs.s1>>7)&0x1f));\n"
			"	if (isValid) {\n"
			"		global uint *pI, *pJ;\n"
			"		uint4  I, J, mask; \n"
			"		uint4 Isum4, Jsum4;\n"
			"		int   gx = (lx<<3) + ((offs.s0 >> 5) & 0x3FFF);\n"
			"		int   gy = (ly<<1) + (offs.s0 >> 19);\n"
			"		uint2 cam_id = (uint2)((offs.s0 & 0x1f), ((offs.s1>>12) & 0x1f));\n"
			"		pIn_buf += pIn_offs + mad24(gy, (int)pIn_stride, (gx<<2));\n"
			"		pI	   =  (global uint *)(pIn_buf + size*cam_id.x);\n"
			"		pJ	   =  (global uint *)(pIn_buf + size*cam_id.y);\n"
			"		I = vload4(0, pI);\n"
			"		J = vload4(0, pJ); \n"
			"		mask.s0	= select(0xff000000, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000)));\n"
			"		mask.s1	= select(0xff000000, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000)));\n"
			"		mask.s2	= select(0xff000000, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000)));\n"
			"		mask.s3	= select(0xff000000, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000)));\n"
			"		Isum4	= (I&mask)>>24; Jsum4 = (J & mask)>>24;\n"
			"		I = vload4(1, pI);\n"
			"		J = vload4(1, pJ);\n"
			"		mask.s0	= select(0xff000000, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000)));\n"
			"		mask.s1	= select(0xff000000, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000)));\n"
			"		mask.s2	= select(0xff000000, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000)));\n"
			"		mask.s3	= select(0xff000000, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000)));\n"
			"		Isum4	+= (I&mask)>>24; Jsum4 += (J & mask)>>24;\n"
			"		pI += (pIn_stride>>2); pJ += (pIn_stride>>2);\n"
			"		I = vload4(0, pI);\n"
			"		J = vload4(0, pJ); \n"
			"		mask.s0	= select(0xff000000, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000)));\n"
			"		mask.s1	= select(0xff000000, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000)));\n"
			"		mask.s2	= select(0xff000000, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000)));\n"
			"		mask.s3	= select(0xff000000, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000)));\n"
			"		Isum4	+= (I&mask)>>24; Jsum4 += (J & mask)>>24;\n"
			"		I = vload4(1, pI); \n"
			"		J = vload4(1, pJ); \n"
			"		mask.s0	= select(0xff000000, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000)));\n"
			"		mask.s1	= select(0xff000000, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000)));\n"
			"		mask.s2	= select(0xff000000, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000)));\n"
			"		mask.s3	= select(0xff000000, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000)));\n"
			"		Isum4 += ((I&mask) >> 24); Jsum4 += ((J & mask) >> 24); \n"
			"		sumI[lid] = mad24(Isum4.s3, (uint)1, mad24(Isum4.s2, (uint)1, mad24(Isum4.s1, (uint)1, Isum4.s0)));\n"
			"		sumJ[lid] = mad24(Jsum4.s3, (uint)1, mad24(Jsum4.s2, (uint)1, mad24(Jsum4.s1, (uint)1, Jsum4.s0)));\n"
			"		barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	opencl_kernel_code +=
		"		// aggregate sum and count from all threads\n"
		"		if (lid < 128)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+128];\n"
		"			sumJ[lid]	+= sumJ[lid+128];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		if (lid < 64)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+64];\n"
		"			sumJ[lid]	+= sumJ[lid+64];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		if (lid < 32)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+32];\n"
		"			sumJ[lid]	+= sumJ[lid+32];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"\n"
		"		if (lid < 16)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+16];\n"
		"			sumJ[lid]	+= sumJ[lid+16];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"\n"
		"		if (lid < 8)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+8];\n"
		"			sumJ[lid]	+= sumJ[lid+8];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		uint idx1, s1; uint4 t1;\n"
		"		if (!lid)\n"
		"		{\n"
		"			idx1 = mad24(cam_id.x, cols, cam_id.y);\n"
		"			t1 = ((local uint4*)sumI)[0] + ((local uint4*)sumI)[1];\n"
		"			s1 = t1.s0 + t1.s1 + t1.s2 + t1.s3;\n"
		"			atomic_add(&pAMat[idx1], (int)(s1*0.0625f));\n"
		"		}\n"
		"		else if (lid == 1){\n"
		"			idx1 = mad24(cam_id.y, cols, cam_id.x);\n"
		"			t1 = ((local uint4*)sumJ)[0] + ((local uint4*)sumJ)[1];\n"
		"			s1 = t1.s0 + t1.s1 + t1.s2 + t1.s3;\n"
		"			atomic_add(&pAMat[idx1], (int)(s1*0.0625f));\n"
		"		}\n"
		"	}\n"
		"	}\n"
		"}\n";
	if (mask_image)ERROR_CHECK_STATUS(vxReleaseImage(&mask_image));
	return VX_SUCCESS;
}

//! \brief The exposure_comp_applygains kernel publisher.
vx_status exposure_comp_calcErrorFn_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.expcomp_compute_gainmatrix",
		AMDOVX_KERNEL_STITCHING_EXPCOMP_COMPUTE_GAINMAT,
		exposure_comp_calcErrorFn_kernel,
		5,
		exposure_comp_calcErrorFn_input_validator,
		exposure_comp_calcErrorFn_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	// set codegen for opencl
	amd_kernel_query_target_support_f query_target_support_f = exposure_comp_calcErrorFn_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = exposure_comp_calcErrorFn_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = exposure_comp_calcErrorFn_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));
	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_OPTIONAL));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	return VX_SUCCESS;
}


//! \brief The input validator callback.
static vx_status VX_CALLBACK exposure_comp_applygains_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0)
	{ // image of format RGBX
		vx_array arr = (vx_array)avxGetNodeParamRef(node, 1);
		ERROR_CHECK_OBJECT(arr);
		vx_uint32 num_cam = 0;
		vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 3);			// input scalar - num cameras
		ERROR_CHECK_OBJECT(scalar);
		ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &num_cam));
		ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGBX) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation doesn't support input image format: %4.4s\n", &input_format);
		}
		else if ((input_height % num_cam) != 0) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation invalid input image dimensions: %dx%d (height should be multiple of %d)\n", input_width, input_height, num_cam);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 1)
	{ // array object for gains
		vx_enum itemtype = VX_TYPE_INVALID;
		vx_size capacity = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
		if (itemtype != VX_TYPE_FLOAT32) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation gains array type should be float32\n");
		}
		else if (capacity == 0) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation gains array capacity should be positive\n");
		}
		else {
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 2)
	{ // array of type VX_TYPE_UINT64
		vx_size itemsize = 0, capacity = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize != sizeof(StitchExpCompCalcEntry)) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation offset array type should be VX_TYPE_UINT64\n");
		}
		else if (capacity == 0) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation array capacity should be positive\n");
		}
		else {
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 3)
	{ // scalar for numcam
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exp_comp num_cameras scalar type should be a UINT32\n");
		}
	}
	else if ((index == 4) || (index == 5))
	{ // optional parameter for block_gain buffer width/height
		if (ref){
			vx_enum itemtype = VX_TYPE_INVALID;
			ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
			ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
			if (itemtype == VX_TYPE_UINT32) {
				status = VX_SUCCESS;
			}
			else {
				status = VX_ERROR_INVALID_TYPE;
				vxAddLogEntry((vx_reference)node, status, "ERROR: exp_comp num_cameras scalar type should be a UINT32\n");
			}
		}
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK exposure_comp_applygains_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 6)
	{ // image of format RGBX
		// get image configuration
		vx_image image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		// set output image meta data
		vx_df_image format = VX_DF_IMAGE_RGBX;
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		status = VX_SUCCESS;
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK exposure_comp_applygains_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK exposure_comp_applygains_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK exposure_comp_applygains_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	vx_array arr = (vx_array)parameters[2];				// input array
	vx_scalar sc_width = (vx_scalar)parameters[4];		// input scalar - bg width
	// Get the number of elements in the stitchWarpRemapEntry array
	vx_size arr_numitems = 0;
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &arr_numitems, sizeof(arr_numitems)));
	opencl_global_work[0] = arr_numitems*opencl_local_work[0];
	opencl_global_work[1] = 2*opencl_local_work[1];
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK exposure_comp_applygains_opencl_codegen(
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
	vx_size		wg_num, num_gains;
	vx_uint32 input_width = 0, input_height = 0, output_width = 0, output_height = 0;
	vx_df_image input_format = VX_DF_IMAGE_VIRT, output_format = VX_DF_IMAGE_VIRT;
	vx_image image = (vx_image)avxGetNodeParamRef(node, 0);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	image = (vx_image)avxGetNodeParamRef(node, 6);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	vx_array wg_offsets = (vx_array)avxGetNodeParamRef(node, 2);
	ERROR_CHECK_STATUS(vxQueryArray(wg_offsets, VX_ARRAY_ATTRIBUTE_CAPACITY, &wg_num, sizeof(wg_num)));
	ERROR_CHECK_STATUS(vxReleaseArray(&wg_offsets));
	vx_uint32 num_cam = 0;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 3);			// input scalar - num cameras
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &num_cam));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	if (!num_cam) num_cam = 1;	// has to be atleast 1
	vx_array gains = (vx_array)avxGetNodeParamRef(node, 1);
	ERROR_CHECK_STATUS(vxQueryArray(gains, VX_ARRAY_ATTRIBUTE_CAPACITY, &num_gains, sizeof(num_gains)));

	// check if we get extra parameters for bg_width and bg_height ( for doing block_gain compensation
	vx_uint32 bg_width=1, bg_height=1;
	vx_scalar sc_width = (vx_scalar)avxGetNodeParamRef(node, 4);			// input scalar - bg width
	vx_scalar sc_height = (vx_scalar)avxGetNodeParamRef(node, 5);			// input scalar - bg height
	if (sc_width)ERROR_CHECK_STATUS(vxReadScalarValue(sc_width, &bg_width));
	if (sc_height)ERROR_CHECK_STATUS(vxReadScalarValue(sc_height, &bg_height));
	bg_width = std::max(1, (int)bg_width);
	bg_height = std::max(1, (int)bg_height);
	if (num_gains < bg_width*bg_height*num_cam)
		return VX_ERROR_INVALID_DIMENSION;
	vx_int32 bRGBGain = (num_gains >= bg_width*bg_height*num_cam * 3);			// if gain array gives gain for R, G and B seperate
	vx_uint32 height_one_in, height_one_out;
	height_one_in = (vx_uint32)(input_height / num_cam);
	height_one_out = (vx_uint32)(output_height / num_cam);
	// set kernel configuration
	strcpy(opencl_kernel_function_name, "exposure_comp_apply_gains");
	opencl_work_dim = 2;
	opencl_local_work[0] = 16;
	opencl_local_work[1] = 16;
	opencl_global_work[0] = wg_num*opencl_local_work[0];
	opencl_global_work[1] = 2*opencl_local_work[1];
	// opencl kernel header and reading
	char item[8192];
	if (sc_width && sc_height){
		vx_float32 xscale = 1.0f, yscale = 1.0f, xoffset = 0.0f, yoffset = 0.0f;
		// calculate xscale and yscale for block_gain computation
		if (bg_width >= 1){
			xscale = (vx_float32)bg_width / output_width;
			xoffset = (vx_float32)(xscale*0.5 - 0.5);
		}
		if (bg_height >= 1){
			yscale = (vx_float32)(bg_height*num_cam) / output_height;
			yoffset = (vx_float32)(yscale*0.5 - 0.5);
		}
		if (!bRGBGain){
			snprintf(item, sizeof(item),
				"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
				"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
				"\n"
				"float4 amd_unpack(uint src)\n"
				"{\n"
				"	return (float4)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src), amd_unpack3(src));\n"
				"}\n"
				"float BilinearSample(__global float *p, uint ystride, float fy0, float fy1, int x, float fx0, float fx1)\n"
				"{\n"
				"  float4 f;\n"
				"  p += x;\n"
				"  f.s0 = p[0]; f.s1 = p[1];\n"
				"  p += ystride;\n"
				"  f.s2 = p[0]; f.s3 = p[1];\n"
				"  f.s0 = mad(f.s0, fx0, f.s1 * fx1);\n"
				"  f.s2 = mad(f.s2, fx0, f.s3 * fx1);\n"
				"  f.s0 = mad(f.s0, fy0, f.s2 * fy1);\n"
				"  return f.s0;\n"
				"}\n"
				"\n"
				"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
				"void %s(uint pIn_width, uint pIn_height, __global uchar * pIn_buf, uint pIn_stride, uint pIn_offset,\n"
				"        __global uchar * pG_buf, uint pG_offs, uint pG_num,\n"
				"        __global uchar * pExpData_buf, uint pExpData_offset, uint pExpData_num, uint numcam, \n"
				"         uint bg_width, uint bg_height, \n"
				"        uint pOut_width, uint pOut_height, __global uchar * pOut_buf, uint pOut_stride, uint pOut_offset)\n"
				"{\n"
				"	int grp_id = get_global_id(0)>>4;\n"
				"   if (grp_id < pExpData_num) {\n"
				"	uint2 size = (uint2)((pIn_stride*%d), (pOut_stride*%d));\n"
				"	float4 scalexy = (float4)(%f, %f, %f, %f); uint size_bg = bg_width *%d;\n"
				, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height_one_in, height_one_out, xscale, yscale, xoffset, yoffset, bg_height);
			opencl_kernel_code = item;
			opencl_kernel_code +=
				"	uint2 offs = ((__global uint2 *)(pExpData_buf+pExpData_offset))[grp_id];\n"
				"	int cam_id = offs.s0&0x3f;\n"
				"	__global float *pGainBuf = (__global float *)(pG_buf + pG_offs);\n"
				"	pGainBuf += (cam_id*size_bg);\n"
				"	int  lx = get_local_id(0);\n"
				"	int  ly = get_global_id(1);\n"
				"   int   gx = lx + ((offs.s0 >> 6) & 0xFFF);\n"
				"   int   gy = ly + (offs.s0 >> 17) ;\n"
				"   pIn_buf += pIn_offset + (size.x*cam_id) + mad24(gy, (int)pIn_stride, (gx<<5));\n"
				"   pOut_buf += pOut_offset + (size.y*cam_id) + mad24(gy, (int)pOut_stride, (gx<<5));\n"
				"   uchar4 offs4 = as_uchar4(offs.s1); \n"
				"   if (((lx<<3) < (int)offs4.s2) && (ly <= (int)offs4.s3)) {\n"
				"	float fx, fy, fy0, fy1, fint, frac;\n"
				"	fx = mad((gx<<3), scalexy.s0, scalexy.s2); fy = mad(gy, scalexy.s1, scalexy.s3);\n"
				"	fy0 = floor(fy); fy1 = fy - fy0; fy0 = 1.0f - fy1;\n"
				"	float4 f, f4; \n"
				"	pGainBuf += mul24((uint)fy, bg_width);\n"
				"	fint = floor(fx); frac = fx - fint; f.s0 = BilinearSample(pGainBuf, bg_width, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f.s1 = BilinearSample(pGainBuf, bg_width, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f.s2 = BilinearSample(pGainBuf, bg_width, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f.s3 = BilinearSample(pGainBuf, bg_width, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	uint8 r0 =  *(__global uint8 *)pIn_buf;\n"
				"	f4 = amd_unpack(r0.s0)*(float4)f.s0; r0.s0 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s1)*(float4)f.s1; r0.s1 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s2)*(float4)f.s2; r0.s2 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s3)*(float4)f.s3; r0.s3 = amd_pack(f4); \n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f.s0 = BilinearSample(pGainBuf, bg_width, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f.s1 = BilinearSample(pGainBuf, bg_width, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f.s2 = BilinearSample(pGainBuf, bg_width, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f.s3 = BilinearSample(pGainBuf, bg_width, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	f4 = amd_unpack(r0.s4)*(float4)f.s0; r0.s4 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s5)*(float4)f.s1; r0.s5 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s6)*(float4)f.s2; r0.s6 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s7)*(float4)f.s3; r0.s7 = amd_pack(f4); \n"
				"	*(__global uint8 *)(pOut_buf) = r0;\n"
				"}\n"
				"}\n"
				"}\n";
		}
		else
		{
			snprintf(item, sizeof(item),
				"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
				"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
				"\n"
				"float4 amd_unpack(uint src)\n"
				"{\n"
				"	return (float4)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src), amd_unpack3(src));\n"
				"}\n"
				"float3 BilinearSample3(__global float *p, uint ystride, float fy0, float fy1, int x, float fx0, float fx1)\n"
				"{\n"
				"  float3 f0, f1, f2, f3;\n"
				"  p += x*3;\n"
				"  f0 = (float3)(p[0],p[1],p[2]); f1 = (float3)(p[3], p[4], p[5]);\n"
				"  p += ystride;\n"
				"  f2 = (float3)(p[0],p[1],p[2]); f3 = (float3)(p[3], p[4], p[5]);\n"
				"  f0 = mad(f0, fx0, f1 * fx1);\n"
				"  f2 = mad(f2, fx0, f3 * fx1);\n"
				"  f0 = mad(f0, fy0, f2 * fy1);\n"
				"  return f0;\n"
				"}\n"
				"\n"
				"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
				"void %s(uint pIn_width, uint pIn_height, __global uchar * pIn_buf, uint pIn_stride, uint pIn_offset,\n"
				"        __global uchar * pG_buf, uint pG_offs, uint pG_num,\n"
				"        __global uchar * pExpData_buf, uint pExpData_offset, uint pExpData_num, uint numcam, \n"
				"         uint bg_width, uint bg_height, \n"
				"        uint pOut_width, uint pOut_height, __global uchar * pOut_buf, uint pOut_stride, uint pOut_offset)\n"
				"{\n"
				"	int grp_id = get_global_id(0)>>4;\n"
				"   if (grp_id < pExpData_num) {\n"
				"	uint2 size = (uint2)((pIn_stride*%d), (pOut_stride*%d));\n"
				"	float4 scalexy = (float4)(%f, %f, %f, %f); uint size_bg = bg_width*3*%d;\n"
				, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height_one_in, height_one_out, xscale, yscale, xoffset, yoffset, bg_height);
			opencl_kernel_code = item;
			opencl_kernel_code +=
				"	uint2 offs = ((__global uint2 *)(pExpData_buf+pExpData_offset))[grp_id];\n"
				"	int cam_id = offs.s0&0x3f; uint gstride = bg_width*3;\n"
				"	__global float *pGainBuf = (__global float *)(pG_buf + pG_offs);\n"
				"	pGainBuf += (cam_id*size_bg);\n"
				"	int  lx = get_local_id(0);\n"
				"	int  ly = get_global_id(1);\n"
				"   int   gx = lx + ((offs.s0 >> 6) & 0xFFF);\n"
				"   int   gy = ly + (offs.s0 >> 17) ;\n"
				"   pIn_buf += pIn_offset + (size.x*cam_id) + mad24(gy, (int)pIn_stride, (gx<<5));\n"
				"   pOut_buf += pOut_offset + (size.y*cam_id) + mad24(gy, (int)pOut_stride, (gx<<5));\n"
				"   uchar4 offs4 = as_uchar4(offs.s1); \n"
				"   if (((lx<<3) < (int)offs4.s2) && (ly <= (int)offs4.s3)) {\n"
				"	float fx, fy, fy0, fy1, fint, frac;\n"
				"	fx = mad((gx<<3), scalexy.s0, scalexy.s2); fy = mad(gy, scalexy.s1, scalexy.s3);\n"
				"	fy0 = floor(fy); fy1 = fy - fy0; fy0 = 1.0f - fy1;\n"
				"	float4 f4; float3 f0, f1, f2, f3;\n"
				"	pGainBuf += mul24((uint)fy, gstride);\n"
				"	fint = floor(fx); frac = fx - fint; f0 = BilinearSample3(pGainBuf, gstride, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f1 = BilinearSample3(pGainBuf, gstride, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f2 = BilinearSample3(pGainBuf, gstride, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f3 = BilinearSample3(pGainBuf, gstride, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	uint8 r0 =  *(__global uint8 *)pIn_buf;\n"
				"	f4 = amd_unpack(r0.s0)*(float4)(f0, 1.0f); r0.s0 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s1)*(float4)(f1, 1.0f); r0.s1 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s2)*(float4)(f2, 1.0f); r0.s2 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s3)*(float4)(f3, 1.0f); r0.s3 = amd_pack(f4); \n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f0 = BilinearSample3(pGainBuf, gstride, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f1 = BilinearSample3(pGainBuf, gstride, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f2 = BilinearSample3(pGainBuf, gstride, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	fx += scalexy.s0; fint = floor(fx); frac = fx - fint; f3 = BilinearSample3(pGainBuf, gstride, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
				"	f4 = amd_unpack(r0.s4)*(float4)(f0, 1.0f); r0.s4 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s5)*(float4)(f1, 1.0f); r0.s5 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s6)*(float4)(f2, 1.0f); r0.s6 = amd_pack(f4); \n"
				"	f4 = amd_unpack(r0.s7)*(float4)(f3, 1.0f); r0.s7 = amd_pack(f4); \n"
				"	*(__global uint8 *)(pOut_buf) = r0;\n"
				"}\n"
			"}\n"
		"}\n";
		}
		ERROR_CHECK_STATUS(vxReleaseScalar(&sc_width));
		ERROR_CHECK_STATUS(vxReleaseScalar(&sc_height));
	}
	else if (num_gains == num_cam * 12) // if gain array gives color transform for R, G and B with bias offset
	{
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"\n"
			"float4 amd_unpack(uint src)\n"
			"{\n"
			"	return (float4)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src), amd_unpack3(src));\n"
			"}\n"
			"\n"
			"uint RGBTran(uint rgbx, float4 r4, float4 g4, float4 b4) {\n"
			"  float4 fin, fout;\n"
			"  fin = amd_unpack(rgbx);\n"
			"  fout.s0 = mad(fin.s0, r4.s0, mad(fin.s1, r4.s1, mad(fin.s2, r4.s2, r4.s3)));\n"
			"  fout.s1 = mad(fin.s0, g4.s0, mad(fin.s1, g4.s1, mad(fin.s2, g4.s2, g4.s3)));\n"
			"  fout.s2 = mad(fin.s0, b4.s0, mad(fin.s1, b4.s1, mad(fin.s2, b4.s2, b4.s3)));\n"
			"  fout.s3 = fin.s3;\n"
			"  return amd_pack(fout);\n"
			"}\n"
			"\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint pIn_width, uint pIn_height, __global uchar * pIn_buf, uint pIn_stride, uint pIn_offset,\n"
			"        __global uchar * pG_buf, uint pG_offs, uint pG_num,\n"
			"        __global uchar * pExpData_buf, uint pExpData_offset, uint pExpData_num, uint numcam, \n"
			"        uint pOut_width, uint pOut_height, __global uchar * pOut_buf, uint pOut_stride, uint pOut_offset)\n"
			"{\n"
			"  int grp_id = get_global_id(0)>>4;\n"
			"  if (grp_id < pExpData_num) {\n"
			"    uint2 size = (uint2)((pIn_stride*%d), (pOut_stride*%d));\n"
			"    uint2 offs = ((__global uint2 *)(pExpData_buf+pExpData_offset))[grp_id];\n"
			"    pG_buf += pG_offs; int cam_id = offs.s0&0x3f;\n"
			"    __global float4 * pg = (__global float4 *)pG_buf; pg += cam_id*3;\n"
			"    float4 r4 = pg[0], g4 = pg[1], b4 = pg[2];\n"
			"    int  lx = get_local_id(0);\n"
			"    int  ly = get_global_id(1);\n"
			"    int   gx = lx + ((offs.s0 >> 6) & 0xFFF);\n"
			"    int   gy = ly + ((offs.s0 >> 18) << 1);\n"
			"    pIn_buf += pIn_offset + (size.x*cam_id) + mad24(gy, (int)pIn_stride, (gx<<5));\n"
			"    pOut_buf += pOut_offset + (size.y*cam_id) + mad24(gy, (int)pOut_stride, (gx<<5));\n"
			"    uchar4 offs4 = as_uchar4(offs.s1); \n"
			"    if (((lx<<3) < (int)offs4.s2) && (ly <= (int)offs4.s3)) {\n"
			"      uint8 r0, r1;\n"
			"      r0 =  *(__global uint8 *)pIn_buf;\n"
			"      r0.s0 = RGBTran(r0.s0, r4, g4 , b4);\n"
			"      r0.s1 = RGBTran(r0.s1, r4, g4 , b4);\n"
			"      r0.s2 = RGBTran(r0.s2, r4, g4 , b4);\n"
			"      r0.s3 = RGBTran(r0.s3, r4, g4 , b4);\n"
			"      r0.s4 = RGBTran(r0.s4, r4, g4 , b4);\n"
			"      r0.s5 = RGBTran(r0.s5, r4, g4 , b4);\n"
			"      r0.s6 = RGBTran(r0.s6, r4, g4 , b4);\n"
			"      r0.s7 = RGBTran(r0.s7, r4, g4 , b4);\n"
			"      *(__global uint8 *)(pOut_buf) = r0;\n"
			"    }\n"
			"  }\n"
			"}\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height_one_in, height_one_out);
		opencl_kernel_code = item;
	}
	else
	{
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"\n"
			"float4 amd_unpack(uint src)\n"
			"{\n"
			"	return (float4)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src), amd_unpack3(src));\n"
			"}\n"
			"\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint pIn_width, uint pIn_height, __global uchar * pIn_buf, uint pIn_stride, uint pIn_offset,\n"
			"        __global uchar * pG_buf, uint pG_offs, uint pG_num,\n"
			"        __global uchar * pExpData_buf, uint pExpData_offset, uint pExpData_num, uint numcam, \n"
			"        uint pOut_width, uint pOut_height, __global uchar * pOut_buf, uint pOut_stride, uint pOut_offset)\n"
			"{\n"
			"	int grp_id = get_global_id(0)>>4;\n"
			"   if (grp_id < pExpData_num) {\n"
			"	uint2 size = (uint2)((pIn_stride*%d), (pOut_stride*%d));\n"
			"	uint2 offs = ((__global uint2 *)(pExpData_buf+pExpData_offset))[grp_id];\n"
			"	pG_buf += pG_offs; int cam_id = offs.s0&0x3f;\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height_one_in, height_one_out);
		opencl_kernel_code = item;
		if (bRGBGain){
			opencl_kernel_code +=
				"	__global float* pg = (__global float *)pG_buf; pg += cam_id*3;\n"
				"	float4 g4 = (float4)(pg[0], pg[1], pg[2], (float)1.0f);\n";
		}
		else
		{
			opencl_kernel_code +=
				"	__global float* pg = (__global float *)pG_buf; pg += cam_id;\n"
				"	float4 g4 = (float4)((float3)pg[0], (float)1.0f);\n";
		}
		opencl_kernel_code +=
			"	int  lx = get_local_id(0);\n"
			"	int  ly = get_global_id(1);\n"
			"   int   gx = lx + ((offs.s0 >> 6) & 0xFFF);\n"
			"   int   gy = ly + ((offs.s0 >> 18) << 1);\n"
			"   pIn_buf += pIn_offset + (size.x*cam_id) + mad24(gy, (int)pIn_stride, (gx<<5));\n"
			"   pOut_buf += pOut_offset + (size.y*cam_id) + mad24(gy, (int)pOut_stride, (gx<<5));\n"
			"   uchar4 offs4 = as_uchar4(offs.s1); \n"
			"   if (((lx<<3) < (int)offs4.s2) && (ly <= (int)offs4.s3)) {\n"
			"	uint8 r0; float4 f4;\n"
			"	r0 =  *(__global uint8 *)pIn_buf;\n"
			"	f4 = amd_unpack(r0.s0)*g4; r0.s0 = amd_pack(f4); \n"
			"	f4 = amd_unpack(r0.s1)*g4; r0.s1 = amd_pack(f4); \n"
			"	f4 = amd_unpack(r0.s2)*g4; r0.s2 = amd_pack(f4); \n"
			"	f4 = amd_unpack(r0.s3)*g4; r0.s3 = amd_pack(f4); \n"
			"	f4 = amd_unpack(r0.s4)*g4; r0.s4 = amd_pack(f4); \n"
			"	f4 = amd_unpack(r0.s5)*g4; r0.s5 = amd_pack(f4); \n"
			"	f4 = amd_unpack(r0.s6)*g4; r0.s6 = amd_pack(f4); \n"
			"	f4 = amd_unpack(r0.s7)*g4; r0.s7 = amd_pack(f4); \n"
			"	*(__global uint8 *)(pOut_buf) = r0;\n"
			"}\n"
		"}\n"
	"}\n";
	}
	return VX_SUCCESS;
}

//! \brief The exposure_comp_applygains kernel publisher.
vx_status exposure_comp_applygains_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.expcomp_applygains",
		AMDOVX_KERNEL_STITCHING_EXPCOMP_APPLYGAINS,
		exposure_comp_applygains_kernel,
		6,
		exposure_comp_applygains_input_validator,
		exposure_comp_applygains_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = exposure_comp_applygains_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = exposure_comp_applygains_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = exposure_comp_applygains_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));
	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK exposure_comp_solvegains_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	// validate each parameter
	if (index == 0 || index == 1)
	{ // scalar of type VX_TYPE_FLOAT32
		vx_enum type = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		if (type == VX_TYPE_FLOAT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation scalar type should be an float32\n");
		}
		if (!index){
			vx_float32 alpha = 0;
			ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)ref, &alpha));
			if (alpha < 1.0) {
				status = VX_SUCCESS;
			}
			else {
				status = VX_ERROR_INVALID_DIMENSION;
				vxAddLogEntry((vx_reference)node, status, "ERROR: exposure compensation alpha value is not valid\n");
			}
		}
		else
		{
			vx_float32 beta = 0.0;
			ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)ref, &beta));
			if (beta >= 1.0) {		// todo: see if there is valid upper range for beta
				status = VX_SUCCESS;
			}
			else {
				status = VX_ERROR_INVALID_DIMENSION;
				vxAddLogEntry((vx_reference)node, status, "ERROR: exposure compensation beta value is not valid\n");
			}
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
	}
	else if ((index == 2) || (index == 3))
	{ 	// matrix of type VX_TYPE_UINT32
		vx_enum type = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ATTRIBUTE_TYPE, &type, sizeof(type)));
		if (type == VX_TYPE_INT32) {
			// check matrix dimensions
			vx_size columns = 0, rows = 0;
			ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
			ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)));
			if (columns && rows) {
				status = VX_SUCCESS;
			}
			else {
				status = VX_ERROR_INVALID_DIMENSION;
				vxAddLogEntry((vx_reference)node, status, "ERROR: exp_comp_solve matrix dimensions are not valid\n");
			}
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exp_comp_solve matrix data types are not valid\n");
		}
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK exposure_comp_solvegains_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 4)
	{ // array of format float32
		vx_array arr = (vx_array)avxGetNodeParamRef(node, index);
		vx_matrix mat = (vx_matrix)avxGetNodeParamRef(node, 2);
		vx_size columns = 0, rows = 0;
		ERROR_CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
		ERROR_CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)));
		// set the capacity and item_type of the array
		vx_enum itemtype = VX_TYPE_INVALID;
		vx_size capacity = 0;
		ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
		capacity = rows; 
		ERROR_CHECK_STATUS(vxReleaseArray(&arr));
		if (itemtype == VX_TYPE_FLOAT32) {
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
			status = VX_SUCCESS;
		}
		else
		{
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exp_comp_solve array type are not valid\n");
		}
	}
	return status;
}

static vx_status VX_CALLBACK exposure_comp_solvegains_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	vx_status status = VX_FAILURE;
	vx_float32 alpha = 0, beta = 0;
	vx_uint32 *pIMat, *pNMat;
	vx_uint32 numCameras;
	CExpCompensator* exp_comp = nullptr;
	status = vxQueryNode(node, VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR, &exp_comp, sizeof(exp_comp)); if (status != VX_SUCCESS) return VX_FAILURE;

	vx_scalar scalar = (vx_scalar)parameters[0];
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &alpha));
	scalar = (vx_scalar)parameters[1];
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &beta));
	vx_size columns = 0, rows = 0;
	vx_matrix mat = (vx_matrix)parameters[2];
	ERROR_CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
	ERROR_CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)));
	pIMat = exp_comp->m_pIMat;
	ERROR_CHECK_STATUS(vxReadMatrix(mat, (void *)pIMat));
	mat = (vx_matrix)parameters[3];
	vx_size rows1 = 0;
	ERROR_CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
	ERROR_CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ATTRIBUTE_ROWS, &rows1, sizeof(rows1)));
	pNMat = exp_comp->m_pNMat;
	ERROR_CHECK_STATUS(vxReadMatrix(mat, (void *)pNMat));
	// get output array pointer
	vx_array arr = (vx_array)parameters[4];
	// set the capacity and item_type of the array
	vx_enum itemtype = VX_TYPE_INVALID;
	vx_size capacity = 0;
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &itemtype, sizeof(itemtype)));
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
	if (itemtype != VX_TYPE_FLOAT32) {
		status = VX_ERROR_INVALID_TYPE;
		vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation_gain array type should be of float32\n");
	}
	else if (capacity != rows) {
		status = VX_ERROR_INVALID_DIMENSION;
		vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation_gain array capacity not enough\n");
	}
	numCameras = (vx_uint32)columns;
	vx_size stride = 0;
	void *base = NULL;
	status = exp_comp->SolveForGains(alpha, beta, pIMat, pNMat, numCameras, arr, (vx_uint32)rows, (vx_uint32)columns);
	return status;
}

static vx_status VX_CALLBACK exposure_comp_solvegains_initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	vx_status status = VX_FAILURE;
	vx_size columns = 0, rows = 0;
	vx_matrix mat = (vx_matrix)parameters[2];
	ERROR_CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
	ERROR_CHECK_STATUS(vxQueryMatrix(mat, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)));

	CExpCompensator *pExpComp = new CExpCompensator((int)rows, (int)columns);
	vx_size size = sizeof(CExpCompensator);
	status = vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_LOCAL_DATA_SIZE, &size, sizeof(size)); if (status != VX_SUCCESS) return VX_FAILURE;
	status = vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR, &pExpComp, sizeof(pExpComp)); if (status != VX_SUCCESS) return VX_FAILURE;
	return status;
}

static vx_status VX_CALLBACK exposure_comp_solvegains_uninitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	vx_status status = VX_FAILURE;
	CExpCompensator* exp_comp = nullptr;
	status = vxQueryNode(node, VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR, &exp_comp, sizeof(exp_comp)); if (status != VX_SUCCESS) return VX_FAILURE;
	delete exp_comp;
	return status;
}


//! \brief The exposure_comp_applygains kernel publisher.
vx_status exposure_comp_solvegains_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.expcomp_solvegains",
		AMDOVX_KERNEL_STITCHING_EXPCOMP_SOLVE,
		exposure_comp_solvegains_kernel,
		5,
		exposure_comp_solvegains_input_validator,
		exposure_comp_solvegains_output_validator,
		exposure_comp_solvegains_initialize,
		exposure_comp_solvegains_uninitialize);
	ERROR_CHECK_OBJECT(kernel);
	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK exposure_comp_calcRGBErrorFn_input_validator(vx_node node, vx_uint32 index)
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
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exp_comp num_cameras scalar type should be a UINT32\n");
		}
	}
	else if (index == 1)
	{ // image of format RGBX
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGBX) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation doesn't support input image format: %4.4s\n", &input_format);
		}
		status = VX_SUCCESS;
	}
	else if (index == 2)
	{ // array object for offsets
		vx_size itemsize = 0;
		vx_size capacity = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof(capacity)));
		if (itemsize != sizeof(StitchOverlapPixelEntry)) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation gains array type should be float32\n");
		}
		else if (capacity == 0) {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation gains array capacity should be positive\n");
		}
		else {
			status = VX_SUCCESS;
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 3)
	{ // image of format U008
		if (ref){
			// check input image format and dimensions
			vx_uint32 input_width = 0, input_height = 0;
			vx_df_image input_format = VX_DF_IMAGE_VIRT;
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
			ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
			if (input_format != VX_DF_IMAGE_U8) {
				status = VX_ERROR_INVALID_TYPE;
				vxAddLogEntry((vx_reference)node, status, "ERROR: exposure_compensation mask image should be of format U008\n");
			}
		}
		status = VX_SUCCESS;
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK exposure_comp_calcRGBErrorFn_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	if (index == 4)
	{ // matrix of type VX_TYPE_INT32
		vx_enum type = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_TYPE, &type, sizeof(type)));
		if (type == VX_TYPE_INT32) {
			// check matrix dimensions
			vx_size columns = 0, rows = 0;
			ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
			ERROR_CHECK_STATUS(vxQueryMatrix((vx_matrix)ref, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)));
			if (rows < 3 * columns)	// required for R, G and B
				rows = 3 * columns;
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_COLUMNS, &columns, sizeof(columns)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_ATTRIBUTE_ROWS, &rows, sizeof(rows)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_MATRIX_TYPE, &type, sizeof(type)));
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: exposure compensation matrix type should be an float32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseMatrix((vx_matrix *)&ref));
	}
	return status;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK exposure_comp_calcRGBErrorFn_opencl_codegen(
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
	vx_size		arr_size;
	vx_uint32 num_cameras = 0;
	vx_uint32 input_width = 0, input_height = 0, output_width = 0, output_height = 0;
	vx_df_image input_format = VX_DF_IMAGE_VIRT, output_format = VX_DF_IMAGE_VIRT;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);			// input scalar - num cameras
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &num_cameras));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 1);
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	vx_array exp_data = (vx_array)avxGetNodeParamRef(node, 2);
	ERROR_CHECK_STATUS(vxQueryArray(exp_data, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_size, sizeof(arr_size)));
	ERROR_CHECK_STATUS(vxReleaseArray(&exp_data));
	vx_image mask_image = (vx_image)avxGetNodeParamRef(node, 3);
	if (mask_image != NULL){
		ERROR_CHECK_STATUS(vxQueryImage(mask_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage(mask_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage(mask_image, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
	}

	// set kernel configuration
	vx_uint32 height_one = (vx_uint32)(input_height / num_cameras);
	strcpy(opencl_kernel_function_name, "exposure_comp_calc_errorRGBfn_mask");
	opencl_work_dim = 2;
	opencl_local_work[0] = 16;
	opencl_local_work[1] = 16;
	opencl_global_work[0] = arr_size*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1];
	// kernel header and reading
	char item[8192];
	opencl_kernel_code = 
		// lookup table generated with gamma = 2.2
		"__constant uchar g_Gamma2LinearLookUp[256] = { \n"
		"	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, \n"
		"	2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 12,\n"
		"	12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 29,\n"
		"	30, 31, 31, 32, 33, 33, 34, 35, 36, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 45, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 55,\n"
		"	56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91,\n"
		"	92, 93, 95, 96, 97, 99, 100, 101, 103, 104, 105, 107, 108, 109, 111, 112, 114, 115, 117, 118, 119, 121, 122, 124, 125, 127, 128, 130, 131, 133, 135, 136,\n"
		"	138, 139, 141, 142, 144, 146, 147, 149, 151, 152, 154, 156, 157, 159, 161, 162, 164, 166, 168, 169, 171, 173, 175, 176, 178, 180, 182, 184, 186, 187, 189, 191,\n"
		"	193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 244, 246, 248, 250, 252, 255};\n";
	if (mask_image){
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"__attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"__kernel void %s(uint num_cameras,\n" // opencl_kernel_function_name
			"			uint	pIn_width, uint	pIn_height, __global uchar *pIn_buf, uint pIn_stride, uint	pIn_offs,\n"
			"			__global uchar * exp_data, uint	exp_data_offs, uint exp_data_num,\n"
			"			uint	pWt_width, uint	pWt_height, __global uchar *pWt_buf, uint pWt_stride, uint	pWt_offs,\n"
			"			__global int * pAMat, uint cols, uint rows)\n"
			"{\n"
			"	int grp_id = get_global_id(0)>>4;\n"
			"   if (grp_id < exp_data_num) {\n"
			"	__local uchar gamma2Linear[256];\n"
			"	__local uint4 sumI[256], sumJ[256];\n"
			"	uint2 offs = ((__global uint2 *)(exp_data+exp_data_offs))[grp_id];\n"
			"	uint size = (uint)(pIn_stride*%d);\n"
			"	uint wt_size = (uint)(pWt_stride*%d);\n"
			"	uint row1 = %d;\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height_one, height_one, num_cameras);
		opencl_kernel_code += item;
		opencl_kernel_code +=
			"	int lx = get_local_id(0);\n"
			"	int ly = get_local_id(1);\n"
			"	int lid = mad24(ly, (int)get_local_size(0), lx);\n"
			"	gamma2Linear[lid] = g_Gamma2LinearLookUp[lid];\n"
			"	barrier(CLK_LOCAL_MEM_FENCE);\n"
			"   sumI[lid] = (uint4)0; sumJ[lid] = (uint4)0;\n"
			"	bool isValid = ((lx<<3) < (int)(offs.s1&0x7f)) && (ly*2 < (int)((offs.s1>>7)&0x1f));\n"
			"	if (isValid) {\n"
			"		global uint *pI, *pJ;\n"
			"		uint4 maskSrc, I, J, mask; \n"
			"		uint4 Isum4, Jsum4;\n"
			"		int   gx = (lx<<3) + ((offs.s0 >> 5) & 0x3FFF);\n"
			"		int   gy = (ly<<1) + (offs.s0 >> 19);\n"
			"		uint2 cam_id = (uint2)((offs.s0 & 0x1f), ((offs.s1>>12) & 0x1f));\n"
			"		pIn_buf += pIn_offs + mad24(gy, (int)pIn_stride, (gx<<2));\n"
			"		pWt_buf += pWt_offs + mad24(gy, (int)pWt_stride, gx);\n"
			"		pI	   =  (global uint *)(pIn_buf + size*cam_id.x);\n"
			"		pJ	   =  (global uint *)(pIn_buf + size*cam_id.y);\n"
			"		maskSrc.s01	   =  *(global uint2 *)(pWt_buf + wt_size*cam_id.x);\n"
			"		maskSrc.s01	   &=  *(global uint2 *)(pWt_buf + wt_size*cam_id.y); pWt_buf += pWt_stride;\n"
			"		maskSrc.s23	   =  *(global uint2 *)(pWt_buf + wt_size*cam_id.x);\n"
			"		maskSrc.s23	   &=  *(global uint2 *)(pWt_buf + wt_size*cam_id.y);\n"
			"		char4 maskIJ = as_char4(maskSrc.s0);\n"
			"		I = vload4(0, pI);\n"
			"		J = vload4(0, pJ); \n"
			"		mask.s0	= select(0xffffffff, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xffffffff, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xffffffff, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xffffffff, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		I.s0 = gamma2Linear[I.s0&0xFF]|(gamma2Linear[(I.s0&0xFF00)>>8]<<8)|(gamma2Linear[(I.s0&0xFF0000)>>16]<<16);\n"
			"		I.s1 = gamma2Linear[I.s1&0xFF]|(gamma2Linear[(I.s1&0xFF00)>>8]<<8)|(gamma2Linear[(I.s1&0xFF0000)>>16]<<16);\n"
			"		I.s2 = gamma2Linear[I.s2&0xFF]|(gamma2Linear[(I.s2&0xFF00)>>8]<<8)|(gamma2Linear[(I.s2&0xFF0000)>>16]<<16);\n"
			"		I.s3 = gamma2Linear[I.s3&0xFF]|(gamma2Linear[(I.s3&0xFF00)>>8]<<8)|(gamma2Linear[(I.s3&0xFF0000)>>16]<<16);\n"
			"		J.s0 = gamma2Linear[J.s0&0xFF]|(gamma2Linear[(J.s0&0xFF00)>>8]<<8)|(gamma2Linear[(J.s0&0xFF0000)>>16]<<16);\n"
			"		J.s1 = gamma2Linear[J.s1&0xFF]|(gamma2Linear[(J.s1&0xFF00)>>8]<<8)|(gamma2Linear[(J.s1&0xFF0000)>>16]<<16);\n"
			"		J.s2 = gamma2Linear[J.s2&0xFF]|(gamma2Linear[(J.s2&0xFF00)>>8]<<8)|(gamma2Linear[(J.s2&0xFF0000)>>16]<<16);\n"
			"		J.s3 = gamma2Linear[J.s3&0xFF]|(gamma2Linear[(J.s3&0xFF00)>>8]<<8)|(gamma2Linear[(J.s3&0xFF0000)>>16]<<16);\n"
			"		Isum4 = convert_uint4(as_uchar4(I.s0 & mask.s0));		Jsum4 = convert_uint4(as_uchar4(J.s0 & mask.s0));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s1 & mask.s1));		Jsum4 += convert_uint4(as_uchar4(J.s1 & mask.s1));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s2 & mask.s2));		Jsum4 += convert_uint4(as_uchar4(J.s2 & mask.s2));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s3 & mask.s3));		Jsum4 += convert_uint4(as_uchar4(J.s3 & mask.s3));\n"
			"		I = vload4(1, pI);\n"
			"		J = vload4(1, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s1);\n"
			"		mask.s0	= select(0xffffffff, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xffffffff, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xffffffff, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xffffffff, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		I.s0 = gamma2Linear[I.s0&0xFF]|(gamma2Linear[(I.s0&0xFF00)>>8]<<8)|(gamma2Linear[(I.s0&0xFF0000)>>16]<<16);\n"
			"		I.s1 = gamma2Linear[I.s1&0xFF]|(gamma2Linear[(I.s1&0xFF00)>>8]<<8)|(gamma2Linear[(I.s1&0xFF0000)>>16]<<16);\n"
			"		I.s2 = gamma2Linear[I.s2&0xFF]|(gamma2Linear[(I.s2&0xFF00)>>8]<<8)|(gamma2Linear[(I.s2&0xFF0000)>>16]<<16);\n"
			"		I.s3 = gamma2Linear[I.s3&0xFF]|(gamma2Linear[(I.s3&0xFF00)>>8]<<8)|(gamma2Linear[(I.s3&0xFF0000)>>16]<<16);\n"
			"		J.s0 = gamma2Linear[J.s0&0xFF]|(gamma2Linear[(J.s0&0xFF00)>>8]<<8)|(gamma2Linear[(J.s0&0xFF0000)>>16]<<16);\n"
			"		J.s1 = gamma2Linear[J.s1&0xFF]|(gamma2Linear[(J.s1&0xFF00)>>8]<<8)|(gamma2Linear[(J.s1&0xFF0000)>>16]<<16);\n"
			"		J.s2 = gamma2Linear[J.s2&0xFF]|(gamma2Linear[(J.s2&0xFF00)>>8]<<8)|(gamma2Linear[(J.s2&0xFF0000)>>16]<<16);\n"
			"		J.s3 = gamma2Linear[J.s3&0xFF]|(gamma2Linear[(J.s3&0xFF00)>>8]<<8)|(gamma2Linear[(J.s3&0xFF0000)>>16]<<16);\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s0 & mask.s0));		Jsum4 += convert_uint4(as_uchar4(J.s0 & mask.s0));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s1 & mask.s1));		Jsum4 += convert_uint4(as_uchar4(J.s1 & mask.s1));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s2 & mask.s2));		Jsum4 += convert_uint4(as_uchar4(J.s2 & mask.s2));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s3 & mask.s3));		Jsum4 += convert_uint4(as_uchar4(J.s3 & mask.s3));\n"
			"		pI += (pIn_stride>>2); pJ += (pIn_stride>>2);\n"
			"		I = vload4(0, pI);\n"
			"		J = vload4(0, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s2);\n"
			"		mask.s0	= select(0xffffffff, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xffffffff, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xffffffff, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xffffffff, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		I.s0 = gamma2Linear[I.s0&0xFF]|(gamma2Linear[(I.s0&0xFF00)>>8]<<8)|(gamma2Linear[(I.s0&0xFF0000)>>16]<<16);\n"
			"		I.s1 = gamma2Linear[I.s1&0xFF]|(gamma2Linear[(I.s1&0xFF00)>>8]<<8)|(gamma2Linear[(I.s1&0xFF0000)>>16]<<16);\n"
			"		I.s2 = gamma2Linear[I.s2&0xFF]|(gamma2Linear[(I.s2&0xFF00)>>8]<<8)|(gamma2Linear[(I.s2&0xFF0000)>>16]<<16);\n"
			"		I.s3 = gamma2Linear[I.s3&0xFF]|(gamma2Linear[(I.s3&0xFF00)>>8]<<8)|(gamma2Linear[(I.s3&0xFF0000)>>16]<<16);\n"
			"		J.s0 = gamma2Linear[J.s0&0xFF]|(gamma2Linear[(J.s0&0xFF00)>>8]<<8)|(gamma2Linear[(J.s0&0xFF0000)>>16]<<16);\n"
			"		J.s1 = gamma2Linear[J.s1&0xFF]|(gamma2Linear[(J.s1&0xFF00)>>8]<<8)|(gamma2Linear[(J.s1&0xFF0000)>>16]<<16);\n"
			"		J.s2 = gamma2Linear[J.s2&0xFF]|(gamma2Linear[(J.s2&0xFF00)>>8]<<8)|(gamma2Linear[(J.s2&0xFF0000)>>16]<<16);\n"
			"		J.s3 = gamma2Linear[J.s3&0xFF]|(gamma2Linear[(J.s3&0xFF00)>>8]<<8)|(gamma2Linear[(J.s3&0xFF0000)>>16]<<16);\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s0 & mask.s0));		Jsum4 += convert_uint4(as_uchar4(J.s0 & mask.s0));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s1 & mask.s1));		Jsum4 += convert_uint4(as_uchar4(J.s1 & mask.s1));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s2 & mask.s2));		Jsum4 += convert_uint4(as_uchar4(J.s2 & mask.s2));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s3 & mask.s3));		Jsum4 += convert_uint4(as_uchar4(J.s3 & mask.s3));\n"
			"		I = vload4(1, pI); \n"
			"		J = vload4(1, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s3);\n"
			"		mask.s0	= select(0xffffffff, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xffffffff, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xffffffff, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xffffffff, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		I.s0 = gamma2Linear[I.s0&0xFF]|(gamma2Linear[(I.s0&0xFF00)>>8]<<8)|(gamma2Linear[(I.s0&0xFF0000)>>16]<<16);\n"
			"		I.s1 = gamma2Linear[I.s1&0xFF]|(gamma2Linear[(I.s1&0xFF00)>>8]<<8)|(gamma2Linear[(I.s1&0xFF0000)>>16]<<16);\n"
			"		I.s2 = gamma2Linear[I.s2&0xFF]|(gamma2Linear[(I.s2&0xFF00)>>8]<<8)|(gamma2Linear[(I.s2&0xFF0000)>>16]<<16);\n"
			"		I.s3 = gamma2Linear[I.s3&0xFF]|(gamma2Linear[(I.s3&0xFF00)>>8]<<8)|(gamma2Linear[(I.s3&0xFF0000)>>16]<<16);\n"
			"		J.s0 = gamma2Linear[J.s0&0xFF]|(gamma2Linear[(J.s0&0xFF00)>>8]<<8)|(gamma2Linear[(J.s0&0xFF0000)>>16]<<16);\n"
			"		J.s1 = gamma2Linear[J.s1&0xFF]|(gamma2Linear[(J.s1&0xFF00)>>8]<<8)|(gamma2Linear[(J.s1&0xFF0000)>>16]<<16);\n"
			"		J.s2 = gamma2Linear[J.s2&0xFF]|(gamma2Linear[(J.s2&0xFF00)>>8]<<8)|(gamma2Linear[(J.s2&0xFF0000)>>16]<<16);\n"
			"		J.s3 = gamma2Linear[J.s3&0xFF]|(gamma2Linear[(J.s3&0xFF00)>>8]<<8)|(gamma2Linear[(J.s3&0xFF0000)>>16]<<16);\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s0 & mask.s0));		Jsum4 += convert_uint4(as_uchar4(J.s0 & mask.s0));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s1 & mask.s1));		Jsum4 += convert_uint4(as_uchar4(J.s1 & mask.s1));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s2 & mask.s2));		Jsum4 += convert_uint4(as_uchar4(J.s2 & mask.s2));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s3 & mask.s3));		Jsum4 += convert_uint4(as_uchar4(J.s3 & mask.s3));\n"
			"		sumI[lid] = Isum4; sumJ[lid] = Jsum4;\n"
			"		barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	else
	{
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"__attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"__kernel void %s(uint num_cameras,\n" // opencl_kernel_function_name
			"			uint	pIn_width, uint	pIn_height, __global uchar *pIn_buf, uint pIn_stride, uint	pIn_offs,\n"
			"			__global uchar * exp_data, uint	exp_data_offs, uint exp_data_num,\n"
			"			__global int * pAMat, uint cols, uint rows)\n"
			"{\n"
			"	int grp_id = get_global_id(0)>>4;\n"
			"   if (grp_id < exp_data_num) {\n"
			"	__local uchar gamma2Linear[256];\n"
			"	__local uint4  sumI[256], sumJ[256];\n"
			"	uint2 offs = ((__global uint2 *)(exp_data+exp_data_offs))[grp_id];\n"
			"	uint size = (uint)(pIn_stride*%d);\n"
			"	uint row1 = %d;\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height_one, num_cameras);
		opencl_kernel_code += item;
		opencl_kernel_code +=
			"	int lx = get_local_id(0);\n"
			"	int ly = get_local_id(1);\n"
			"	int lid = mad24(ly, (int)get_local_size(0), lx);\n"
			"	gamma2Linear[lid] = g_Gamma2LinearLookUp[lid];\n"
			"	barrier(CLK_LOCAL_MEM_FENCE);\n"
			"   sumI[lid] = (uint4)0; sumJ[lid] = (uint4)0;\n"
			"	bool isValid = ((lx<<3) < (int)(offs.s1&0x7f)) && (ly*2 < (int)((offs.s1>>7)&0x1f));\n"
			"	if (isValid) {\n"
			"		global uint *pI, *pJ;\n"
			"		uint4  I, J, mask; \n"
			"		uint4 Isum4, Jsum4;\n"
			"		int   gx = (lx<<3) + ((offs.s0 >> 5) & 0x3FFF);\n"
			"		int   gy = (ly<<1) + (offs.s0 >> 19);\n"
			"		uint2 cam_id = (uint2)((offs.s0 & 0x1f), ((offs.s1>>12) & 0x1f));\n"
			"		pIn_buf += pIn_offs + mad24(gy, (int)pIn_stride, (gx<<2));\n"
			"		pI	   =  (global uint *)(pIn_buf + size*cam_id.x);\n"
			"		pJ	   =  (global uint *)(pIn_buf + size*cam_id.y);\n"
			"		I = vload4(0, pI);\n"
			"		J = vload4(0, pJ); \n"
			"		mask.s0	= select(0xffffffff, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000)));\n"
			"		mask.s1	= select(0xffffffff, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000)));\n"
			"		mask.s2	= select(0xffffffff, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000)));\n"
			"		mask.s3	= select(0xffffffff, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000)));\n"
			"		I.s0 = gamma2Linear[I.s0&0xFF]|(gamma2Linear[(I.s0&0xFF00)>>8]<<8)|(gamma2Linear[(I.s0&0xFF0000)>>16]<<16);\n"
			"		I.s1 = gamma2Linear[I.s1&0xFF]|(gamma2Linear[(I.s1&0xFF00)>>8]<<8)|(gamma2Linear[(I.s1&0xFF0000)>>16]<<16);\n"
			"		I.s2 = gamma2Linear[I.s2&0xFF]|(gamma2Linear[(I.s2&0xFF00)>>8]<<8)|(gamma2Linear[(I.s2&0xFF0000)>>16]<<16);\n"
			"		I.s3 = gamma2Linear[I.s3&0xFF]|(gamma2Linear[(I.s3&0xFF00)>>8]<<8)|(gamma2Linear[(I.s3&0xFF0000)>>16]<<16);\n"
			"		J.s0 = gamma2Linear[J.s0&0xFF]|(gamma2Linear[(J.s0&0xFF00)>>8]<<8)|(gamma2Linear[(J.s0&0xFF0000)>>16]<<16);\n"
			"		J.s1 = gamma2Linear[J.s1&0xFF]|(gamma2Linear[(J.s1&0xFF00)>>8]<<8)|(gamma2Linear[(J.s1&0xFF0000)>>16]<<16);\n"
			"		J.s2 = gamma2Linear[J.s2&0xFF]|(gamma2Linear[(J.s2&0xFF00)>>8]<<8)|(gamma2Linear[(J.s2&0xFF0000)>>16]<<16);\n"
			"		J.s3 = gamma2Linear[J.s3&0xFF]|(gamma2Linear[(J.s3&0xFF00)>>8]<<8)|(gamma2Linear[(J.s3&0xFF0000)>>16]<<16);\n"
			"		Isum4 = convert_uint4(as_uchar4(I.s0 & mask.s0));		Jsum4 = convert_uint4(as_uchar4(J.s0 & mask.s0));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s1 & mask.s1));		Jsum4 += convert_uint4(as_uchar4(J.s1 & mask.s1));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s2 & mask.s2));		Jsum4 += convert_uint4(as_uchar4(J.s2 & mask.s2));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s3 & mask.s3));		Jsum4 += convert_uint4(as_uchar4(J.s3 & mask.s3));\n"
			"		I = vload4(1, pI);\n"
			"		J = vload4(1, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s1);\n"
			"		mask.s0	= select(0xffffffff, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000)));\n"
			"		mask.s1	= select(0xffffffff, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000)));\n"
			"		mask.s2	= select(0xffffffff, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000)));\n"
			"		mask.s3	= select(0xffffffff, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000)));\n"
			"		I.s0 = gamma2Linear[I.s0&0xFF]|(gamma2Linear[(I.s0&0xFF00)>>8]<<8)|(gamma2Linear[(I.s0&0xFF0000)>>16]<<16);\n"
			"		I.s1 = gamma2Linear[I.s1&0xFF]|(gamma2Linear[(I.s1&0xFF00)>>8]<<8)|(gamma2Linear[(I.s1&0xFF0000)>>16]<<16);\n"
			"		I.s2 = gamma2Linear[I.s2&0xFF]|(gamma2Linear[(I.s2&0xFF00)>>8]<<8)|(gamma2Linear[(I.s2&0xFF0000)>>16]<<16);\n"
			"		I.s3 = gamma2Linear[I.s3&0xFF]|(gamma2Linear[(I.s3&0xFF00)>>8]<<8)|(gamma2Linear[(I.s3&0xFF0000)>>16]<<16);\n"
			"		J.s0 = gamma2Linear[J.s0&0xFF]|(gamma2Linear[(J.s0&0xFF00)>>8]<<8)|(gamma2Linear[(J.s0&0xFF0000)>>16]<<16);\n"
			"		J.s1 = gamma2Linear[J.s1&0xFF]|(gamma2Linear[(J.s1&0xFF00)>>8]<<8)|(gamma2Linear[(J.s1&0xFF0000)>>16]<<16);\n"
			"		J.s2 = gamma2Linear[J.s2&0xFF]|(gamma2Linear[(J.s2&0xFF00)>>8]<<8)|(gamma2Linear[(J.s2&0xFF0000)>>16]<<16);\n"
			"		J.s3 = gamma2Linear[J.s3&0xFF]|(gamma2Linear[(J.s3&0xFF00)>>8]<<8)|(gamma2Linear[(J.s3&0xFF0000)>>16]<<16);\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s0 & mask.s0));		Jsum4 += convert_uint4(as_uchar4(J.s0 & mask.s0));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s1 & mask.s1));		Jsum4 += convert_uint4(as_uchar4(J.s1 & mask.s1));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s2 & mask.s2));		Jsum4 += convert_uint4(as_uchar4(J.s2 & mask.s2));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s3 & mask.s3));		Jsum4 += convert_uint4(as_uchar4(J.s3 & mask.s3));\n"
			"		pI += (pIn_stride>>2); pJ += (pIn_stride>>2);\n"
			"		I = vload4(0, pI);\n"
			"		J = vload4(0, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s2);\n"
			"		mask.s0	= select(0xffffffff, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xffffffff, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xffffffff, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xffffffff, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		I.s0 = gamma2Linear[I.s0&0xFF]|(gamma2Linear[(I.s0&0xFF00)>>8]<<8)|(gamma2Linear[(I.s0&0xFF0000)>>16]<<16);\n"
			"		I.s1 = gamma2Linear[I.s1&0xFF]|(gamma2Linear[(I.s1&0xFF00)>>8]<<8)|(gamma2Linear[(I.s1&0xFF0000)>>16]<<16);\n"
			"		I.s2 = gamma2Linear[I.s2&0xFF]|(gamma2Linear[(I.s2&0xFF00)>>8]<<8)|(gamma2Linear[(I.s2&0xFF0000)>>16]<<16);\n"
			"		I.s3 = gamma2Linear[I.s3&0xFF]|(gamma2Linear[(I.s3&0xFF00)>>8]<<8)|(gamma2Linear[(I.s3&0xFF0000)>>16]<<16);\n"
			"		J.s0 = gamma2Linear[J.s0&0xFF]|(gamma2Linear[(J.s0&0xFF00)>>8]<<8)|(gamma2Linear[(J.s0&0xFF0000)>>16]<<16);\n"
			"		J.s1 = gamma2Linear[J.s1&0xFF]|(gamma2Linear[(J.s1&0xFF00)>>8]<<8)|(gamma2Linear[(J.s1&0xFF0000)>>16]<<16);\n"
			"		J.s2 = gamma2Linear[J.s2&0xFF]|(gamma2Linear[(J.s2&0xFF00)>>8]<<8)|(gamma2Linear[(J.s2&0xFF0000)>>16]<<16);\n"
			"		J.s3 = gamma2Linear[J.s3&0xFF]|(gamma2Linear[(J.s3&0xFF00)>>8]<<8)|(gamma2Linear[(J.s3&0xFF0000)>>16]<<16);\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s0 & mask.s0));		Jsum4 += convert_uint4(as_uchar4(J.s0 & mask.s0));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s1 & mask.s1));		Jsum4 += convert_uint4(as_uchar4(J.s1 & mask.s1));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s2 & mask.s2));		Jsum4 += convert_uint4(as_uchar4(J.s2 & mask.s2));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s3 & mask.s3));		Jsum4 += convert_uint4(as_uchar4(J.s3 & mask.s3));\n"
			"		I = vload4(1, pI); \n"
			"		J = vload4(1, pJ); \n"
			"		maskIJ = as_char4(maskSrc.s3);\n"
			"		mask.s0	= select(0xffffffff, 0u, ((I.s0==0x80000000) | (J.s0==0x80000000))) & (int)maskIJ.s0;\n"
			"		mask.s1	= select(0xffffffff, 0u, ((I.s1==0x80000000) | (J.s1==0x80000000))) & (int)maskIJ.s1;\n"
			"		mask.s2	= select(0xffffffff, 0u, ((I.s2==0x80000000) | (J.s2==0x80000000))) & (int)maskIJ.s2;\n"
			"		mask.s3	= select(0xffffffff, 0u, ((I.s3==0x80000000) | (J.s3==0x80000000))) & (int)maskIJ.s3;\n"
			"		I.s0 = gamma2Linear[I.s0&0xFF]|(gamma2Linear[(I.s0&0xFF00)>>8]<<8)|(gamma2Linear[(I.s0&0xFF0000)>>16]<<16);\n"
			"		I.s1 = gamma2Linear[I.s1&0xFF]|(gamma2Linear[(I.s1&0xFF00)>>8]<<8)|(gamma2Linear[(I.s1&0xFF0000)>>16]<<16);\n"
			"		I.s2 = gamma2Linear[I.s2&0xFF]|(gamma2Linear[(I.s2&0xFF00)>>8]<<8)|(gamma2Linear[(I.s2&0xFF0000)>>16]<<16);\n"
			"		I.s3 = gamma2Linear[I.s3&0xFF]|(gamma2Linear[(I.s3&0xFF00)>>8]<<8)|(gamma2Linear[(I.s3&0xFF0000)>>16]<<16);\n"
			"		J.s0 = gamma2Linear[J.s0&0xFF]|(gamma2Linear[(J.s0&0xFF00)>>8]<<8)|(gamma2Linear[(J.s0&0xFF0000)>>16]<<16);\n"
			"		J.s1 = gamma2Linear[J.s1&0xFF]|(gamma2Linear[(J.s1&0xFF00)>>8]<<8)|(gamma2Linear[(J.s1&0xFF0000)>>16]<<16);\n"
			"		J.s2 = gamma2Linear[J.s2&0xFF]|(gamma2Linear[(J.s2&0xFF00)>>8]<<8)|(gamma2Linear[(J.s2&0xFF0000)>>16]<<16);\n"
			"		J.s3 = gamma2Linear[J.s3&0xFF]|(gamma2Linear[(J.s3&0xFF00)>>8]<<8)|(gamma2Linear[(J.s3&0xFF0000)>>16]<<16);\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s0 & mask.s0));		Jsum4 += convert_uint4(as_uchar4(J.s0 & mask.s0));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s1 & mask.s1));		Jsum4 += convert_uint4(as_uchar4(J.s1 & mask.s1));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s2 & mask.s2));		Jsum4 += convert_uint4(as_uchar4(J.s2 & mask.s2));\n"
			"		Isum4 += convert_uint4(as_uchar4(I.s3 & mask.s3));		Jsum4 += convert_uint4(as_uchar4(J.s3 & mask.s3));\n"
			"		sumI[lid] = Isum4; sumJ[lid] = Jsum4;\n"
			"		barrier(CLK_LOCAL_MEM_FENCE);\n";
	}
	opencl_kernel_code +=
		"		// aggregate sum and count from all threads\n"
		"		if (lid < 128)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+128];\n"
		"			sumJ[lid]	+= sumJ[lid+128];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		if (lid < 64)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+64];\n"
		"			sumJ[lid]	+= sumJ[lid+64];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		if (lid < 32)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+32];\n"
		"			sumJ[lid]	+= sumJ[lid+32];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		if (lid < 16)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+16];\n"
		"			sumJ[lid]	+= sumJ[lid+16];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		if (lid < 8)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+8];\n"
		"			sumJ[lid]	+= sumJ[lid+8];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		if (lid < 4)\n"
		"		{\n"
		"			sumI[lid]	+= sumI[lid+4];\n"
		"			sumJ[lid]	+= sumJ[lid+4];\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		uint idx1;\n"
		"		if (!lid)\n"
		"		{\n"
		"			idx1 = mad24(cam_id.x, cols, cam_id.y);\n"
		"			uint4 sum	= sumI[0] + sumI[1] + sumI[2] + sumI[3];\n"
		"			atomic_add(&pAMat[idx1], (int)(sum.s0*0.0625f));\n"
		"			idx1 += mul24(cols, row1);\n"
		"			atomic_add(&pAMat[idx1], (int)(sum.s1*0.0625f));\n"
		"			idx1 += mul24(cols, row1);\n"
		"			atomic_add(&pAMat[idx1], (int)(sum.s2*0.0625f));\n"
		"		}\n"
		"		else if (lid == 1){\n"
		"			idx1 = mad24(cam_id.y, cols, cam_id.x);\n"
		"			uint4 sum = sumJ[0] + sumJ[1] + sumJ[2] + sumJ[3];\n"
		"			atomic_add(&pAMat[idx1], (int)(sum.s0*0.0625f));\n"
		"			idx1 += mul24(cols, row1);\n"
		"			atomic_add(&pAMat[idx1], (int)(sum.s1*0.0625f));\n"
		"			idx1 += mul24(cols, row1);\n"
		"			atomic_add(&pAMat[idx1], (int)(sum.s2*0.0625f));\n"
		"		}\n"
		"	}\n"
		"	}\n"
		"}\n";
	if (mask_image)ERROR_CHECK_STATUS(vxReleaseImage(&mask_image));
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK exposure_comp_calcRGBErrorFn_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}


//! \brief The exposure_comp_applygains kernel publisher.
vx_status exposure_comp_calcRGBErrorFn_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.expcomp_compute_gainmatrix_rgb",
		AMDOVX_KERNEL_STITCHING_EXPCOMP_COMPUTE_GAINMAT_RGB,
		exposure_comp_calcRGBErrorFn_kernel,
		5,
		exposure_comp_calcRGBErrorFn_input_validator,
		exposure_comp_calcRGBErrorFn_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	// set codegen for opencl
	amd_kernel_query_target_support_f query_target_support_f = exposure_comp_calcErrorFn_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = exposure_comp_calcRGBErrorFn_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = exposure_comp_calcErrorFn_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));
	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_OPTIONAL));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_MATRIX, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	return VX_SUCCESS;
}


//////////////////////////////////////////////////////////////////////
// Calculate buffer sizes and generate data in buffers for exposure compensation
//   CalculateLargestExpCompBufferSizes  - useful when reinitialize is enabled
//   CalculateSmallestExpCompBufferSizes - useful when reinitialize is disabled
//   GenerateExpCompBuffers              - generate tables

vx_status CalculateLargestExpCompBufferSizes(
	vx_uint32 numCamera,                    // [in] number of cameras
	vx_uint32 eqrWidth,                     // [in] output equirectangular image width
	vx_uint32 eqrHeight,                    // [in] output equirectangular image height
	vx_size * validTableEntryCount,         // [out] number of entries needed by expComp valid table
	vx_size * overlapTableEntryCount        // [out] number of entries needed by expComp overlap table
	)
{
	vx_size num128x32blocks = ((eqrWidth + 127) >> 7) * ((eqrHeight + 31) >> 5);
	*validTableEntryCount = num128x32blocks * numCamera;
	*overlapTableEntryCount = num128x32blocks * numCamera * (numCamera - 1) / 2;
	return VX_SUCCESS;
}

vx_status CalculateSmallestExpCompBufferSizes(
	vx_uint32 numCamera,                           // [in] number of cameras
	vx_uint32 eqrWidth,                            // [in] output equirectangular image width
	vx_uint32 eqrHeight,                           // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,            // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,   // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,         // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32]
	const vx_uint32 * paddedPixelCamMap,           // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,  // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,        // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32](optional)
	vx_size * validTableEntryCount,                // [out] number of entries needed by expComp valid table
	vx_size * overlapTableEntryCount               // [out] number of entries needed by expComp overlap table
	)
{
	const vx_rectangle_t * const * overlapRegion = paddedPixelCamMap ? overlapPadded : overlapValid;

	// count validTable entries
	vx_uint32 validEntryCount = 0;
	for (vx_uint32 i = 0; i < numCamera; i++) {
		vx_uint32 camMaskBit = (1 << i);
		vx_uint32 start_x = overlapValid[i][i].start_x, end_x = overlapValid[i][i].end_x;
		vx_uint32 start_y = overlapValid[i][i].start_y, end_y = overlapValid[i][i].end_y;
		if ((start_x < end_x) && (start_y < end_y))	{
			for (vx_uint32 ys = start_y; ys < end_y; ys += 32) {
				for (vx_uint32 xs = start_x; xs < end_x; xs += 128) {
					vx_uint32 xe = (xs + 128) < end_x ? (xs + 128) : end_x;
					vx_uint32 ye = (ys + 32) < end_y ? (ys + 32) : end_y;
					// count valid pixels
					vx_int32 count = 0;
					for (vx_uint32 y = ys; y < ye; y++) {
						for (vx_uint32 x = xs; x < xe; x++) {
							if ((validPixelCamMap[y * eqrWidth + x] & camMaskBit) == camMaskBit) {
								count = 1;
								break;
							}
						}
						if (count > 0)
							break;
					}
					if (count > 0) {
						validEntryCount++;
					}
				}
			}
		}
	}

	// count overlapTable entries
	vx_uint32 overlapEntryCount = 0;
	for (vx_uint32 i = 1; i < numCamera; i++) {
		for (vx_uint32 j = 0; j < i; j++) {
			vx_uint32 overlapMaskBits = (1 << i) | (1 << j);
			vx_uint32 start_x = overlapRegion[i][j].start_x, end_x = overlapRegion[i][j].end_x;
			vx_uint32 start_y = overlapRegion[i][j].start_y, end_y = overlapRegion[i][j].end_y;
			if ((start_x < end_x) && (start_y < end_y))	{
				for (vx_uint32 ys = start_y; ys < end_y; ys += 32) {
					for (vx_uint32 xs = start_x; xs < end_x; xs += 128) {
						vx_uint32 xe = (xs + 128) < end_x ? (xs + 128) : end_x;
						vx_uint32 ye = (ys + 32) < end_y ? (ys + 32) : end_y;
						// count valid pixels
						vx_int32 count = 0;
						for (vx_uint32 y = ys; y < ye; y++) {
							for (vx_uint32 x = xs; x < xe; x++) {
								if ((validPixelCamMap[y * eqrWidth + x] & overlapMaskBits) == overlapMaskBits) {
									count = 1;
									break;
								}
							}
							if (count > 0)
								break;
						}
						if (count > 0) {
							overlapEntryCount++;
						}
					}
				}
			}
		}
	}

	// updated output entry counts
	*validTableEntryCount = validEntryCount;
	*overlapTableEntryCount = overlapEntryCount;

	return VX_SUCCESS;
}

vx_status GenerateExpCompBuffers(
	vx_uint32 numCamera,                           // [in] number of cameras
	vx_uint32 eqrWidth,                            // [in] output equirectangular image width
	vx_uint32 eqrHeight,                           // [in] output equirectangular image height
	const vx_uint32 * validPixelCamMap,            // [in] valid pixel camera index map: size: [eqrWidth * eqrHeight]
	const vx_rectangle_t * const * overlapValid,   // [in] overlap regions: overlapValid[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i
	const vx_uint32 * validCamOverlapInfo,         // [in] camera overlap info - use "validCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32]
	const vx_uint32 * paddedPixelCamMap,           // [in] padded pixel camera index map: size: [eqrWidth * eqrHeight](optional)
	const vx_rectangle_t * const * overlapPadded,  // [in] overlap regions: overlapPadded[cam_i][cam_j] for overlap of cam_i and cam_j, cam_j <= cam_i(optional)
	const vx_uint32 * paddedCamOverlapInfo,        // [in] camera overlap info - use "paddedCamOverlapInfo[cam_i] & (1 << cam_j)": size: [32](optional)
	vx_size validTableSize,                        // [in] size of valid table, in terms of number of entries
	vx_size overlapTableSize,                      // [in] size of overlap table, in terms of number of entries
	StitchExpCompCalcEntry * validTable,           // [out] expComp valid table
	StitchOverlapPixelEntry * overlapTable,        // [out] expComp overlap table
	vx_size * validTableEntryCount,                // [out] number of entries needed by expComp valid table
	vx_size * overlapTableEntryCount,              // [out] number of entries needed by expComp overlap table
	vx_int32 * overlapPixelCountMatrix             // [out] expComp overlap pixel count matrix: size: [numCamera * numCamera]
	)
{
	const vx_rectangle_t * const * overlapRegion = paddedPixelCamMap ? overlapPadded : overlapValid;

	// generate validTable
	vx_uint32 validEntryCount = 0;
	for (vx_uint32 i = 0; i < numCamera; i++) {
		vx_uint32 camMaskBit = (1 << i);
		vx_uint32 start_x = overlapValid[i][i].start_x, end_x = overlapValid[i][i].end_x;
		vx_uint32 start_y = overlapValid[i][i].start_y, end_y = overlapValid[i][i].end_y;
		if ((start_x < end_x) && (start_y < end_y))	{
			for (vx_uint32 ys = start_y; ys < end_y; ys += 32) {
				for (vx_uint32 xs = start_x; xs < end_x; xs += 128) {
					vx_uint32 xe = (xs + 128) < end_x ? (xs + 128) : end_x;
					vx_uint32 ye = (ys +  32) < end_y ? (ys +  32) : end_y;
					// count valid pixels
					vx_int32 count = 0;
					for (vx_uint32 y = ys; y < ye; y++) {
						for (vx_uint32 x = xs; x < xe; x++) {
							if ((validPixelCamMap[y * eqrWidth + x] & camMaskBit) == camMaskBit) {
								count = 1;
								break;
							}
						}
						if (count > 0)
							break;
					}
					if (count > 0) {
						// add valid entry
						if (validEntryCount < validTableSize){
							StitchExpCompCalcEntry validEntry;
							validEntry.camId = i;
							validEntry.dstX = (xs >> 3);
							validEntry.dstY = (ys >> 1);
							validEntry.start_x = xs & 7;
							validEntry.start_y = ys & 1;
							validEntry.end_x = xe - xs - 1;
							validEntry.end_y = ye - ys - 1;
							validTable[validEntryCount] = validEntry;
						}
						validEntryCount++;
					}
				}
			}
		}
	}

	// generate overlapTable and overlapPixelCountMatrix
	memset(overlapPixelCountMatrix, 0, numCamera * numCamera * sizeof(vx_int32));
	vx_uint32 overlapEntryCount = 0;
	for (vx_uint32 i = 1; i < numCamera; i++) {
		for (vx_uint32 j = 0; j < i; j++) {
			vx_uint32 overlapMaskBits = (1 << i) | (1 << j);
			vx_uint32 start_x = overlapRegion[i][j].start_x, end_x = overlapRegion[i][j].end_x;
			vx_uint32 start_y = overlapRegion[i][j].start_y, end_y = overlapRegion[i][j].end_y;
			if ((start_x < end_x) && (start_y < end_y))	{
				for (vx_uint32 ys = start_y; ys < end_y; ys += 32) {
					for (vx_uint32 xs = start_x; xs < end_x; xs += 128) {
						vx_uint32 xe = (xs + 128) < end_x ? (xs + 128) : end_x;
						vx_uint32 ye = (ys +  32) < end_y ? (ys +  32) : end_y;
						// count valid pixels
						vx_int32 count = 0;
						for (vx_uint32 y = ys; y < ye; y++) {
							for (vx_uint32 x = xs; x < xe; x++) {
								if ((validPixelCamMap[y * eqrWidth + x] & overlapMaskBits) == overlapMaskBits) {
									count++;
								}
							}
						}
						if (count > 0) {
							overlapPixelCountMatrix[i * numCamera + j] += count;
							overlapPixelCountMatrix[j * numCamera + i] += count;
							// add overlapTable entry
							if (overlapTable){
								if (overlapEntryCount < overlapTableSize) {
									StitchOverlapPixelEntry overlapEntry;
									overlapEntry.camId0 = i;
									overlapEntry.start_x = xs;
									overlapEntry.start_y = ys;
									overlapEntry.end_x = xe - xs - 1;
									overlapEntry.end_y = ye - ys - 1;
									overlapEntry.camId1 = j;
									overlapEntry.camId2 = 0x1F;
									overlapEntry.camId3 = 0x1F;
									overlapEntry.camId4 = 0x1F;
									overlapTable[overlapEntryCount] = overlapEntry;
								}
								overlapEntryCount++;
							}
						}
					}
				}
			}
		}
	}

	// check for buffer overflow error condition and updated output entry counts
	if (validEntryCount > validTableSize || (overlapTable && (overlapEntryCount > overlapTableSize))) {
		return VX_ERROR_NOT_SUFFICIENT;
	}
	*validTableEntryCount = validEntryCount;
	*overlapTableEntryCount = overlapEntryCount;

	return VX_SUCCESS;
}
