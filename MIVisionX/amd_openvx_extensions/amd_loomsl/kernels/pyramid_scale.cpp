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
#include "pyramid_scale.h"
#include "multiband_blender.h"

//! \brief The input validator callback.
static vx_status VX_CALLBACK half_scale_gaussian_input_validator(vx_node node, vx_uint32 index)
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
			vxAddLogEntry((vx_reference)node, status, "ERROR: half_scale_gaussian nCam scalar type should be a UINT32\n");
		}
	}
	if (index == 1)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: half_scale_gaussian nCam scalar type should be a UINT32\n");
		}
	}
	if (index == 2)
	{ // array object of StitchBlendValidEntry type
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchBlendValidEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: warp array element (StitchBlendValidEntry) size should be 32 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	if (index == 3)
	{ // image of format U008 or RGBX
		vx_df_image format = VX_DF_IMAGE_VIRT;
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		if (format == VX_DF_IMAGE_RGBX || format == VX_DF_IMAGE_U8 || format == VX_DF_IMAGE_S16 ) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: half_scale_gaussian doesn't support input image format: %4.4s\n", &format);
		}
		if (!width || !height) {
			status = VX_ERROR_INVALID_DIMENSION;
		}
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK half_scale_gaussian_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 4)
	{ // image of format U016 or RGBX
		// get image configuration
		vx_image image = (vx_image)avxGetNodeParamRef(node, 3);
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
		 
		//not possible to ensure the following when we go beyond 4 levels
		if (output_width < (input_width + 1) >> 1) {
			output_width = (input_width + 1) >> 1;
		}
		if (output_height < (input_height + 1) >> 1) {
			output_height = (input_height + 1) >> 1;
		}
		if ((output_format != VX_DF_IMAGE_U8) && (output_format != VX_DF_IMAGE_S16) && (output_format != VX_DF_IMAGE_RGBX)) {
			output_format = input_format;
		}
		if ((input_format == VX_DF_IMAGE_S16) && (output_format != VX_DF_IMAGE_S16)){
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: half_scale_gaussian doesn't support output image format: %4.4s\n", &output_format);
		}
		else
		{
			// set output image meta data
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
			ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
			status = VX_SUCCESS;
		}
	}
	return status;
}

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK half_scale_gaussian_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK half_scale_gaussian_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	vx_uint32 arr_offset;
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 2);				// input array
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 1);			// input scalar - grayscale compute method
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &arr_offset));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	ERROR_CHECK_OBJECT(arr);
//	printf("ArrOff: %d\n", arr_offset);
	StitchBlendValidEntry *pBlendArr = nullptr;
	vx_size stride_blend_arr = sizeof(StitchBlendValidEntry);
	ERROR_CHECK_STATUS(vxAccessArrayRange(arr, arr_offset-1, arr_offset, &stride_blend_arr, (void **)&pBlendArr, VX_READ_ONLY));
	arr_numitems = *((vx_uint32 *)pBlendArr);
	ERROR_CHECK_STATUS(vxCommitArrayRange(arr, arr_offset - 1, arr_offset, pBlendArr));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	opencl_global_work[0] = ((arr_numitems << 8) + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);
	return VX_SUCCESS;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK half_scale_gaussian_opencl_codegen(
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
	// read the output configuration
	vx_size arr_capacity = 0;
	vx_uint32 output_width = 0, output_height = 0, input_width = 0, input_height = 0;
	vx_uint32 num_cameras = 0;
	vx_df_image input_format = VX_DF_IMAGE_VIRT;
	vx_df_image output_format = VX_DF_IMAGE_VIRT;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);			// input scalar - num cameras
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &num_cameras));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 2);				// input array
	ERROR_CHECK_OBJECT(arr);
	ERROR_CHECK_STATUS(vxQueryArray(arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &arr_capacity, sizeof(arr_capacity)));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 3);				// input image
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	image = (vx_image)avxGetNodeParamRef(node, 4);				// output image
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &output_width, sizeof(output_width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &output_height, sizeof(output_height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &output_format, sizeof(output_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));

	vx_uint32 ip_image_height_offs = (vx_uint32)(input_height / num_cameras);
	vx_uint32 op_image_height_offs = (vx_uint32)(output_height / num_cameras);

	vx_uint32 work_items = (vx_uint32)arr_capacity;
	strcpy(opencl_kernel_function_name, "half_scale_gaussian");
	opencl_work_dim = 1;
	opencl_local_work[0] = 256;
	opencl_global_work[0] = ((work_items << 8) + opencl_local_work[0] - 1) & ~(opencl_local_work[0] - 1);

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
		"__kernel __attribute__((reqd_work_group_size(%d, 1, 1)))\n" // opencl_local_work[0]
		"void %s(uint num_cameras,\n"
		"         uint arr_offs,\n"
		"        __global char * valid_pix_buf, uint valid_pix_buf_offset, uint valid_pix_num_items,\n"
		"        uint ip_width, uint ip_height, __global uchar * ip_buf, uint ip_stride, uint ip_offset,\n" // opencl_kernel_function_name
		"        uint op_width, uint op_height, __global uchar * op_buf, uint op_stride, uint op_offset)\n"
		"{\n"
		"  int gid = get_global_id(0);\n"
		"  int grp_id = gid >> 8;\n"
		"  int lid = get_local_id(0);\n"
		"  int ly = lid >> 4;\n"
		"  int lx = lid - (ly << 4);\n"
		"  int height1 = %d;\n"
		, (int)opencl_local_work[0], opencl_kernel_function_name, ip_image_height_offs);
	opencl_kernel_code = item;

	if (input_format == VX_DF_IMAGE_U8) {
		opencl_kernel_code +=
			"  __local uchar lbuf[5040];    // Each 16x16 would load 144x35 bytes into LDS, 2 row padding on top and bottom and 4 pixel padding on the left and right\n";
	}
	else if (input_format == VX_DF_IMAGE_S16){
		opencl_kernel_code +=
			"  __local uchar lbuf[4760<<1];    // Each 16x16 would load 136x35*2 bytes into LDS, 2 row padding on top and bottom and 4 pixel padding on the left and right\n";
	}
	else if (input_format == VX_DF_IMAGE_RGBX) {
		opencl_kernel_code +=
			"  __local uchar lbuf[4760 << 2];    // Each 16x16 would load 136x35*4 bytes into LDS, 2 row padding on top and bottom and 4 pixel padding on the left and right\n";
	}
	opencl_kernel_code +=
		"  if(grp_id < valid_pix_num_items)\n"
		"  {\n"
		"    valid_pix_buf += valid_pix_buf_offset + ((grp_id + arr_offs) << 3) ;\n"			// each entry is 8 bytes
		"    uint2 wgInfo = *(__global uint2 * ) valid_pix_buf;\n"
		"    bool outputValid = (lx <= ((wgInfo.s1 & 0xFF) >> 2)) && (ly <= ((wgInfo.s1 >> 8) & 0xFF)) ? true : false;\n"
		"    int camId = wgInfo.s0 & 0x1F;\n"
		"    int gx = (wgInfo.s0 >> 5) & 0x3FFF;\n"
		"    int gy = (wgInfo.s0 >> 19) & 0x1FFF;\n"
		"    int border = (wgInfo.s1 >> 30)&0x3;\n";
		if (input_format == VX_DF_IMAGE_U8) {
		if (output_format == VX_DF_IMAGE_U8){
			snprintf(item, sizeof(item),
				"    op_buf += op_offset + ((gy + ly + (camId * %d)) * op_stride) + (lx << 2) + gx;\n"	// op_image_height_offs
				"    __global uchar * gbuf = ip_buf + ip_offset + (((gy << 1) + 1 + (camId * %d)) * ip_stride) + (gx << 1);\n"  // ip_image_height_offs
				, op_image_height_offs, ip_image_height_offs);
		}
		else if (output_format == VX_DF_IMAGE_S16){
			snprintf(item, sizeof(item),
				"    op_buf += op_offset + ((gy + ly + (camId * %d)) * op_stride) + (lx << 3) + (gx<<1);\n"	// op_image_height_offs
				"    __global uchar * gbuf = ip_buf + ip_offset + (((gy << 1) + 1 + (camId * %d)) * ip_stride) + (gx << 1);\n"  // ip_image_height_offs
				, op_image_height_offs, ip_image_height_offs);

		}
		opencl_kernel_code += item;
		opencl_kernel_code +=
			"    if (!border){ // loading 144x35 bytes into LDS using 16x16 workgroup\n"
			"	   int loffset = ly * 144 + (lx << 3);\n"
			"      int goffset = (ly - 2) * ip_stride + (lx << 3) - 8; \n"
			"      *(__local uint2 *)(lbuf + loffset) = vload2(0, (__global uint *)(gbuf + goffset));\n"
			"      goffset += 16 * ip_stride;\n"
			"      *(__local uint2 *)(lbuf + loffset + 16*144) = vload2(0, (__global uint *)(gbuf + goffset));\n"
			"      if (ly < 3) {\n"
			"        goffset += 16 * ip_stride;\n"
			"        *(__local uint2 *)(lbuf + loffset + 32*144) = vload2(0, (__global uint *)(gbuf + goffset));\n"
			"      }\n"
			"      if (lid < 35) {\n"
			"        *(__local uint4 *)(lbuf + (lid * 144) + 128) = vload4(0, (__global uint *)(gbuf + ((lid - 2) * (int)ip_stride) + 120));\n"
			"      }\n"
			"    }\n"
			"	else {\n"
			"	   int loffset = ly * 144 + (lx << 3);\n"
			"      int gybase = (gy<<1) + ly - 1;\n"
			"      int goffset = (gx<<1) + (lx << 3) - 8;\n"
			"      goffset = select(goffset, (int)(ip_width-8), goffset<0); goffset = select(goffset, 0, goffset>(ip_width-8));\n"
			"      __global uchar *gbuf = ip_buf + ip_offset + (camId*height1)*ip_stride;\n"
			"      *(__local uint2 *)(lbuf + loffset) = vload2(0, (__global uint *)(gbuf + goffset + ip_stride * max(0, gybase)));\n"
			"      *(__local uint2 *)(lbuf + loffset + 16*144) = vload2(0, (__global uint *)(gbuf + goffset + ip_stride * (gybase+16)));\n"
			"      if (ly < 3) {\n"
			"        *(__local uint2 *)(lbuf + loffset + 32*144) = vload2(0, (__global uint *)(gbuf + goffset + ip_stride * min(height1-1, gybase+32)));\n"
			"      }\n"
			"      if (lid < 35) {\n"
			"		 gybase = max(0, min(height1-1, ((gy<<1) + lid - 1)));\n"
			"		 goffset = (gx<<1) + 120; goffset = select(goffset, 0, goffset>(ip_width-8));\n"
			"        *(__local uint2 *)(lbuf + (lid * 144) + 128) = vload2(0, (__global uint *)(gbuf + goffset + ip_stride * gybase));\n"
			"		 goffset += 8; goffset = select(goffset, 0, goffset>(ip_width-8));\n"
			"        *(__local uint2 *)(lbuf + (lid * 144) + 136) = vload2(0, (__global uint *)(gbuf + goffset + ip_stride * gybase));\n"
			"      }\n"
			"    }\n"
			"    barrier(CLK_LOCAL_MEM_FENCE);\n"
			"    __local uchar * lbuf_ptr = lbuf + ly*144 + (lx<<3);\n"
			"    __local uchar * lbuf_ptr_src = lbuf_ptr+4;\n"
			"    float4 sum; float v;\n"
			"    // Horizontal filtering\n"
			"    uint4 L0= vload4(0, (__local uint *) lbuf_ptr_src);\n"
			"    v = amd_unpack3(L0.s0);                                             sum.s0 = v;\n"
			"    v = amd_unpack0(L0.s1);                                             sum.s0 = mad(v, 4.0f, sum.s0);\n"
			"    v = amd_unpack1(L0.s1);              sum.s0 = mad(v, 6.0f, sum.s0); sum.s1 = v;\n"
			"    v = amd_unpack2(L0.s1);              sum.s0 = mad(v, 4.0f, sum.s0); sum.s1 = mad(v, 4.0f, sum.s1);\n"
			"    v = amd_unpack3(L0.s1); sum.s0 += v; sum.s1 = mad(v, 6.0f, sum.s1); sum.s2 = v;\n"
			"    v = amd_unpack0(L0.s2);              sum.s1 = mad(v, 4.0f, sum.s1); sum.s2 = mad(v, 4.0f, sum.s2);\n"
			"    v = amd_unpack1(L0.s2); sum.s1 += v; sum.s2 = mad(v, 6.0f, sum.s2); sum.s3 = v;\n"
			"    v = amd_unpack2(L0.s2);              sum.s2 = mad(v, 4.0f, sum.s2); sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"    v = amd_unpack3(L0.s2); sum.s2 += v; sum.s3 = mad(v, 6.0f, sum.s3);\n"
			"    v = amd_unpack0(L0.s3);              sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"    v = amd_unpack1(L0.s3); sum.s3 += v;\n"
			"    L0.s0 = (uint)sum.s0 + (((uint)sum.s1) << 16);\n"
			"    L0.s1 = (uint)sum.s2 + (((uint)sum.s3) << 16);\n"
			"    *(__local uint2 *) lbuf_ptr = L0.s01;\n"
			"    L0 = vload4(0, (__local uint *) &lbuf_ptr_src[2304]);\n"
			"    v = amd_unpack3(L0.s0);                                             sum.s0 = v;\n"
			"    v = amd_unpack0(L0.s1);                                             sum.s0 = mad(v, 4.0f, sum.s0);\n"
			"    v = amd_unpack1(L0.s1);              sum.s0 = mad(v, 6.0f, sum.s0); sum.s1 = v;\n"
			"    v = amd_unpack2(L0.s1);              sum.s0 = mad(v, 4.0f, sum.s0); sum.s1 = mad(v, 4.0f, sum.s1);\n"
			"    v = amd_unpack3(L0.s1); sum.s0 += v; sum.s1 = mad(v, 6.0f, sum.s1); sum.s2 = v;\n"
			"    v = amd_unpack0(L0.s2);              sum.s1 = mad(v, 4.0f, sum.s1); sum.s2 = mad(v, 4.0f, sum.s2);\n"
			"    v = amd_unpack1(L0.s2); sum.s1 += v; sum.s2 = mad(v, 6.0f, sum.s2); sum.s3 = v;\n"
			"    v = amd_unpack2(L0.s2);              sum.s2 = mad(v, 4.0f, sum.s2); sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"    v = amd_unpack3(L0.s2); sum.s2 += v; sum.s3 = mad(v, 6.0f, sum.s3);\n"
			"    v = amd_unpack0(L0.s3);              sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"    v = amd_unpack1(L0.s3); sum.s3 += v;\n"
			"    L0.s0 = (uint)sum.s0 + (((uint)sum.s1) << 16);\n"
			"    L0.s1 = (uint)sum.s2 + (((uint)sum.s3) << 16);\n"
			"    *(__local uint2 *) &lbuf_ptr[2304] = L0.s01;\n"
			"    if (ly < 3) {\n"
			"      L0 = vload4(0, (__local uint *) &lbuf_ptr_src[4608]);\n"
			"      v = amd_unpack3(L0.s0);                                             sum.s0 = v;\n"
			"      v = amd_unpack0(L0.s1);                                             sum.s0 = mad(v, 4.0f, sum.s0);\n"
			"      v = amd_unpack1(L0.s1);              sum.s0 = mad(v, 6.0f, sum.s0); sum.s1 = v;\n"
			"      v = amd_unpack2(L0.s1);              sum.s0 = mad(v, 4.0f, sum.s0); sum.s1 = mad(v, 4.0f, sum.s1);\n"
			"      v = amd_unpack3(L0.s1); sum.s0 += v; sum.s1 = mad(v, 6.0f, sum.s1); sum.s2 = v;\n"
			"      v = amd_unpack0(L0.s2);              sum.s1 = mad(v, 4.0f, sum.s1); sum.s2 = mad(v, 4.0f, sum.s2);\n"
			"      v = amd_unpack1(L0.s2); sum.s1 += v; sum.s2 = mad(v, 6.0f, sum.s2); sum.s3 = v;\n"
			"      v = amd_unpack2(L0.s2);              sum.s2 = mad(v, 4.0f, sum.s2); sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"      v = amd_unpack3(L0.s2); sum.s2 += v; sum.s3 = mad(v, 6.0f, sum.s3);\n"
			"      v = amd_unpack0(L0.s3);              sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"      v = amd_unpack1(L0.s3); sum.s3 += v;\n"
			"      L0.s0 = (uint)sum.s0 + (((uint)sum.s1) << 16);\n"
			"      L0.s1 = (uint)sum.s2 + (((uint)sum.s3) << 16);\n"
			"      *(__local uint2 *) &lbuf_ptr[4608] = L0.s01;\n"
			"    }\n"
			"    barrier(CLK_LOCAL_MEM_FENCE);\n"
			"    // Vertical filtering\n"
			"    lbuf_ptr += ly * 144;\n"
			"    L0.s01 = vload2(0, (__local uint *) lbuf_ptr);\n"
			"    sum.s0 = (float)(L0.s0 & 0xffff); sum.s1 = (float)(L0.s0 >> 16); sum.s2 = (float)(L0.s1 & 0xffff); sum.s3 = (float)(L0.s1 >> 16);\n"
			"    L0.s01 = vload2(0, (__local uint *)&lbuf_ptr[144]);\n"
			"    sum.s0 = mad((float)(L0.s0 & 0xffff), 4.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 4.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 4.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 4.0f, sum.s3);\n"
			"    L0.s01 = vload2(0, (__local uint *)&lbuf_ptr[288]);\n"
			"    sum.s0 = mad((float)(L0.s0 & 0xffff), 6.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 6.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 6.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 6.0f, sum.s3);\n"
			"    L0.s01 = vload2(0, (__local uint *)&lbuf_ptr[432]);\n"
			"    sum.s0 = mad((float)(L0.s0 & 0xffff), 4.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 4.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 4.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 4.0f, sum.s3);\n"
			"    L0.s01 = vload2(0, (__local uint *)&lbuf_ptr[576]);\n"
			"    sum.s0 += (float)(L0.s0 & 0xffff); sum.s1 += (float)(L0.s0 >> 16); sum.s2 += (float)(L0.s1 & 0xffff); sum.s3 += (float)(L0.s1 >> 16);\n";
		if (output_format == VX_DF_IMAGE_U8){
			opencl_kernel_code +=
			"    sum = sum * (float4)0.00390625f;\n"  // todo:: do
			"    if (outputValid) {\n"
			"      *(__global uint *) op_buf = amd_pack(sum);\n"
			"    }\n"
			"  }\n"
			"}\n";
			}
			else
			{
			opencl_kernel_code +=
			"    sum = sum * (float4)0.5f;\n"  // multiply by 128 to increase precision while storing
			"    if (outputValid) {\n"
			"      *(__global short4 *) op_buf = convert_short4_sat_rte(sum);\n"
			"    }\n"
			"  }\n"
			"}\n";
			}
	}
	else if (input_format == VX_DF_IMAGE_S16) {
			snprintf(item, sizeof(item),
				"    op_buf += op_offset + ((gy + ly + (camId * %d)) * op_stride) + (lx << 3) + (gx<<1);\n"	// op_image_height_offs
				"    __global uchar * gbuf = ip_buf + ip_offset + (((gy << 1) + 1 + (camId * %d)) * ip_stride) + (gx << 2);\n"  // ip_image_height_offs
				"    int lstride = 136 << 1;\n"
				, op_image_height_offs, ip_image_height_offs);
		opencl_kernel_code += item;
		opencl_kernel_code +=
			"    { // loading 136x35 words into LDS using 16x16 workgroup\n"
			"      int loffset = ly * lstride + (lx << 4);\n"
			"      int goffset = (ly - 2) * ip_stride + (lx << 4) - 8;\n"
			"      *(__local uint4 *)(lbuf + loffset) = vload4(0, (__global uint *)(gbuf + goffset));\n"
			"      loffset += (lstride<<4);  goffset += (ip_stride<<4);\n"
			"      *(__local uint4 *)(lbuf + loffset) = vload4(0, (__global uint *)(gbuf + goffset));\n"
			"      if (ly < 3) {\n"
			"        loffset += (lstride<<4);  goffset += (ip_stride<<4);\n"
			"        *(__local uint4 *)(lbuf + loffset) = vload4(0, (__global uint *)(gbuf + goffset));\n"
			"      }\n"
			"      if (lid < 35) {\n"
			"        *(__local uint4 *)(lbuf + (lid * lstride) + 256) = vload4(0, (__global uint *)(gbuf + ((lid - 2) * (int)ip_stride) + 248));\n"
			"      }\n"
			"      barrier(CLK_LOCAL_MEM_FENCE);\n"
			"    }\n"
			"    __local uchar * lbuf_ptr = lbuf + ly * lstride + (lx << 4);\n"
			"    float4 sum; float v;\n"
			"    // Horizontal filtering\n"
			"    uint8 L0 = vload8(0, (__local uint *) lbuf_ptr); \n"
			"    v = convert_float(L0.S1 >> 16);											   sum.s0 = v;\n"
			"    v = convert_float(L0.S2 & 0xFFFF);                                            sum.s0 = mad(v, 4.0f, sum.s0);\n"
			"    v = convert_float(L0.S2 >> 16);				sum.s0 = mad(v, 6.0f, sum.s0); sum.s1 = v;\n"
			"    v = convert_float(L0.S3 & 0xFFFF);				sum.s0 = mad(v, 4.0f, sum.s0); sum.s1 = mad(v, 4.0f, sum.s1);\n"
			"    v = convert_float(L0.S3 >> 16); sum.s0 += v;	sum.s1 = mad(v, 6.0f, sum.s1); sum.s2 = v;\n"
			"    v = convert_float(L0.S4 & 0xFFFF);             sum.s1 = mad(v, 4.0f, sum.s1); sum.s2 = mad(v, 4.0f, sum.s2);\n"
			"    v = convert_float(L0.S4 >> 16); sum.s1 += v;	sum.s2 = mad(v, 6.0f, sum.s2); sum.s3 = v;\n"
			"    v = convert_float(L0.S5 & 0xFFFF);             sum.s2 = mad(v, 4.0f, sum.s2); sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"    v = convert_float(L0.S5 >> 16); sum.s2 += v;	sum.s3 = mad(v, 6.0f, sum.s3);\n"
			"    v = convert_float(L0.S6 & 0xFFFF);             sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"    v = convert_float(L0.S6 >> 16); sum.s3 += v;\n"
			"    //L0.s0 = (uint)sum.s0 + (((uint)sum.s1) << 16);\n"
			"    //L0.s1 = (uint)sum.s2 + (((uint)sum.s3) << 16);\n"
			"    *(__local uint4 *) lbuf_ptr = (uint4)((uint)sum.s0, (uint)sum.s1, (uint)sum.s2, (uint)sum.s3);\n"
			"    L0 = vload8(0, (__local uint *) &lbuf_ptr[4352]);\n"
			"    v = convert_float(L0.s1 >> 16);                                             sum.s0 = v;\n"
			"    v = convert_float(L0.s2 & 0xFFFF);                                             sum.s0 = mad(v, 4.0f, sum.s0);\n"
			"    v = convert_float(L0.s2 >> 16);              sum.s0 = mad(v, 6.0f, sum.s0); sum.s1 = v;\n"
			"    v = convert_float(L0.s3 & 0xFFFF);              sum.s0 = mad(v, 4.0f, sum.s0); sum.s1 = mad(v, 4.0f, sum.s1);\n"
			"    v = convert_float(L0.s3 >> 16); sum.s0 += v; sum.s1 = mad(v, 6.0f, sum.s1); sum.s2 = v;\n"
			"    v = convert_float(L0.s4 & 0xFFFF);              sum.s1 = mad(v, 4.0f, sum.s1); sum.s2 = mad(v, 4.0f, sum.s2);\n"
			"    v = convert_float(L0.s4 >> 16); sum.s1 += v; sum.s2 = mad(v, 6.0f, sum.s2); sum.s3 = v;\n"
			"    v = convert_float(L0.s5 & 0xFFFF);              sum.s2 = mad(v, 4.0f, sum.s2); sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"    v = convert_float(L0.s5 >> 16); sum.s2 += v; sum.s3 = mad(v, 6.0f, sum.s3);\n"
			"    v = convert_float(L0.s6 & 0xFFFF);              sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"    v = convert_float(L0.s6 >> 16); sum.s3 += v;\n"
			"    //L0.s0 = (uint)sum.s0 + (((uint)sum.s1) << 16);\n"
			"    //L0.s1 = (uint)sum.s2 + (((uint)sum.s3) << 16);\n"
			"    *(__local uint4 *) &lbuf_ptr[4352] = (uint4)((uint)sum.s0, (uint)sum.s1, (uint)sum.s2, (uint)sum.s3);\n"
			"    if (ly < 3) {\n"
			"      L0 = vload8(0, (__local uint *) &lbuf_ptr[8704]);\n"
			"      v = convert_float(L0.s1 >> 16);                                             sum.s0 = v;\n"
			"      v = convert_float(L0.s2 & 0xFFFF);                                             sum.s0 = mad(v, 4.0f, sum.s0);\n"
			"      v = convert_float(L0.s2 >> 16);              sum.s0 = mad(v, 6.0f, sum.s0); sum.s1 = v;\n"
			"      v = convert_float(L0.s3 & 0xFFFF);              sum.s0 = mad(v, 4.0f, sum.s0); sum.s1 = mad(v, 4.0f, sum.s1);\n"
			"      v = convert_float(L0.s3 >> 16); sum.s0 += v; sum.s1 = mad(v, 6.0f, sum.s1); sum.s2 = v;\n"
			"      v = convert_float(L0.s4 & 0xFFFF);              sum.s1 = mad(v, 4.0f, sum.s1); sum.s2 = mad(v, 4.0f, sum.s2);\n"
			"      v = convert_float(L0.s4 >> 16); sum.s1 += v; sum.s2 = mad(v, 6.0f, sum.s2); sum.s3 = v;\n"
			"      v = convert_float(L0.s5 & 0xFFFF);              sum.s2 = mad(v, 4.0f, sum.s2); sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"      v = convert_float(L0.s5 >> 16); sum.s2 += v; sum.s3 = mad(v, 6.0f, sum.s3);\n"
			"      v = convert_float(L0.s6 & 0xFFFF);              sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"      v = convert_float(L0.s6 >> 16); sum.s3 += v;\n"
			"      //L0.s0 = (uint)sum.s0 + (((uint)sum.s1) << 16);\n"
			"      //L0.s1 = (uint)sum.s2 + (((uint)sum.s3) << 16);\n"
			"      *(__local uint4 *) &lbuf_ptr[8704] = (uint4)((uint)sum.s0, (uint)sum.s1, (uint)sum.s2, (uint)sum.s3);\n"
			"    }\n"
			"    barrier(CLK_LOCAL_MEM_FENCE);\n"
			"    // Vertical filtering\n"
			"    lbuf_ptr += ly * lstride;\n"
			"    L0.lo = vload4(0, (__local uint *) lbuf_ptr);\n"
			"    sum = (float4)((float)L0.s0, (float)L0.s1, (float)L0.s2, (float)L0.s3);\n"
			"    lbuf_ptr += lstride;\n"
			"    L0.lo = vload4(0, (__local uint *)lbuf_ptr);\n"
			"    sum.s0 = mad((float)L0.s0, 4.0f, sum.s0); sum.s1 = mad((float)L0.s1, 4.0f, sum.s1); sum.s2 = mad((float)L0.s2, 4.0f, sum.s2); sum.s3 = mad((float)L0.s3, 4.0f, sum.s3);\n"
			"    lbuf_ptr += lstride;\n"
			"    L0.lo = vload4(0, (__local uint *)lbuf_ptr);\n"
			"    sum.s0 = mad((float)L0.s0, 6.0f, sum.s0); sum.s1 = mad((float)L0.s1, 6.0f, sum.s1); sum.s2 = mad((float)L0.s2, 6.0f, sum.s2); sum.s3 = mad((float)L0.s3, 6.0f, sum.s3);\n"
			"    lbuf_ptr += lstride;\n"
			"    L0.lo = vload4(0, (__local uint *)lbuf_ptr);\n"
			"    sum.s0 = mad((float)L0.s0, 4.0f, sum.s0); sum.s1 = mad((float)L0.s1, 4.0f, sum.s1); sum.s2 = mad((float)L0.s2, 4.0f, sum.s2); sum.s3 = mad((float)L0.s3, 4.0f, sum.s3);\n"
			"    lbuf_ptr += lstride;\n"
			"    L0.lo = vload4(0, (__local uint *)lbuf_ptr);\n"
			"    sum.s0 += (float)L0.s0; sum.s1 += (float)L0.s1; sum.s2 += (float)L0.s2; sum.s3 += (float)L0.s3;\n"
			"    sum = sum * (float4)0.00390625f;\n"
			"    if (outputValid) {\n"
			"      *(__global short4 *) op_buf = convert_short4_sat_rte(sum);\n"
			"    }\n"
			"  }\n"
			"}\n";
	}
	else if (input_format == VX_DF_IMAGE_RGBX) {
		snprintf(item, sizeof(item),
			"    op_buf += op_offset + ((gy + ly + (camId * %d)) * op_stride) + (lx << 4) + (gx << 2);\n"	// op_image_height_offs
			"    __global uchar * gbuf = ip_buf + ip_offset + (((gy << 1) + 1 + (camId * %d)) * ip_stride) + (gx << 3);\n"  // ip_image_height_offs
			"    int lstride = 136 << 2;\n"
			, op_image_height_offs, ip_image_height_offs);
		opencl_kernel_code += item;
		opencl_kernel_code +=
			"    if (!border)\n"
			"    { // loading 136x35x4 bytes into LDS using 16x16 workgroup\n"
			"      int loffset = ly * lstride + (lx << 5);\n"
			"      int goffset = (ly - 2) * ip_stride + (lx << 5) - 16;\n"
			"      *(__local uint8 *)(lbuf + loffset) = vload8(0, (__global uint *)(gbuf + goffset));\n"
			"      loffset += (lstride << 4);  goffset += (ip_stride << 4);\n"
			"      *(__local uint8 *)(lbuf + loffset) = vload8(0, (__global uint *)(gbuf + goffset));\n"
			"      if (ly < 3) {\n"
			"        loffset += (lstride << 4);  goffset += (ip_stride << 4);\n"
			"        *(__local uint8 *)(lbuf + loffset) = vload8(0, (__global uint *)(gbuf + goffset));\n"
			"      }\n"
			"      if (lid < 35) {\n"
			"        *(__local uint8 *)(lbuf + (lid * lstride) + (128 << 2)) = vload8(0, (__global uint *)(gbuf + ((lid - 2) * (int)ip_stride) + (128 << 2) - 16));\n"
			"      }\n"
			"    }else\n"
			"    {\n"
			"      int2 goffset; \n"
			"      gbuf = ip_buf + ip_offset + (camId*height1)*ip_stride;\n"
			"      int loffset = ly * lstride + (lx << 5);\n"
			"      int gybase = (gy<<1) + ly - 1;\n"
			"      goffset.s0 = (gx<<3) + (lx << 5) - 16; goffset.s1 = goffset.s0 + 16 ; \n"
			"      goffset.s0 = select(goffset.s0, (int)((ip_width-4)<<2), goffset.s0<0);\n"
			"      goffset.s1 = select(goffset.s1, 0, goffset.s1>((ip_width-4)<<2));\n"
			"     *(__local uint4 *)(lbuf + loffset)       = vload4(0, (__global uint *)(gbuf + goffset.s0 + ip_stride * max(0, gybase)));\n"
			"     *(__local uint4 *)(lbuf + loffset + 16)  = vload4(0, (__global uint *)(gbuf + goffset.s1 + ip_stride * max(0, gybase)));\n"
			"      loffset += (lstride << 4);\n"
			"      *(__local uint4 *)(lbuf + loffset)    = vload4(0, (__global uint *)(gbuf + goffset.s0 + ip_stride * (gybase+16)));\n"
			"      *(__local uint4 *)(lbuf + loffset+16) = vload4(0, (__global uint *)(gbuf + goffset.s1 + ip_stride * (gybase+16)));\n"
			"      if (ly < 3) {\n"
			"        loffset += (lstride << 4);\n"
			"        gybase = min(height1-1, gybase+32);\n"
			"        *(__local uint4 *)(lbuf + loffset)    = vload4(0, (__global uint *)(gbuf + goffset.s0 + ip_stride * gybase));\n"
			"        *(__local uint4 *)(lbuf + loffset+16) = vload4(0, (__global uint *)(gbuf + goffset.s1 + ip_stride * gybase));\n"
			"      }\n"
			"      if (lid < 35) {\n"
			"        gybase = max(0, min(height1-1, ((gy<<1) + lid - 1)));\n"
			"        goffset.s0 = (gx<<3) + (124<<2); goffset.s1 = goffset.s0 + 16;\n"
			"        goffset.s1 = select(goffset.s1, 0, goffset.s1>((ip_width-4)<<2));\n"
			"        *(__local uint4 *)(lbuf + (lid * lstride) + (128 << 2)) = vload4(0, (__global uint *)(gbuf + goffset.s0 + ip_stride * gybase));\n"
			"        *(__local uint4 *)(lbuf + (lid * lstride) + (132 << 2)) = vload4(0, (__global uint *)(gbuf + goffset.s1 + ip_stride * gybase));\n"
			"      }\n"
			"    }\n"
			"    barrier(CLK_LOCAL_MEM_FENCE);\n"
			"    // Horizontal filtering\n"
			"    __local uchar * lbuf_ptr = lbuf + (ly * lstride) + (lx << 5);\n"
			"    float4 sum0, sum1, sum2, sum3, v;\n"
			"    uint4 L0 = vload4(0, (__local uint *) lbuf_ptr);\n"
			"    uint4 L1 = vload4(0, (__local uint *) &lbuf_ptr[16]);\n"
			"    uint4 L2 = vload4(0, (__local uint *) &lbuf_ptr[32]);\n"
			"    v = amd_unpack(L0.s3);                                                 sum0 = v;\n"
			"    v = amd_unpack(L1.s0);                                                 sum0 = mad(v, (float4)(4.0f), sum0);\n"
			"    v = amd_unpack(L1.s1);            sum0 = mad(v, (float4)(6.0f), sum0); sum1 = v;\n"
			"    v = amd_unpack(L1.s2);            sum0 = mad(v, (float4)(4.0f), sum0); sum1 = mad(v, (float4)(4.0f), sum1);\n"
			"    v = amd_unpack(L1.s3); sum0 += v; sum1 = mad(v, (float4)(6.0f), sum1); sum2 = v;\n"
			"    v = amd_unpack(L2.s0);            sum1 = mad(v, (float4)(4.0f), sum1); sum2 = mad(v, (float4)(4.0f), sum2);\n"
			"    v = amd_unpack(L2.s1); sum1 += v; sum2 = mad(v, (float4)(6.0f), sum2); sum3 = v;\n"
			"    L0.s0 = (uint)sum0.s0 + (((uint)sum0.s1) << 16); L0.s1 = (uint)sum0.s2 + (((uint)sum0.s3) << 16);\n"
			"    L0.s2 = (uint)sum1.s0 + (((uint)sum1.s1) << 16); L0.s3 = (uint)sum1.s2 + (((uint)sum1.s3) << 16);\n"
			"    *(__local uint4 *) lbuf_ptr = L0;\n"
			"    L1 = vload4(0, (__local uint *) &lbuf_ptr[48]);\n"
			"    v = amd_unpack(L2.s2);            sum2 = mad(v, (float4)(4.0f), sum2); sum3 = mad(v, (float4)(4.0f), sum3);\n"
			"    v = amd_unpack(L2.s3); sum2 += v; sum3 = mad(v, (float4)(6.0f), sum3);\n"
			"    v = amd_unpack(L1.s0);            sum3 = mad(v, (float4)(4.0f), sum3);\n"
			"    v = amd_unpack(L1.s1); sum3 += v;\n"
			"    L1.s0 = (uint)sum2.s0 + (((uint)sum2.s1) << 16); L1.s1 = (uint)sum2.s2 + (((uint)sum2.s3) << 16);\n"
			"    L1.s2 = (uint)sum3.s0 + (((uint)sum3.s1) << 16); L1.s3 = (uint)sum3.s2 + (((uint)sum3.s3) << 16);\n"
			"    *(__local uint4 *) &lbuf_ptr[16] = L1;\n"
			"    L0 = vload4(0, (__local uint *) &lbuf_ptr[lstride << 4]);\n"
			"    L1 = vload4(0, (__local uint *) &lbuf_ptr[(lstride << 4) + 16]);\n"
			"    L2 = vload4(0, (__local uint *) &lbuf_ptr[(lstride << 4) + 32]);\n"
			"    v = amd_unpack(L0.s3);                                                 sum0 = v;\n"
			"    v = amd_unpack(L1.s0);                                                 sum0 = mad(v, (float4)(4.0f), sum0);\n"
			"    v = amd_unpack(L1.s1);            sum0 = mad(v, (float4)(6.0f), sum0); sum1 = v;\n"
			"    v = amd_unpack(L1.s2);            sum0 = mad(v, (float4)(4.0f), sum0); sum1 = mad(v, (float4)(4.0f), sum1);\n"
			"    v = amd_unpack(L1.s3); sum0 += v; sum1 = mad(v, (float4)(6.0f), sum1); sum2 = v;\n"
			"    v = amd_unpack(L2.s0);            sum1 = mad(v, (float4)(4.0f), sum1); sum2 = mad(v, (float4)(4.0f), sum2);\n"
			"    v = amd_unpack(L2.s1); sum1 += v; sum2 = mad(v, (float4)(6.0f), sum2); sum3 = v;\n"
			"    L0.s0 = (uint)sum0.s0 + (((uint)sum0.s1) << 16); L0.s1 = (uint)sum0.s2 + (((uint)sum0.s3) << 16);\n"
			"    L0.s2 = (uint)sum1.s0 + (((uint)sum1.s1) << 16); L0.s3 = (uint)sum1.s2 + (((uint)sum1.s3) << 16);\n"
			"    *(__local uint4 *) &lbuf_ptr[lstride << 4] = L0;\n"
			"    L1 = vload4(0, (__local uint *) &lbuf_ptr[16 * lstride + 48]);\n"
			"    v = amd_unpack(L2.s2);            sum2 = mad(v, (float4)(4.0f), sum2); sum3 = mad(v, (float4)(4.0f), sum3);\n"
			"    v = amd_unpack(L2.s3); sum2 += v; sum3 = mad(v, (float4)(6.0f), sum3);\n"
			"    v = amd_unpack(L1.s0);            sum3 = mad(v, (float4)(4.0f), sum3);\n"
			"    v = amd_unpack(L1.s1); sum3 += v;\n"
			"    L1.s0 = (uint)sum2.s0 + (((uint)sum2.s1) << 16); L1.s1 = (uint)sum2.s2 + (((uint)sum2.s3) << 16);\n"
			"    L1.s2 = (uint)sum3.s0 + (((uint)sum3.s1) << 16); L1.s3 = (uint)sum3.s2 + (((uint)sum3.s3) << 16);\n"
			"    *(__local uint4 *) &lbuf_ptr[(lstride << 4) + 16] = L1;\n"
			"    if (ly < 3) {\n"
			"      L0 = vload4(0, (__local uint *) &lbuf_ptr[lstride << 5]);\n"
			"      L1 = vload4(0, (__local uint *) &lbuf_ptr[(lstride << 5) + 16]);\n"
			"      L2 = vload4(0, (__local uint *) &lbuf_ptr[(lstride << 5) + 32]);\n"
			"      v = amd_unpack(L0.s3);                                                 sum0 = v;\n"
			"      v = amd_unpack(L1.s0);                                                 sum0 = mad(v, (float4)(4.0f), sum0);\n"
			"      v = amd_unpack(L1.s1);            sum0 = mad(v, (float4)(6.0f), sum0); sum1 = v;\n"
			"      v = amd_unpack(L1.s2);            sum0 = mad(v, (float4)(4.0f), sum0); sum1 = mad(v, (float4)(4.0f), sum1);\n"
			"      v = amd_unpack(L1.s3); sum0 += v; sum1 = mad(v, (float4)(6.0f), sum1); sum2 = v;\n"
			"      v = amd_unpack(L2.s0);            sum1 = mad(v, (float4)(4.0f), sum1); sum2 = mad(v, (float4)(4.0f), sum2);\n"
			"      v = amd_unpack(L2.s1); sum1 += v; sum2 = mad(v, (float4)(6.0f), sum2); sum3 = v;\n"
			"      L0.s0 = (uint)sum0.s0 + (((uint)sum0.s1) << 16); L0.s1 = (uint)sum0.s2 + (((uint)sum0.s3) << 16);\n"
			"      L0.s2 = (uint)sum1.s0 + (((uint)sum1.s1) << 16); L0.s3 = (uint)sum1.s2 + (((uint)sum1.s3) << 16);\n"
			"      *(__local uint4 *) &lbuf_ptr[lstride << 5] = L0;\n"
			"      L1 = vload4(0, (__local uint *) &lbuf_ptr[2 * 136 * 16 * 4 + 48]);\n"
			"      v = amd_unpack(L2.s2);            sum2 = mad(v, (float4)(4.0f), sum2); sum3 = mad(v, (float4)(4.0f), sum3);\n"
			"      v = amd_unpack(L2.s3); sum2 += v; sum3 = mad(v, (float4)(6.0f), sum3);\n"
			"      v = amd_unpack(L1.s0);            sum3 = mad(v, (float4)(4.0f), sum3);\n"
			"      v = amd_unpack(L1.s1); sum3 += v;\n"
			"      L1.s0 = (uint)sum2.s0 + (((uint)sum2.s1) << 16); L1.s1 = (uint)sum2.s2 + (((uint)sum2.s3) << 16);\n"
			"      L1.s2 = (uint)sum3.s0 + (((uint)sum3.s1) << 16); L1.s3 = (uint)sum3.s2 + (((uint)sum3.s3) << 16);\n"
			"      *(__local uint4 *) &lbuf_ptr[(lstride << 5) + 16] = L1;\n"
			"    }\n"
			"    barrier(CLK_LOCAL_MEM_FENCE);\n"
			"    // Vertical filtering\n"
			"    lbuf_ptr += (ly * lstride);\n"
			"    L0 = vload4(0, (__local uint *) lbuf_ptr); L1 = vload4(0, (__local uint *) &lbuf_ptr[16]);\n"
			"    v.s0 = (float)(L0.s0 & 0xFFFF); v.s1 = (float)(L0.s0 >> 16); v.s2 = (float)(L0.s1 & 0xFFFF); v.s3 = (float)(L0.s1 >> 16); sum0 = v;\n"
			"    v.s0 = (float)(L0.s2 & 0xFFFF); v.s1 = (float)(L0.s2 >> 16); v.s2 = (float)(L0.s3 & 0xFFFF); v.s3 = (float)(L0.s3 >> 16); sum1 = v;\n"
			"    v.s0 = (float)(L1.s0 & 0xFFFF); v.s1 = (float)(L1.s0 >> 16); v.s2 = (float)(L1.s1 & 0xFFFF); v.s3 = (float)(L1.s1 >> 16); sum2 = v;\n"
			"    v.s0 = (float)(L1.s2 & 0xFFFF); v.s1 = (float)(L1.s2 >> 16); v.s2 = (float)(L1.s3 & 0xFFFF); v.s3 = (float)(L1.s3 >> 16); sum3 = v;\n"
			"    lbuf_ptr += lstride;\n"
			"    L0 = vload4(0, (__local uint *) lbuf_ptr); L1 = vload4(0, (__local uint *) &lbuf_ptr[16]);\n"
			"    v.s0 = (float)(L0.s0 & 0xFFFF); v.s1 = (float)(L0.s0 >> 16); v.s2 = (float)(L0.s1 & 0xFFFF); v.s3 = (float)(L0.s1 >> 16); sum0 = mad(v, (float4)(4.0f), sum0);\n"
			"    v.s0 = (float)(L0.s2 & 0xFFFF); v.s1 = (float)(L0.s2 >> 16); v.s2 = (float)(L0.s3 & 0xFFFF); v.s3 = (float)(L0.s3 >> 16); sum1 = mad(v, (float4)(4.0f), sum1);\n"
			"    v.s0 = (float)(L1.s0 & 0xFFFF); v.s1 = (float)(L1.s0 >> 16); v.s2 = (float)(L1.s1 & 0xFFFF); v.s3 = (float)(L1.s1 >> 16); sum2 = mad(v, (float4)(4.0f), sum2);\n"
			"    v.s0 = (float)(L1.s2 & 0xFFFF); v.s1 = (float)(L1.s2 >> 16); v.s2 = (float)(L1.s3 & 0xFFFF); v.s3 = (float)(L1.s3 >> 16); sum3 = mad(v, (float4)(4.0f), sum3);\n"
			"    lbuf_ptr += lstride;\n"
			"    L0 = vload4(0, (__local uint *) lbuf_ptr); L1 = vload4(0, (__local uint *) &lbuf_ptr[16]);\n"
			"    v.s0 = (float)(L0.s0 & 0xFFFF); v.s1 = (float)(L0.s0 >> 16); v.s2 = (float)(L0.s1 & 0xFFFF); v.s3 = (float)(L0.s1 >> 16); sum0 = mad(v, (float4)(6.0f), sum0);\n"
			"    v.s0 = (float)(L0.s2 & 0xFFFF); v.s1 = (float)(L0.s2 >> 16); v.s2 = (float)(L0.s3 & 0xFFFF); v.s3 = (float)(L0.s3 >> 16); sum1 = mad(v, (float4)(6.0f), sum1);\n"
			"    v.s0 = (float)(L1.s0 & 0xFFFF); v.s1 = (float)(L1.s0 >> 16); v.s2 = (float)(L1.s1 & 0xFFFF); v.s3 = (float)(L1.s1 >> 16); sum2 = mad(v, (float4)(6.0f), sum2);\n"
			"    v.s0 = (float)(L1.s2 & 0xFFFF); v.s1 = (float)(L1.s2 >> 16); v.s2 = (float)(L1.s3 & 0xFFFF); v.s3 = (float)(L1.s3 >> 16); sum3 = mad(v, (float4)(6.0f), sum3);\n"
			"    lbuf_ptr += lstride;\n"
			"    L0 = vload4(0, (__local uint *) lbuf_ptr); L1 = vload4(0, (__local uint *) &lbuf_ptr[16]);\n"
			"    v.s0 = (float)(L0.s0 & 0xFFFF); v.s1 = (float)(L0.s0 >> 16); v.s2 = (float)(L0.s1 & 0xFFFF); v.s3 = (float)(L0.s1 >> 16); sum0 = mad(v, (float4)(4.0f), sum0);\n"
			"    v.s0 = (float)(L0.s2 & 0xFFFF); v.s1 = (float)(L0.s2 >> 16); v.s2 = (float)(L0.s3 & 0xFFFF); v.s3 = (float)(L0.s3 >> 16); sum1 = mad(v, (float4)(4.0f), sum1);\n"
			"    v.s0 = (float)(L1.s0 & 0xFFFF); v.s1 = (float)(L1.s0 >> 16); v.s2 = (float)(L1.s1 & 0xFFFF); v.s3 = (float)(L1.s1 >> 16); sum2 = mad(v, (float4)(4.0f), sum2);\n"
			"    v.s0 = (float)(L1.s2 & 0xFFFF); v.s1 = (float)(L1.s2 >> 16); v.s2 = (float)(L1.s3 & 0xFFFF); v.s3 = (float)(L1.s3 >> 16); sum3 = mad(v, (float4)(4.0f), sum3);\n"
			"    lbuf_ptr += lstride;\n"
			"    v.s0 = (float)(L0.s0 & 0xFFFF); v.s1 = (float)(L0.s0 >> 16); v.s2 = (float)(L0.s1 & 0xFFFF); v.s3 = (float)(L0.s1 >> 16); sum0 += v;\n"
			"    v.s0 = (float)(L0.s2 & 0xFFFF); v.s1 = (float)(L0.s2 >> 16); v.s2 = (float)(L0.s3 & 0xFFFF); v.s3 = (float)(L0.s3 >> 16); sum1 += v;\n"
			"    v.s0 = (float)(L1.s0 & 0xFFFF); v.s1 = (float)(L1.s0 >> 16); v.s2 = (float)(L1.s1 & 0xFFFF); v.s3 = (float)(L1.s1 >> 16); sum2 += v;\n"
			"    v.s0 = (float)(L1.s2 & 0xFFFF); v.s1 = (float)(L1.s2 >> 16); v.s2 = (float)(L1.s3 & 0xFFFF); v.s3 = (float)(L1.s3 >> 16); sum3 += v;\n"
			"    sum0 *= (float4)0.00390625f; sum1 *= (float4)0.00390625f; sum2 *= (float4)0.00390625f; sum3 *= (float4)0.00390625f;\n"
			"    L0.s0 = amd_pack(sum0); L0.s1 = amd_pack(sum1); L0.s2 = amd_pack(sum2); L0.s3 = amd_pack(sum3);\n"
			"    if (outputValid) {\n"
			"      *(__global uint4 *) op_buf = L0;\n"
			"    }\n"
			"  }\n"
			"}\n";
	}
	
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK half_scale_gaussian_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_SUPPORTED;
}

//! \brief The kernel publisher.
vx_status half_scale_gaussian_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.half_scale_gaussian",
		AMDOVX_KERNEL_STITCHING_HALF_SCALE_GAUSSIAN,
		half_scale_gaussian_kernel,
		5,
		half_scale_gaussian_input_validator,
		half_scale_gaussian_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = half_scale_gaussian_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = half_scale_gaussian_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = half_scale_gaussian_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));

	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));

	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

	return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK upscale_gaussian_subtract_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	if (index == 0)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		vx_uint32 num_cameras = 0;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)ref, &num_cameras));
		if (itemtype == VX_TYPE_UINT32){
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian_subtract num_cameras scalar type should be VX_TYPE_UINT32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
	}
	else if (index == 1)
	{ // object of SCALAR type
		vx_enum itemtype = VX_TYPE_INVALID;
		vx_uint32 num_cameras = 0;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		ERROR_CHECK_STATUS(vxReadScalarValue((vx_scalar)ref, &num_cameras));
		if (itemtype == VX_TYPE_UINT32){
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian_subtract arr_offs scalar type should be VX_TYPE_UINT32\n");
		}
		ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
	}
	else if (index == 2)
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
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian image %d should be an image of RGBX type\n", index);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 3)
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
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian image %d should be an image of RGB2 type\n", index);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 4)
	{ // array object for offsets
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchBlendValidEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: warp array element (StitchBlendValidEntry) size should be 32 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	else if (index == 5)
	{ // image object of U008 type
		if (ref){
			vx_df_image format = VX_DF_IMAGE_VIRT;
			ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
			if (format == VX_DF_IMAGE_U8) {
				status = VX_SUCCESS;
			}
			else {
				status = VX_ERROR_INVALID_TYPE;
				vxAddLogEntry((vx_reference)node, status, "ERROR: weight image should be an image of U008 type\n");
			}
			ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		}
		status = VX_SUCCESS;
	}

	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK upscale_gaussian_subtract_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 6)
	{ // image of format RGB4 of pyramid image
		vx_image image = (vx_image)avxGetNodeParamRef(node, 2);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 input_width = 0, input_height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		if (width != input_width)
			width = input_width;
		if (input_height != height)
		{ // pick default output height as the input map height
			height = input_height;
		}
		// set output image meta data
		vx_df_image format = VX_DF_IMAGE_RGB4_AMD;
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		status = VX_SUCCESS;
	}
	return status;
}
//! \brief The kernel target support callback.
static vx_status VX_CALLBACK upscale_gaussian_subtract_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK upscale_gaussian_subtract_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK upscale_gaussian_subtract_opencl_codegen(
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
	vx_size		wg_num;
	vx_uint32   numCam = 0;
	vx_uint32 width = 0, height = 0;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);			// input scalar - num_cam
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &numCam));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 6);				// output image
	vx_df_image out_format = VX_DF_IMAGE_VIRT;
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &out_format, sizeof(out_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	vx_array wg_offsets = (vx_array)avxGetNodeParamRef(node, 4);
	ERROR_CHECK_STATUS(vxQueryArray(wg_offsets, VX_ARRAY_ATTRIBUTE_CAPACITY, &wg_num, sizeof(wg_num)));
	ERROR_CHECK_STATUS(vxReleaseArray(&wg_offsets));
	vx_image weight_image = (vx_image)avxGetNodeParamRef(node, 5);
	vx_df_image wt_format = VX_DF_IMAGE_VIRT;
	vx_uint32 input_width = 0, input_height = 0;
	if (weight_image != NULL){
		ERROR_CHECK_STATUS(vxQueryImage(weight_image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage(weight_image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage(weight_image, VX_IMAGE_ATTRIBUTE_FORMAT, &wt_format, sizeof(wt_format)));
	}
	image = (vx_image)avxGetNodeParamRef(node, 3);				// input image
	vx_uint32 InHeight1;
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &InHeight1, sizeof(InHeight1)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	// set kernel configuration
	strcpy(opencl_kernel_function_name, "upscale_gaussian_subtract");
	opencl_work_dim = 2;
	opencl_local_work[0] = 16;
	opencl_local_work[1] = 4;
	opencl_global_work[0] = wg_num*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1] << 1;
	vx_uint32 height1 = height;
	if (numCam){
		height1 = (vx_uint32)(height / numCam);
		InHeight1 = (vx_uint32)(InHeight1 / numCam);
	}
	//printf("UGS: Height1: %d Inheight1: %d\n", height1, InHeight1);

	char item[8192];
	if (weight_image){
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"\n"
			"float3 amd_unpack_3(uint src)\n"
			"{\n"
			"	return (float3)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src));\n"
			"}\n"
			"\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint num_cam, uint arr_offs, \n"
			"	uint ip_width, uint ip_height, __global uchar * ip_buf, uint ip_stride, uint ip_offset, \n"
			" 	uint ip1_width, uint ip1_height, __global uchar * ip1_buf, uint ip1_stride, uint ip1_offset,\n"
			"	__global uchar * pG_buf, uint pG_offs, uint pG_num,\n"
			"	uint wt_width, uint wt_height, __global uchar * wt_buf, uint wt_stride, uint wt_offset,\n"
			"   uint op_width, uint op_height, __global uchar * op_buf, uint op_stride, uint op_offset)\n"
			"{\n"
			"	int grp_id = get_global_id(0)>>4, lx = get_local_id(0), ly = get_global_id(1);\n"
			"	pG_buf += (pG_offs + (arr_offs<<3));\n"
			"	if (grp_id < pG_num) {\n"
			"	int size_x = get_local_size(0) - 1; \n"
			"	uint2 offs = ((__global uint2 *)pG_buf)[grp_id];\n"
			"	uint camera_id = offs.x & 0x1f; int gx = (lx<<2) + ((offs.x >> 5) & 0x3FFF); int gy = (offs.x >> 19);\n"
			"	if (!get_group_id(1) | (get_group_id(1) && (gy+8 < %d))) {\n"
			"	gy += (ly<<1);\n"
			"   bool outputValid = (lx*4 <= (offs.y & 0xFF)) && (ly*2 <= ((offs.y >> 8)&0xFF));\n"
			"	int border = (offs.y >> 30) & 0x3;\n"
			"	int ybound = %d;\n"
			"	ip_buf += ip_offset + mad24(gy, (int)ip_stride, gx<<2);\n"
			"	op_buf  += op_offset + mad24(gy, (int)op_stride, gx*6);\n"
			"	ip_buf += (camera_id * ip_stride*%d);\n"
			"	ip1_buf += ip1_offset + (camera_id * ip1_stride*%d);\n"
			"	op_buf += (camera_id * op_stride*%d);\n"
			"	wt_buf += (camera_id * wt_stride*%d);\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height1, (InHeight1 - 1), height1, InHeight1, height1, height1);
		opencl_kernel_code = item;
		if (wt_format == VX_DF_IMAGE_U8){
			opencl_kernel_code +=
			"	wt_buf += wt_offset + mad24(gy, (int)wt_stride, gx);\n";
		}
		else
		{
			opencl_kernel_code +=
			"	wt_buf += wt_offset + mad24(gy, (int)wt_stride, gx<<1);\n";
		}
	}
	else
	{
		snprintf(item, sizeof(item),
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
			"\n"
			"float3 amd_unpack_3(uint src)\n"
			"{\n"
			"	return (float3)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src));\n"
			"}\n"
			"\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint num_cam, uint arr_offs, \n"
			"	uint ip_width, uint ip_height, __global uchar * ip_buf, uint ip_stride, uint ip_offset, \n"
			" 	uint ip1_width, uint ip1_height, __global uchar * ip1_buf, uint ip1_stride, uint ip1_offset,\n"
			"	 __global uchar * pG_buf, uint pG_offs, uint pG_num,\n"
			"   uint op_width, uint op_height, __global uchar * op_buf, uint op_stride, uint op_offset)\n"
			"{\n"
			"	int grp_id = get_global_id(0)>>4, lx = get_local_id(0), ly = get_global_id(1);\n"
			"	pG_buf += (pG_offs + (arr_offs<<3));\n"
			"	int size_x = get_local_size(0) - 1; \n"
			"	uint2 offs = ((__global uint2 *)pG_buf)[grp_id];\n"
			"	uint camera_id = offs.x & 0x1f; int gx = (lx<<2) + ((offs.x >> 5) & 0x3FFF); uint gy = (offs.x >> 19);\n"
			"	if (!get_group_id(1) | (get_group_id(1) && (gy+8 < %d))) {\n"
			"	gy += (ly<<1);\n"
			"   bool outputValid = (lx*4 <= (offs.y & 0xFF)) && (ly*2 <= ((offs.y >> 8)&0xFF));\n"
			"	int border = (offs.y >> 30)&0x3;\n"
			"	int ybound = %d;\n"
			"	ip_buf += ip_offset + mad24(gy, (int)ip_stride, gx<<2);\n"
			"	op_buf  += op_offset + mad24(gy, (int)op_stride, gx*6);\n"
			"	ip_buf += (camera_id * ip_stride*%d);\n"
			"	ip1_buf += ip1_offset + (camera_id * ip1_stride*%d);\n"
			"	op_buf += (camera_id * op_stride*%d);\n"
			, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, height1, (InHeight1 - 1), height1, InHeight1, height1);
		opencl_kernel_code = item;
	}

	opencl_kernel_code +=
		"	// load from src2 (for upscale) to LDS\n"
		"   ly = ly & 3;\n"
		"	__local uint lsSrc[36*6];\n"
		"	__local short lsVert[64*6*3];\n"
		"	int loffset = mad24(ly, (int)36, (lx<<1)); \n"
		"	int2 loffs2  = (int2)((loffset + 4*36), 4*ip1_stride);\n"
		"  if (!border){\n"
		"   ip1_buf += mad24(gy>>1, (int)ip1_stride, gx<<1);\n"
		"  	ip1_buf -= (ip1_stride + 8);\n"
		"  	if (!lx){\n"
		"  		*(__local uint2 *)(lsSrc+loffset) = vload2(0, (global uint *)ip1_buf);	\n"
		"  	}\n"
		"  	*(__local uint2 *)(lsSrc+loffset+2) = vload2(1, (global uint *)ip1_buf);\n"
		"  	if (lx == size_x){\n"
		"  		*(__local uint2 *)(lsSrc+loffset+4) = vload2(2, (global uint *)ip1_buf);	\n"
		"  	}\n"
		"  	if (ly < 2){\n"
		"  		if (!lx){\n"
		"  			*(__local uint2 *)(lsSrc+loffs2.x) = vload2(0, (global uint *)(ip1_buf+loffs2.y));	\n"
		"  		}\n"
		"  		*(__local uint2 *)(lsSrc+loffs2.x+2) =  vload2(1, (global uint *)(ip1_buf+loffs2.y));\n"
		"  		if (lx == size_x){\n"
		"  			*(__local uint2 *)(lsSrc+loffs2.x+4) =  vload2(2, (global uint *)(ip1_buf+loffs2.y));\n"
		"  		}\n"
		"  	}\n"
		"  }else\n"
		"  {\n"
		"    int gybase = (gy>>1) - 1;\n"
		"    int3 goffset = (int3)((gx<<1) - 8, (gx<<1), (gx<<1)+8);  \n"
		"    __global uchar *gbuf = ip1_buf + ip1_stride*max(0, min(ybound, gybase));\n"
		"  	 if (!lx){\n"
		"		goffset.s0 = select(goffset.s0, (int)((ip1_width-2)<<2), goffset.s0<0);\n"
		"  		*(__local uint2 *)(lsSrc+loffset) = vload2(0, (global uint *)(gbuf + goffset.s0));	\n"
		"  	 }\n"
		"    goffset.s1 = select(goffset.s1, 0, goffset.s1>((ip1_width-2)<<2));\n"
		"    *(__local uint2 *)(lsSrc+loffset+2)  = vload2(0, (global uint *)(gbuf + goffset.s1));\n"
		"    if (lx == size_x){\n"
		"      goffset.s2 = select(goffset.s2, 0, goffset.s2>((ip1_width-2)<<2));\n"
		"      *(__local uint2 *)(lsSrc+loffset+4)  = vload2(0, (__global uint *)(gbuf + goffset.s2));\n"
		"    }\n"
		"    if (ly < 2){\n"
		"      gbuf = ip1_buf + ip1_stride * min(ybound,(gybase+4));\n"
		"  	   if (!lx){\n"
		"  		 *(__local uint2 *)(lsSrc+loffs2.x) = vload2(0, (global uint *)(gbuf + goffset.s0));	\n"
		"  	   }\n"
		"      *(__local uint2 *)(lsSrc+loffs2.x+2)  = vload2(0, (global uint *)(gbuf + goffset.s1));\n"
		"      if (lx == size_x){\n"
		"         *(__local uint2 *)(lsSrc+loffs2.x+4)  = vload2(0, (__global uint *)(gbuf + goffset.s2));\n"
		"      }\n"
		"    }\n"
		"  }\n"
		"	barrier(CLK_LOCAL_MEM_FENCE);\n"
		"	uint lvoffset = ly * 64 + (lx << 2); \n"
		"	__local short *pVertFilt = &lsVert[lvoffset*3];\n"
		"	__local short *pVertFilt2 = pVertFilt + 256*3;	 //4*64\n"
		"	// do horizontal filter pass\n"
		"   {\n"
		"     uint4 pix0 = *(__local uint4 *)(lsSrc + loffset); uint pix1 = lsSrc[loffset+4];\n"
		"     float3 f0, f1, f2;\n"
		"     short8 filt_pix0; short4 filt_pix1;\n"
		"     f0 = amd_unpack_3(pix0.s1); f1 = amd_unpack_3(pix0.s2); f2 = amd_unpack_3(pix0.s3);\n"
		"     filt_pix0.s012 = convert_short3(mad(f1, (float3)6.0f, f0 + f2));\n"
		"     filt_pix0.s345 = convert_short3(4.0f * (f1 + f2));\n"
		"     f0 = amd_unpack_3(pix1);\n"
		"     f1 = mad(f2, (float3)6.0f, f1 + f0);\n"
		"     filt_pix0.s6 = (short)f1.s0; filt_pix0.s7 = (short)f1.s1; filt_pix1.s0 = (short)f1.s2;\n"
		"     filt_pix1.s123 = convert_short3(4.0f * (f2 + f0));\n"
		"     *(__local short8 *)&pVertFilt[0] = filt_pix0; *(__local short4 *)&pVertFilt[8] = filt_pix1;\n"
		"   }\n"
		"   if (ly < 2)\n"
		"   {\n"
		"     uint4 pix0 = *(__local uint4 *)(lsSrc + loffs2.x); uint pix1 = lsSrc[loffs2.x+4];\n"
		"     float3 f0, f1, f2;\n"
		"     short8 filt_pix0; short4 filt_pix1;\n"
		"     f0 = amd_unpack_3(pix0.s1); f1 = amd_unpack_3(pix0.s2); f2 = amd_unpack_3(pix0.s3);\n"
		"     filt_pix0.s012 = convert_short3(mad(f1, (float3)6.0f, f0 + f2));\n"
		"     filt_pix0.s345 = convert_short3(4.0f * (f1 + f2));\n"
		"     f0 = amd_unpack_3(pix1);\n"
		"     f1 = mad(f2, (float3)6.0f, f1 + f0);\n"
		"     filt_pix0.s6 = (short)f1.s0; filt_pix0.s7 = (short)f1.s1; filt_pix1.s0 = (short)f1.s2;\n"
		"     filt_pix1.s123 = convert_short3(4.0f * (f2 + f0));\n"
		"     *(__local short8 *)&pVertFilt2[0] = filt_pix0; *(__local short4 *)&pVertFilt2[8] = filt_pix1;\n"
		"   }\n"
		"	barrier(CLK_LOCAL_MEM_FENCE);\n";
	if (weight_image){
		opencl_kernel_code +=
			"	// do vertical filtering for 4\n"
			"   {\n"
			"     short8 tmp_8; short4 tmp_4;\n"
			"     float3 tmp;\n"
			"     float8 row0_8, row1_8; float4 row0_4, row1_4;\n"
			"     tmp_8 = *(__local short8 *)&pVertFilt[0]; tmp_4 = *(__local short4 *)&pVertFilt[8];\n"
			"     row0_8 = convert_float8(tmp_8); row0_4 = convert_float4(tmp_4);\n"
			"     tmp_8 = *(__local short8 *)&pVertFilt[3*64]; tmp_4 = *(__local short4 *)&pVertFilt[3*64 + 8];\n"
			"     row0_8 = mad(convert_float8(tmp_8), (float8)(6.0f), row0_8); row0_4 = mad(convert_float4(tmp_4), (float4)(6.0f), row0_4);\n"
			"     row1_8 = 4.0f * convert_float8(tmp_8); row1_4 = 4.0f * convert_float4(tmp_4);\n"
			"     tmp_8 = *(__local short8 *)&pVertFilt[3*128]; tmp_4 = *(__local short4 *)&pVertFilt[3*128 + 8];\n"
			"     row0_8 += convert_float8(tmp_8); row0_4 += convert_float4(tmp_4);\n"
			"     row1_8 = mad(convert_float8(tmp_8), (float8)(4.0f), row1_8); row1_4 = mad(convert_float4(tmp_4), (float4)(4.0f), row1_4);\n"
			"     row0_8 *= (float8)0.015625f; row0_4 *= (float4)0.015625f;"
			"     row1_8 *= (float8)0.015625f; row1_4 *= (float4)0.015625f;"
			"     uint4 px = *(__global uint4 *)ip_buf;"
			"     tmp = amd_unpack_3(px.s0); row0_8.s012 = tmp - row0_8.s012;\n"
			"     tmp = amd_unpack_3(px.s1); row0_8.s345 = tmp - row0_8.s345;\n"
			"     tmp = amd_unpack_3(px.s2); row0_8.s67  = tmp.s01 - row0_8.s67; row0_4.s0 = tmp.s2 - row0_4.s0;\n"
			"     tmp = amd_unpack_3(px.s3); row0_4.s123 = tmp - row0_4.s123;\n"
			"     px = *(__global uint4 *)(ip_buf + ip_stride);"
			"     tmp = amd_unpack_3(px.s0); row1_8.s012 = tmp - row1_8.s012;\n"
			"     tmp = amd_unpack_3(px.s1); row1_8.s345 = tmp - row1_8.s345;\n"
			"     tmp = amd_unpack_3(px.s2); row1_8.s67  = tmp.s01 - row1_8.s67; row1_4.s0 = tmp.s2 - row1_4.s0;\n"
			"     tmp = amd_unpack_3(px.s3); row1_4.s123 = tmp - row1_4.s123;\n";
		if (wt_format == VX_DF_IMAGE_U8){
			opencl_kernel_code +=
			"     if (outputValid) {\n"
			"	    uint wt_in, wt_in1; \n"
			"		wt_in  = *(__global uint *)(wt_buf);\n"
			"		wt_in1  = *(__global uint *)(wt_buf+wt_stride);\n"
			"		row0_8  *= (float8)((float3)amd_unpack0(wt_in), (float3)amd_unpack1(wt_in), (float2)amd_unpack2(wt_in)); row0_8 *= (float8)0.0627451f;\n"
			"		row0_4  *= (float4)((float)amd_unpack2(wt_in), (float3)amd_unpack3(wt_in)); row0_4 *= (float4)0.0627451f; \n"
			"		row1_8  *= (float8)((float3)amd_unpack0(wt_in1), (float3)amd_unpack1(wt_in1), (float2)amd_unpack2(wt_in1)); row1_8 *= (float8)0.0627451f;\n"
			"		row1_4  *= (float4)((float)amd_unpack2(wt_in1), (float3)amd_unpack3(wt_in1)); row1_4 *= (float4)0.0627451f;\n"
			"       *(__global short8 *)op_buf = convert_short8_sat_rte(row0_8); *(__global short4 *)(op_buf + 16) = convert_short4_sat_rte(row0_4);\n"
			"       *(__global short8 *)(op_buf + op_stride) = convert_short8_sat_rte(row1_8); *(__global short4 *)(op_buf + op_stride + 16) = convert_short4_sat_rte(row1_4);\n"
			"     }\n"
			"   }\n"
			" }\n"
			" }\n"
			"}\n";
			}
			else
			{
				opencl_kernel_code +=
			"    if (outputValid) {\n"
			"	    short4 wt_in, wt_in1; \n"
			"		wt_in  = *(__global short4 *)(wt_buf);\n"
			"		wt_in1  = *(__global short4 *)(wt_buf+wt_stride);\n"
			"		row0_8  *= (float8)((float3)wt_in.s0, (float3)wt_in.s1, (float2)wt_in.s2); row0_8 *= (float8)0.000490196f;\n"
			"		row0_4  *= (float4)((float)wt_in.s2, (float3)wt_in.s3); row0_4 *= (float4)0.000490196f; \n"
			"		row1_8  *= (float8)((float3)wt_in1.s0, (float3)wt_in1.s1, (float2)wt_in1.s2); row1_8 *= (float8)0.000490196f;\n"
			"		row1_4  *= (float4)((float)wt_in1.s2, (float3)wt_in1.s3); row1_4 *= (float4)0.000490196f;\n"
			"       *(__global short8 *)op_buf = convert_short8_sat_rte(row0_8); *(__global short4 *)(op_buf + 16) = convert_short4_sat_rte(row0_4);\n"
			"       *(__global short8 *)(op_buf + op_stride) = convert_short8_sat_rte(row1_8); *(__global short4 *)(op_buf + op_stride + 16) = convert_short4_sat_rte(row1_4);\n"
			"     }\n"
			"   }\n"
			" }\n"
			" }\n"
			"}\n";
			}
	}
	else
	{
		opencl_kernel_code +=
		"	// do vertical filtering for 4\n"
		"   {\n"
		"     short8 tmp_8; short4 tmp_4;\n"
		"     float3 tmp;\n"
		"     float8 row0_8, row1_8; float4 row0_4, row1_4;\n"
		"     tmp_8 = *(__local short8 *)&pVertFilt[0]; tmp_4 = *(__local short4 *)&pVertFilt[8];\n"
		"     row0_8 = convert_float8(tmp_8); row0_4 = convert_float4(tmp_4);\n"
		"     tmp_8 = *(__local short8 *)&pVertFilt[3*64]; tmp_4 = *(__local short4 *)&pVertFilt[3*64 + 8];\n"
		"     row0_8 = mad(convert_float8(tmp_8), (float8)(6.0f), row0_8); row0_4 = mad(convert_float4(tmp_4), (float4)(6.0f), row0_4);\n"
		"     row1_8 = 4.0f * convert_float8(tmp_8); row1_4 = 4.0f * convert_float4(tmp_4);\n"
		"     tmp_8 = *(__local short8 *)&pVertFilt[3*128]; tmp_4 = *(__local short4 *)&pVertFilt[3*128 + 8];\n"
		"     row0_8 += convert_float8(tmp_8); row0_4 += convert_float4(tmp_4);\n"
		"     row1_8 = mad(convert_float8(tmp_8), (float8)(4.0f), row1_8); row1_4 = mad(convert_float4(tmp_4), (float4)(4.0f), row1_4);\n"
		"     row0_8 *= (float8)0.015625f; row0_4 *= (float4)0.015625f;"
		"     row1_8 *= (float8)0.015625f; row1_4 *= (float4)0.015625f;"
		"     uint4 px = *(__global uint4 *)ip_buf;"
		"     tmp = amd_unpack_3(px.s0); row0_8.s012 = tmp - row0_8.s012;\n"
		"     tmp = amd_unpack_3(px.s1); row0_8.s345 = tmp - row0_8.s345;\n"
		"     tmp = amd_unpack_3(px.s2); row0_8.s67  = tmp.s01 - row0_8.s67; row0_4.s0 = tmp.s2 - row0_4.s0;\n"
		"     tmp = amd_unpack_3(px.s3); row0_4.s123 = tmp - row0_4.s123;\n"
		"     px = *(__global uint4 *)(ip_buf + ip_stride);"
		"     tmp = amd_unpack_3(px.s0); row1_8.s012 = tmp - row1_8.s012;\n"
		"     tmp = amd_unpack_3(px.s1); row1_8.s345 = tmp - row1_8.s345;\n"
		"     tmp = amd_unpack_3(px.s2); row1_8.s67  = tmp.s01 - row1_8.s67; row1_4.s0 = tmp.s2 - row1_4.s0;\n"
		"     tmp = amd_unpack_3(px.s3); row1_4.s123 = tmp - row1_4.s123;\n"
		"     if (outputValid) {\n"
		"       *(__global short8 *)op_buf = convert_short8_sat_rte(row0_8); *(__global short4 *)(op_buf + 16) = convert_short4_sat_rte(row0_4);\n"
		"       *(__global short8 *)(op_buf + op_stride) = convert_short8_sat_rte(row1_8); *(__global short4 *)(op_buf + op_stride + 16) = convert_short4_sat_rte(row1_4);\n"
		"     }\n"
		"   }\n"
		" }\n"
		" }\n"
		"}\n";
	}
	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK upscale_gaussian_subtract_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	vx_uint32 arr_offset;
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 4);				// input array
	ERROR_CHECK_OBJECT(arr);
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 1);			// input scalar - array_offset
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &arr_offset));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	StitchBlendValidEntry *pBlendArr = nullptr;
	vx_size stride_blend_arr = sizeof(StitchBlendValidEntry);
	ERROR_CHECK_STATUS(vxAccessArrayRange(arr, arr_offset - 1, arr_offset, &stride_blend_arr, (void **)&pBlendArr, VX_READ_ONLY));
	arr_numitems = *((vx_uint32 *)pBlendArr);
	ERROR_CHECK_STATUS(vxCommitArrayRange(arr, arr_offset - 1, arr_offset, pBlendArr));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	opencl_global_work[0] = arr_numitems*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1] << 1;
	return VX_SUCCESS;
}

//! \macro kernel for building Laplacian pyramids (Gaussian filter with optional upscaling)
vx_status upscale_gaussian_subtract_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.upscale_gaussian_subtract",
		AMDOVX_KERNEL_STITCHING_UPSCALE_GAUSSIAN_SUBTRACT,
		upscale_gaussian_subtract_kernel,
		7,
		upscale_gaussian_subtract_input_validator,
		upscale_gaussian_subtract_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = upscale_gaussian_subtract_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = upscale_gaussian_subtract_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = upscale_gaussian_subtract_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));
	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_OPTIONAL));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK upscale_gaussian_add_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	if (index == 0)
	{ // scalar of VX_TYPE_UINT32
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian_add numCams should be UINT32 type\n");
		}
	}
	else if (index == 1)
	{ // scalar of VX_TYPE_UINT32
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian_add arr_offs should be UINT32 type\n");
		}
	}
	else if (index == 2)
	{ // image of format VX_DF_IMAGE_RGB4_AMD
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGB4_AMD) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian image %d should be an image of RGB4 type\n", index);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 3)
	{ // image of format VX_DF_IMAGE_RGB4_AMD 
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGB4_AMD) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian image %d should be an image of RGB4 type\n", index);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 4)
	{ // array object for offsets
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchBlendValidEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_add array element (StitchBlendValidEntry) size should be 8 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK upscale_gaussian_add_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 5)
	{ // image of format VX_DF_IMAGE_RGB4_AMD of pyramid image
		vx_image image = (vx_image)avxGetNodeParamRef(node, 2);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 input_width = 0, input_height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		if (width != input_width)
			width = input_width;
		if (input_height != height)
		{ // pick default output height as the input map height
			height = input_height;
		}
		// set output image meta data
		vx_df_image format = VX_DF_IMAGE_RGB4_AMD;
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
		status = VX_SUCCESS;
	}
	return status;
}
//! \brief The kernel target support callback.
static vx_status VX_CALLBACK upscale_gaussian_add_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK upscale_gaussian_add_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK upscale_gaussian_add_opencl_codegen(
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
	vx_size		wg_num;
	vx_uint32   numCam = 0;
	vx_uint32 height1, Inheight1;
	vx_uint32 width = 0, height = 0;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);			// input scalar - numcam
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &numCam));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 5);				// output image
	vx_df_image out_format = VX_DF_IMAGE_VIRT;
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &out_format, sizeof(out_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	vx_array wg_offsets = (vx_array)avxGetNodeParamRef(node, 4);
	ERROR_CHECK_STATUS(vxQueryArray(wg_offsets, VX_ARRAY_ATTRIBUTE_CAPACITY, &wg_num, sizeof(wg_num)));
	ERROR_CHECK_STATUS(vxReleaseArray(&wg_offsets));
	// set kernel configuration
	strcpy(opencl_kernel_function_name, "upscale_gaussian_add");
	opencl_work_dim = 2;
	opencl_local_work[0] = 8;
	opencl_local_work[1] = 4;
	opencl_global_work[0] = wg_num*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1] << 1;
	image = (vx_image)avxGetNodeParamRef(node, 3);				// input image
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &Inheight1, sizeof(Inheight1)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	height1 = height; 
	if (numCam){
		height1 = (vx_uint32)(height / numCam);
		Inheight1 = (vx_uint32)(Inheight1 / numCam);
	}
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint num_cam, uint arr_offs,\n"
		" 	uint ip_width, uint ip_height, __global uchar * ip_buf, uint ip_stride, uint ip_offset,\n"
		" 	uint ip1_width, uint ip1_height, __global uchar * ip1_buf, uint ip1_stride, uint ip1_offset,\n"
		"	__global uchar * pG_buf, uint pG_offs, uint pG_num,\n"
		"   uint op_width, uint op_height, __global uchar * op_buf, uint op_stride, uint op_offset)\n"
		"{\n"
		"	int grp_id = get_global_id(0)>>3, lx = get_local_id(0), ly = get_global_id(1);\n"
		"	pG_buf += (pG_offs + (arr_offs<<3));\n"
		"	if (grp_id < pG_num) {\n"
		"		int size_x = get_local_size(0) - 1; \n"
		"		uint2 offs = ((__global uint2 *)pG_buf)[grp_id];\n"
		"		uint camera_id = offs.x & 0x1f; uint gx = (lx<<3) + ((offs.x >> 5) & 0x3FFF); uint gy = ly*2 + (offs.x >> 19);\n"
		"	    bool outputValid = (lx*8 <= (offs.y & 0xFF)) && (ly*2 <= ((offs.y >> 8) & 0xFF));\n"
		"		int border = (offs.y >> 30) & 0x3;\n"
		"		int height1 = %d;\n"
		"		ip_buf += ip_offset + mad24(gy, ip_stride, gx*6);\n"
		"		op_buf  += op_offset + mad24(gy, op_stride, gx*6);\n"
		"		ip_buf += (camera_id * ip_stride*%d);\n"
		"		ip1_buf += ip1_offset + (camera_id * ip1_stride*%d);\n"
		"		op_buf += (camera_id * op_stride*%d);\n"
		, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, Inheight1 - 1, height1, Inheight1, height1);
	opencl_kernel_code = item;

	opencl_kernel_code +=
		"		ly &= 3;\n"
		"		// load src2 (for upscale) to LDS\n"
		"		__local short lsSrc[104*6];\n"
		"		__local short lsVert[64*6*3];\n"
		"		int loffset = mad24(ly, (int)104, lx*12); \n"
		"		int2 loffs2  = (int2)((loffset + 416), 4*ip1_stride);\n"
		"		if (!border){\n"
		"			ip1_buf += mad24(gy>>1, ip1_stride, gx*3);\n"
		"			ip1_buf -= (ip1_stride + 8);\n"
		"			*(__local uint4 *)(lsSrc+loffset) = vload4(0, (global uint *)ip1_buf);\n"
		"			*(__local uint2 *)(lsSrc+loffset+8) = vload2(2, (global uint *)ip1_buf);\n"
		"			if (lx == size_x){\n"
		"			*(__local uint4 *)(lsSrc+loffset+12) = vload4(0, (global uint *)(ip1_buf+24));\n"
		"			}\n"
		"  			if (ly < 2){\n"
		"  				*(__local uint4 *)(lsSrc+loffs2.x) =  vload4(0, (global uint *)(ip1_buf+loffs2.y));\n"
		"  				*(__local uint2 *)(lsSrc+loffs2.x+8) = vload2(2, (global uint *)(ip1_buf+loffs2.y));\n"
		"  				if (lx == size_x){\n"
		"  					*(__local uint4 *)(lsSrc+loffs2.x+12) = vload4(0, (global uint *)(ip1_buf+loffs2.y+24));\n"
		"  				}\n"
		"			}\n"
		"		}else\n"
		"		{\n"
		"			int gybase = (gy>>1) - 1;\n"
		"			__global uchar *gbuf = ip1_buf + ip1_stride * max(0, min(height1, gybase));\n"
		"			int3 goffset = (int3)((gx*3)-8, (gx*3), (gx*3)+24); \n"
		"			if (!lx) {\n"
		"				goffset.s0 = select(goffset.s0, (int)(ip1_width*6-8), goffset.s0<0);\n"
		"				*(__local uint2 *)(lsSrc+loffset) = vload2(0, (global uint *)(gbuf+goffset.s0));\n"
		"			}\n"
		"			uint8 src0;\n"
		"			goffset.s1 = select(goffset.s1, 0, goffset.s1 > (ip1_width*6-12));\n"
		"			src0.s012 = vload3(0, (global uint *)(gbuf+goffset.s1));\n"
		"			int goff2 = goffset.s1+12; goff2 = select(goff2, 0, goff2 > (ip1_width*6-12)); \n"
		"			src0.s345 = vload3(0, (global uint *)(gbuf+goff2));\n"
		"			*(__local uint4 *)(lsSrc+loffset+4)  = src0.s0123;\n"
		"			*(__local uint2 *)(lsSrc+loffset+12) = src0.s45;\n"
		"			if (lx == size_x){\n"
		"			  goffset.s2 = select(goffset.s2, 0, goffset.s2 > (ip1_width*6-8));\n"
		"			  *(__local uint2 *)(lsSrc+loffset+16) = vload2(0, (global uint *)(gbuf+goffset.s2));\n"
		"			}\n"
		"			if (ly < 2){\n"
		"				gbuf = ip1_buf + ip1_stride * min(height1, gybase+4);\n"
		"				if (!lx) *(__local uint2 *)(lsSrc+loffs2.x) = vload2(0, (global uint *)(gbuf+goffset.s0));\n"
		"				src0.s012 = vload3(0, (global uint *)(gbuf + goffset.s1));\n"
		"				src0.s345 = vload3(0, (global uint *)(gbuf + goff2));\n"
		"				*(__local uint4 *)(lsSrc+loffs2.x+4) = src0.s0123;\n"
		"				*(__local uint2 *)(lsSrc+loffs2.x+12) = src0.s45;\n"
		"			    if (lx == size_x){\n"
		"			     *(__local uint2 *)(lsSrc+loffs2.x+16) = vload2(0, (global uint *)(gbuf+goffset.s2));\n"
		"				}\n"
		"			}\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		uint lvoffset = ly * 64 + (lx<<3); \n"
		"		__local short *pLSrc = (__local short *)(lsSrc +loffset);\n"
		"		__local short *pVertFilt = &lsVert[lvoffset*3];\n"
		"		__local short *pVertFilt2 = pVertFilt + 256*3; //4*64 \n"
		"		// do horizontal filter pass\n"
		"		{\n"
		"       short16 pix = vload16(0, pLSrc); short4 pix_4 = vload4(0, (pLSrc+16));\n"
		"       short8 filtpix_8;\n"
		"       short4 filtpix_4;\n"
		"       filtpix_8.s012 = ((short3)6 * pix.s456) + pix.s123 + pix.s789; filtpix_8.s345 = (short3)4 * (pix.s456 + pix.s789);\n"
		"       filtpix_8.s67 = ((short2)6 * pix.s78) + pix.s45 + pix.sAB; filtpix_4.s0 = ((short)6 * pix.s9) + pix.s6 + pix.sC; filtpix_4.s123 = (short3)4 * (pix.s789 + pix.sABC);\n"
		"       *(__local short8 *)pVertFilt = filtpix_8; *(__local short4 *)(pVertFilt + 8) = filtpix_4;\n"
		"       pix.s01234567 = pix.s6789ABCD; pix.s89 = pix.sEF; pix.sABCD = pix_4;\n"
		"       filtpix_8.s012 = ((short3)6 * pix.s456) + pix.s123 + pix.s789; filtpix_8.s345 = (short3)4 * (pix.s456 + pix.s789);\n"
		"       filtpix_8.s67 = ((short2)6 * pix.s78) + pix.s45 + pix.sAB; filtpix_4.s0 = ((short)6 * pix.s9) + pix.s6 + pix.sC; filtpix_4.s123 = (short3)4 * (pix.s789 + pix.sABC);\n"
		"       *(__local short8 *)(pVertFilt + 12) = filtpix_8; *(__local short4 *)(pVertFilt + 20) = filtpix_4;\n"
		"		}\n"
		"		if (ly < 2){\n"
		"			pLSrc =  (__local short *)(lsSrc + loffs2.x);\n"
		"			short16 pix = vload16(0, pLSrc); short4 pix_4 = vload4(0, (pLSrc+16));\n"
		"			short8 filtpix_8;\n"
		"			short4 filtpix_4;\n"
		"			filtpix_8.s012 = ((short3)6 * pix.s456) + pix.s123 + pix.s789; filtpix_8.s345 = (short3)4 * (pix.s456 + pix.s789);\n"
		"			filtpix_8.s67 = ((short2)6 * pix.s78) + pix.s45 + pix.sAB; filtpix_4.s0 = ((short)6 * pix.s9) + pix.s6 + pix.sC; filtpix_4.s123 = (short3)4 * (pix.s789 + pix.sABC);\n"
		"			*(__local short8 *)pVertFilt2 = filtpix_8; *(__local short4 *)(pVertFilt2 + 8) = filtpix_4;\n"
		"			pix.s01234567 = pix.s6789ABCD; pix.s89 = pix.sEF; pix.sABCD = pix_4;\n"
		"			filtpix_8.s012 = ((short3)6 * pix.s456) + pix.s123 + pix.s789; filtpix_8.s345 = (short3)4 * (pix.s456 + pix.s789);\n"
		"			filtpix_8.s67 = ((short2)6 * pix.s78) + pix.s45 + pix.sAB; filtpix_4.s0 = ((short)6 * pix.s9) + pix.s6 + pix.sC; filtpix_4.s123 = (short3)4 * (pix.s789 + pix.sABC);\n"
		"			*(__local short8 *)(pVertFilt2 + 12) = filtpix_8; *(__local short4 *)(pVertFilt2 + 20) = filtpix_4;\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		// do vertical filtering for 8 \n"
		"		__local short *pV0 = pVertFilt; \n"
		"		__local short *pV1 = pVertFilt + 64*3; \n"
		"		__local short *pV2 = pVertFilt + 128*3;\n"
		"       short8 tmp_8; short4 tmp_4; \n"
		"       float8 row0_8, row1_8; float4 row0_4, row1_4;\n"
		"       tmp_8 = *(__local short8 *)&pV0[0]; tmp_4 = *(__local short4 *)&pV0[8];\n"
		"       row0_8 = convert_float8(tmp_8); row0_4 = convert_float4(tmp_4);\n"
		"       tmp_8 = *(__local short8 *)&pV1[0]; tmp_4 = *(__local short4 *)&pV1[8];\n"
		"       row0_8 = mad(convert_float8(tmp_8), (float8)(6.0f), row0_8); row0_4 = mad(convert_float4(tmp_4), (float4)(6.0f), row0_4);\n"
		"       row1_8 = 4.0f * convert_float8(tmp_8); row1_4 = 4.0f * convert_float4(tmp_4);\n"
		"       tmp_8 = *(__local short8 *)&pV2[0]; tmp_4 = *(__local short4 *)&pV2[8]; \n"
		"       row0_8 += convert_float8(tmp_8); row0_4 += convert_float4(tmp_4);\n"
		"       row1_8 = mad(convert_float8(tmp_8), (float8)(4.0f), row1_8); row1_4 = mad(convert_float4(tmp_4), (float4)(4.0f), row1_4);\n"
		"       row0_8 *= (float8)0.015625f; row0_4 *= (float4)0.015625f;\n"
		"       row1_8 *= (float8)0.015625f; row1_4 *= (float4)0.015625f;\n"
		"       tmp_8 = *(__global short8 *)(ip_buf); tmp_4 = *(__global short4 *)(ip_buf + 16);\n"
		"       tmp_8 += convert_short8(row0_8); tmp_4 += convert_short4(row0_4);\n"
		"       if (outputValid) {\n"
		"         *(__global short8 *)(op_buf) = tmp_8; *(__global short4 *)(op_buf + 16) = tmp_4;\n"
		"       }\n"
		"       tmp_8 = *(__global short8 *)(ip_buf + ip_stride); tmp_4 = *(__global short4 *)(ip_buf + ip_stride + 16);\n"
		"       tmp_8 += convert_short8(row1_8); tmp_4 += convert_short4(row1_4);\n"
		"       if (outputValid) {\n"
		"         *(__global short8 *)(op_buf + op_stride) = tmp_8; *(__global short4 *)(op_buf + op_stride + 16) = tmp_4;\n"
		"       }\n"
		"       tmp_8 = *(__local short8 *)&pV0[12]; tmp_4 = *(__local short4 *)&pV0[20];\n"
		"       row0_8 = convert_float8(tmp_8); row0_4 = convert_float4(tmp_4);\n"
		"       tmp_8 = *(__local short8 *)&pV1[12]; tmp_4 = *(__local short4 *)&pV1[20];\n"
		"       row0_8 = mad(convert_float8(tmp_8), (float8)(6.0f), row0_8); row0_4 = mad(convert_float4(tmp_4), (float4)(6.0f), row0_4);\n"
		"       row1_8 = 4.0f * convert_float8(tmp_8); row1_4 = 4.0f * convert_float4(tmp_4);\n"
		"       tmp_8 = *(__local short8 *)&pV2[12]; tmp_4 = *(__local short4 *)&pV2[20]; \n"
		"       row0_8 += convert_float8(tmp_8); row0_4 += convert_float4(tmp_4);\n"
		"       row1_8 = mad(convert_float8(tmp_8), (float8)(4.0f), row1_8); row1_4 = mad(convert_float4(tmp_4), (float4)(4.0f), row1_4);\n"
		"       row0_8 *= (float8)0.015625f; row0_4 *= (float4)0.015625f;\n"
		"       row1_8 *= (float8)0.015625f; row1_4 *= (float4)0.015625f;\n"
		"       tmp_8 = *(__global short8 *)(ip_buf + 24); tmp_4 = *(__global short4 *)(ip_buf + 40);\n"
		"       tmp_8 += convert_short8_sat_rte(row0_8); tmp_4 += convert_short4_sat_rte(row0_4);\n"
		"       if (outputValid) {\n"
		"         *(__global short8 *)(op_buf + 24) = tmp_8; *(__global short4 *)(op_buf + 40) = tmp_4;\n"
		"       }\n"
		"       tmp_8 = *(__global short8 *)(ip_buf + ip_stride + 24); tmp_4 = *(__global short4 *)(ip_buf + ip_stride + 40);\n"
		"       tmp_8 += convert_short8_sat_rte(row1_8); tmp_4 += convert_short4_sat_rte(row1_4);\n"
		"       if (outputValid) {\n"
		"         *(__global short8 *)(op_buf + op_stride + 24) = tmp_8; *(__global short4 *)(op_buf + op_stride + 40) = tmp_4;\n"
		"       }\n"
		"	}\n"
		"}\n";
	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK upscale_gaussian_add_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	vx_uint32 arr_offset;
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 4);				// input array
	ERROR_CHECK_OBJECT(arr);
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 1);			// input scalar - array_offset
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &arr_offset));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	StitchBlendValidEntry *pBlendArr = nullptr;
	vx_size stride_blend_arr = sizeof(StitchBlendValidEntry);
	ERROR_CHECK_STATUS(vxAccessArrayRange(arr, arr_offset - 1, arr_offset, &stride_blend_arr, (void **)&pBlendArr, VX_READ_ONLY));
	arr_numitems = *((vx_uint32 *)pBlendArr);
	ERROR_CHECK_STATUS(vxCommitArrayRange(arr, arr_offset - 1, arr_offset, pBlendArr));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	opencl_global_work[0] = arr_numitems*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1]<<1;
	return VX_SUCCESS;
}

//! \macro kernel for building Laplacian pyramids (Gaussian filter with optional upscaling)
vx_status upscale_gaussian_add_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.upscale_gaussian_add",
		AMDOVX_KERNEL_STITCHING_UPSCALE_GAUSSIAN_ADD,
		upscale_gaussian_add_kernel,
		6,
		upscale_gaussian_add_input_validator,
		upscale_gaussian_add_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);
	amd_kernel_query_target_support_f query_target_support_f = upscale_gaussian_add_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = upscale_gaussian_add_opencl_codegen;
	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = upscale_gaussian_add_opencl_global_work_update;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));
 
	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	return VX_SUCCESS;
}

//! \brief The input validator callback.
static vx_status VX_CALLBACK laplacian_reconstruct_input_validator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	// get reference for parameter at specified index
	vx_reference ref = avxGetNodeParamRef(node, index);
	ERROR_CHECK_OBJECT(ref);
	if (index == 0)
	{ // scalar of VX_TYPE_UINT32
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: laplacian_recon numCams should be UINT32 type\n");
		}
	}
	if (index == 1)
	{ // scalar of VX_TYPE_UINT32
		vx_enum itemtype = VX_TYPE_INVALID;
		ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &itemtype, sizeof(itemtype)));
		if (itemtype == VX_TYPE_UINT32) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: laplacian_recon arr_offs should be UINT32 type\n");
		}
	}
	else if (index == 2)
	{ // image of format VX_DF_IMAGE_RGB4_AMD
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGB4_AMD) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian image %d should be an image of RGB2 type\n", index);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 3)
	{ // image of format VX_DF_IMAGE_RGB4_AMD 
		// check input image format and dimensions
		vx_uint32 input_width = 0, input_height = 0;
		vx_df_image input_format = VX_DF_IMAGE_VIRT;
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &input_format, sizeof(input_format)));
		ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
		if (input_format != VX_DF_IMAGE_RGB4_AMD) {
			status = VX_ERROR_INVALID_TYPE;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_gaussian image %d should be an image of RGB4 type\n", index);
		}
		else {
			status = VX_SUCCESS;
		}
	}
	else if (index == 4)
	{ // array object for offsets
		vx_size itemsize = 0;
		ERROR_CHECK_STATUS(vxQueryArray((vx_array)ref, VX_ARRAY_ATTRIBUTE_ITEMSIZE, &itemsize, sizeof(itemsize)));
		if (itemsize == sizeof(StitchBlendValidEntry)) {
			status = VX_SUCCESS;
		}
		else {
			status = VX_ERROR_INVALID_DIMENSION;
			vxAddLogEntry((vx_reference)node, status, "ERROR: upscale_add array element (StitchBlendValidEntry) size should be 32 bytes\n");
		}
		ERROR_CHECK_STATUS(vxReleaseArray((vx_array *)&ref));
	}
	return status;
}

//! \brief The output validator callback.
static vx_status VX_CALLBACK laplacian_reconstruct_output_validator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_ERROR_INVALID_PARAMETERS;
	if (index == 5)
	{ // image of format RGBX of pyramid image
		vx_image image = (vx_image)avxGetNodeParamRef(node, 2);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 input_width = 0, input_height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &input_width, sizeof(input_width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &input_height, sizeof(input_height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		image = (vx_image)avxGetNodeParamRef(node, index);
		ERROR_CHECK_OBJECT(image);
		vx_uint32 width = 0, height = 0;
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
		ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
		ERROR_CHECK_STATUS(vxReleaseImage(&image));
		if (width != input_width)
			width = input_width;
		if (input_height != height)
		{ // pick default output height as the input map height
			height = input_height;
		}
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
static vx_status VX_CALLBACK laplacian_reconstruct_query_target_support(vx_graph graph, vx_node node,
	vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
	vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
	)
{
	supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
	return VX_SUCCESS;
}

//! \brief The kernel execution.
static vx_status VX_CALLBACK laplacian_reconstruct_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
	return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The OpenCL code generator callback.
static vx_status VX_CALLBACK laplacian_reconstruct_opencl_codegen(
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
	vx_size		wg_num;
	vx_uint32   numCam = 0;
	vx_uint32 height1, Inheight1;
	vx_uint32 width = 0, height = 0;
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 0);			// input scalar - grayscale compute method
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &numCam));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	vx_image image = (vx_image)avxGetNodeParamRef(node, 5);				// output image
	vx_df_image out_format = VX_DF_IMAGE_VIRT;
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &out_format, sizeof(out_format)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	vx_array wg_offsets = (vx_array)avxGetNodeParamRef(node, 4);
	ERROR_CHECK_STATUS(vxQueryArray(wg_offsets, VX_ARRAY_ATTRIBUTE_CAPACITY, &wg_num, sizeof(wg_num)));
	ERROR_CHECK_STATUS(vxReleaseArray(&wg_offsets));
	// set kernel configuration
	strcpy(opencl_kernel_function_name, "laplacian_reconstruct");
	opencl_work_dim = 2;
	opencl_local_work[0] = 8;
	opencl_local_work[1] = 4;
	opencl_global_work[0] = wg_num*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1]<<1;
	image = (vx_image)avxGetNodeParamRef(node, 3);				// input image
	ERROR_CHECK_OBJECT(image);
	ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_HEIGHT, &Inheight1, sizeof(Inheight1)));
	ERROR_CHECK_STATUS(vxReleaseImage(&image));
	height1 = height;
	if (numCam){
		height1 = (vx_uint32)(height / numCam);
		Inheight1 = (vx_uint32)(Inheight1 / numCam);
	}
	char item[8192];
	snprintf(item, sizeof(item),
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint num_cam, uint arr_offs,\n"
		" 	uint ip_width, uint ip_height, __global uchar * ip_buf, uint ip_stride, uint ip_offset,\n"
		" 	uint ip1_width, uint ip1_height, __global uchar * ip1_buf, uint ip1_stride, uint ip1_offset,\n"
		"	__global uchar * pG_buf, uint pG_offs, uint pG_num,\n"
		"   uint op_width, uint op_height, __global uchar * op_buf, uint op_stride, uint op_offset)\n"
		"{\n"
		"	int grp_id = get_global_id(0)>>3, lx = get_local_id(0), ly = get_global_id(1);\n"
		"	pG_buf += (pG_offs + (arr_offs<<3));\n"
		"	if (grp_id < pG_num) {\n"
		"		int size_x = get_local_size(0) - 1; \n"
		"		uint2 offs = ((__global uint2 *)pG_buf)[grp_id];\n"
		"		uint camera_id = offs.x & 0x1f; uint gx = (lx<<3) + ((offs.x >> 5) & 0x3FFF); uint gy = ly*2 + (offs.x >> 19);\n"
		"	    bool outputValid = (lx*8 <= (offs.y & 0xFF)) && (ly*2 <= ((offs.y >> 8) & 0xFF));\n"
		"		int border = (offs.y >> 30) & 0x3;\n"
		"		int height1 = %d;\n"
		"		ip_buf += ip_offset + mad24(gy, ip_stride, gx*6);\n"
		"		op_buf  += op_offset + mad24(gy, op_stride, gx*4);\n"
		"		ip_buf += (camera_id * ip_stride*%d);\n"
		"		ip1_buf += ip1_offset + (camera_id * ip1_stride*%d);\n"
		"		op_buf += (camera_id * op_stride*%d);\n"
		, (int)opencl_local_work[0], (int)opencl_local_work[1], opencl_kernel_function_name, Inheight1 - 1, height1, Inheight1, height1);
	opencl_kernel_code = item;

	opencl_kernel_code +=
		"		ly = ly & 3;\n"
		"		// load src2 (for upscale) to LDS\n"
		"		__local short lsSrc[104*6];\n"
		"		__local float lsVert[64*6*3];\n"
		"		int loffset = mad24(ly, (int)104, lx*12); \n"
		"		int2 loffs2  = (int2)((loffset + 416), 4*ip1_stride);\n"
		"		if (!border){\n"
		"			ip1_buf += mad24(gy>>1, ip1_stride, gx*3);\n"
		"			ip1_buf -= (ip1_stride + 8);\n"
		"			*(__local uint4 *)(lsSrc+loffset) = vload4(0, (global uint *)ip1_buf);\n"
		"			*(__local uint2 *)(lsSrc+loffset+8) = vload2(2, (global uint *)ip1_buf);\n"
		"			if (lx == size_x){\n"
		"			*(__local uint4 *)(lsSrc+loffset+12) = vload4(0, (global uint *)(ip1_buf+24));\n"
		"			}\n"
		"  			if (ly < 2){\n"
		"  				*(__local uint4 *)(lsSrc+loffs2.x) =  vload4(0, (global uint *)(ip1_buf+loffs2.y));\n"
		"  				*(__local uint2 *)(lsSrc+loffs2.x+8) = vload2(2, (global uint *)(ip1_buf+loffs2.y));\n"
		"  				if (lx == size_x){\n"
		"  					*(__local uint4 *)(lsSrc+loffs2.x+12) = vload4(0, (global uint *)(ip1_buf+loffs2.y+24));\n"
		"  				}\n"
		"			}\n"
		"		}else\n"
		"		{\n"
		"			int gybase = (gy>>1) - 1;\n"
		"			__global uchar *gbuf = ip1_buf + ip1_stride * max(0, min(height1, gybase));\n"
		"			int3 goffset = (int3)((gx*3)-8, (gx*3), (gx*3)+24); \n"
		"			if (!lx) {\n"
		"				goffset.s0 = select(goffset.s0, (int)(ip1_width*6-8), goffset.s0<0);\n"
		"				*(__local uint2 *)(lsSrc+loffset) = vload2(0, (global uint *)(gbuf+goffset.s0));\n"
		"			}\n"
		"			uint8 src0;\n"
		"			goffset.s1 = select(goffset.s1, 0, goffset.s1 > (ip1_width*6-12));\n"
		"			src0.s012 = vload3(0, (global uint *)(gbuf+goffset.s1));\n"
		"			int goff2 = goffset.s1+12; goff2 = select(goff2, 0, goff2 > (ip1_width*6-12)); \n"
		"			src0.s345 = vload3(0, (global uint *)(gbuf+goff2));\n"
		"			*(__local uint4 *)(lsSrc+loffset+4)  = src0.s0123;\n"
		"			*(__local uint2 *)(lsSrc+loffset+12) = src0.s45;\n"
		"			if (lx == size_x){\n"
		"			  goffset.s2 = select(goffset.s2, 0, goffset.s2 > (ip1_width*6-8));\n"
		"			  *(__local uint2 *)(lsSrc+loffset+16) = vload2(0, (global uint *)(gbuf+goffset.s2));\n"
		"			}\n"
		"			if (ly < 2){\n"
		"				gbuf = ip1_buf + ip1_stride * min(height1, gybase+4);\n"
		"				if (!lx) *(__local uint2 *)(lsSrc+loffs2.x) = vload2(0, (global uint *)(gbuf+goffset.s0));\n"
		"				src0.s012 = vload3(0, (global uint *)(gbuf + goffset.s1));\n"
		"				src0.s345 = vload3(0, (global uint *)(gbuf + goff2));\n"
		"				*(__local uint4 *)(lsSrc+loffs2.x+4) = src0.s0123;\n"
		"				*(__local uint2 *)(lsSrc+loffs2.x+12) = src0.s45;\n"
		"			    if (lx == size_x){\n"
		"			     *(__local uint2 *)(lsSrc+loffs2.x+16) = vload2(0, (global uint *)(gbuf+goffset.s2));\n"
		"				}\n"
		"			}\n"
		"		}\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		uint lvoffset = ly * 64 + (lx << 3); \n"
		"		__local short *pLSrc = (__local short *)(lsSrc + loffset);\n"
		"		__local float *pVertFilt = &lsVert[lvoffset*3];\n"
		"		__local float *pVertFilt2 = pVertFilt + 768; //256*3 \n"
		"       float8 filtpix_8; float4 filtpix_4;\n"
		"		// do horizontal filter pass\n"
		"	{\n"
		"       short16 spix = vload16(0, pLSrc); short4 spix_4 = vload4(0, (pLSrc+16));\n"
		"       float16 pix = convert_float16((short16)(spix.s12345678, spix.s9ABC, spix.sDEF, spix_4.s0)) ;\n"
		"       float2 pix_2 = convert_float2((short2)spix_4.s12);\n"
		"       filtpix_8.s012 = ((float3)6.0f * pix.s345) + pix.s012 + pix.s678; filtpix_8.s345 = (float3)4.0f * (pix.s345 + pix.s678);\n"
		"       filtpix_8.s67 = ((float2)6.0f * pix.s67) + pix.s34 + pix.s9A; filtpix_4.s0 = (float)((6.0f * pix.s8) + pix.s5 + pix.sB); filtpix_4.s123 = (float3)4.0f * (pix.s678 + pix.s9AB);\n"
		"       *(__local float8 *)pVertFilt = filtpix_8; *(__local float4 *)(pVertFilt + 8) = filtpix_4;\n"
		"       pix.s01234567 = pix.s6789ABCD; pix.s89 = pix.sEF; pix.sAB = pix_2;\n"
		"       filtpix_8.s012 = ((float3)6.0f * pix.s345) + pix.s012 + pix.s678; filtpix_8.s345 = (float3)4.0f * (pix.s345 + pix.s678);\n"
		"       filtpix_8.s67 = ((float2)6.0f * pix.s67) + pix.s34 + pix.s9A; filtpix_4.s0 = (float)((6.0f * pix.s8) + pix.s5 + pix.sB); filtpix_4.s123 = (float3)4.0f * (pix.s678 + pix.s9AB);\n"
		"       *(__local float8 *)(pVertFilt + 12) = filtpix_8; *(__local float4 *)(pVertFilt + 20) = filtpix_4;\n"
		"    }\n"
		"    if (ly < 2){\n"
		"        pLSrc =  (__local short *)(lsSrc + loffs2.x);\n"
		"        short16 spix = vload16(0, pLSrc); short4 spix_4 = vload4(0, (pLSrc+16));\n"
		"        float16 pix = convert_float16((short16)(spix.s12345678, spix.s9ABC, spix.sDEF, spix_4.s0)) ;\n"
		"        float2 pix_2 = convert_float2((short2)spix_4.s12);\n"
		"        filtpix_8.s012 = ((float3)6.0f * pix.s345) + pix.s012 + pix.s678; filtpix_8.s345 = (float3)4.0f * (pix.s345 + pix.s678);\n"
		"        filtpix_8.s67 = ((float2)6.0f * pix.s67) + pix.s34 + pix.s9A; filtpix_4.s0 = (float)((6.0f * pix.s8) + pix.s5 + pix.sB); filtpix_4.s123 = (float3)4.0f * (pix.s678 + pix.s9AB);\n"
		"        *(__local float8 *)pVertFilt2 = filtpix_8; *(__local float4 *)(pVertFilt2 + 8) = filtpix_4;\n"
		"        pix.s01234567 = pix.s6789ABCD; pix.s89 = pix.sEF; pix.sAB = pix_2;\n"
		"        filtpix_8.s012 = ((float3)6.0f * pix.s345) + pix.s012 + pix.s678; filtpix_8.s345 = (float3)4.0f * (pix.s345 + pix.s678);\n"
		"        filtpix_8.s67 = ((float2)6.0f * pix.s67) + pix.s34 + pix.s9A; filtpix_4.s0 = (float)((6.0f * pix.s8) + pix.s5 + pix.sB); filtpix_4.s123 = (float3)4.0f * (pix.s678 + pix.s9AB);\n"
		"        *(__local float8 *)(pVertFilt2 + 12) = filtpix_8; *(__local float4 *)(pVertFilt2 + 20) = filtpix_4;\n"
		"    }\n"
		"		barrier(CLK_LOCAL_MEM_FENCE);\n"
		"		// do vertical filtering for 8 pixels\n"
		"		__local float *pV0 = pVertFilt;\n"
		"		__local float *pV1 = pVertFilt + 64*3;\n"
		"		__local float *pV2 = pVertFilt + 128*3;\n"
		"       float8 row0_8, row1_8; float4 row0_4, row1_4;\n"
		"       uint4 op_pix0, op_pix1;"
		"       row0_8 = *(__local float8 *)&pV0[0]; row0_4 = *(__local float4 *)&pV0[8];\n"
		"       filtpix_8 = *(__local float8 *)&pV1[0]; filtpix_4 = *(__local float4 *)&pV1[8];\n"
		"       row0_8 = mad(filtpix_8, (float8)(6.0f), row0_8); row0_4 = mad(filtpix_4, (float4)(6.0f), row0_4);\n"
		"       row1_8 = 4.0f * filtpix_8; row1_4 = 4.0f * filtpix_4;\n"
		"       filtpix_8 = *(__local float8 *)&pV2[0]; filtpix_4 = *(__local float4 *)&pV2[8]; \n"
		"       row0_8 += filtpix_8; row0_4 += filtpix_4;\n"
		"       row1_8 = mad(filtpix_8, (float8)(4.0f), row1_8); row1_4 = mad(filtpix_4, (float4)(4.0f), row1_4);\n"
		"       row0_8 *= (float8)0.015625f; row0_4 *= (float4)0.015625f;\n"
		"       row1_8 *= (float8)0.015625f; row1_4 *= (float4)0.015625f;\n"
		"       filtpix_8 = convert_float8(*(__global short8 *)(ip_buf)); filtpix_4 = convert_float4(*(__global short4 *)(ip_buf + 16));\n"
		"       row0_8 += filtpix_8; row0_4 += filtpix_4;\n"
		"       row0_8 *= (float8)0.0625f; row0_4 *= (float4)0.0625f;\n"
		"       op_pix0.s0 = amd_pack((float4)(row0_8.s0, row0_8.s1, row0_8.s2, 255.0f)); op_pix0.s1 = amd_pack((float4)(row0_8.s3, row0_8.s4, row0_8.s5, 255.0f));\n"
		"       op_pix0.s2 = amd_pack((float4)(row0_8.s6, row0_8.s7, row0_4.s0, 255.0f)); op_pix0.s3 = amd_pack((float4)(row0_4.s1, row0_4.s2, row0_4.s3, 255.0f));\n"
		"       filtpix_8 = convert_float8(*(__global short8 *)(ip_buf + ip_stride)); filtpix_4 = convert_float4(*(__global short4 *)(ip_buf + ip_stride + 16));\n"
		"       row1_8 += filtpix_8; row1_4 += filtpix_4;\n"
		"       row1_8 *= (float8)0.0625f; row1_4 *= (float4)0.0625f;\n"
		"       op_pix1.s0 = amd_pack((float4)(row1_8.s0, row1_8.s1, row1_8.s2, 255.0f)); op_pix1.s1 = amd_pack((float4)(row1_8.s3, row1_8.s4, row1_8.s5, 255.0f));\n"
		"       op_pix1.s2 = amd_pack((float4)(row1_8.s6, row1_8.s7, row1_4.s0, 255.0f)); op_pix1.s3 = amd_pack((float4)(row1_4.s1, row1_4.s2, row1_4.s3, 255.0f));\n"
		"       if (outputValid) {\n"
		"         *(__global uint4 *)(op_buf) = op_pix0; *(__global uint4 *)(op_buf + op_stride) = op_pix1;\n"
		"       }\n"
		"       //tmp_8 = *(__local short8 *)&pV0[12]; tmp_4 = *(__local short4 *)&pV0[20];\n"
		"       row0_8 = *(__local float8 *)&pV0[12]; row0_4 = *(__local float4 *)&pV0[20];\n"
		"       filtpix_8 = *(__local float8 *)&pV1[12]; filtpix_4 = *(__local float4 *)&pV1[20];\n"
		"       row0_8 = mad(filtpix_8, (float8)(6.0f), row0_8); row0_4 = mad(filtpix_4, (float4)(6.0f), row0_4);\n"
		"       row1_8 = 4.0f * filtpix_8; row1_4 = 4.0f * filtpix_4;\n"
		"       filtpix_8 = *(__local float8 *)&pV2[12]; filtpix_4 = *(__local float4 *)&pV2[20]; \n"
		"       row0_8 += filtpix_8; row0_4 += filtpix_4;\n"
		"       row1_8 = mad(filtpix_8, (float8)(4.0f), row1_8); row1_4 = mad(filtpix_4, (float4)(4.0f), row1_4);\n"
		"       row0_8 *= (float8)0.015625f; row0_4 *= (float4)0.015625f;\n"
		"       row1_8 *= (float8)0.015625f; row1_4 *= (float4)0.015625f;\n"
		"       filtpix_8 = convert_float8(*(__global short8 *)(ip_buf + 24)); filtpix_4 = convert_float4(*(__global short4 *)(ip_buf + 40));\n"
		"       row0_8 += filtpix_8; row0_4 += filtpix_4;\n"
		"       row0_8 *= (float8)0.0625f; row0_4 *= (float4)0.0625f;\n"
		"       op_pix0.s0 = amd_pack((float4)(row0_8.s0, row0_8.s1, row0_8.s2, 255.0f)); op_pix0.s1 = amd_pack((float4)(row0_8.s3, row0_8.s4, row0_8.s5, 255.0f));\n"
		"       op_pix0.s2 = amd_pack((float4)(row0_8.s6, row0_8.s7, row0_4.s0, 255.0f)); op_pix0.s3 = amd_pack((float4)(row0_4.s1, row0_4.s2, row0_4.s3, 255.0f));\n"
		"       filtpix_8 = convert_float8(*(__global short8 *)(ip_buf + ip_stride + 24)); filtpix_4 = convert_float4(*(__global short4 *)(ip_buf + ip_stride + 40));\n"
		"       row1_8 += filtpix_8; row1_4 += filtpix_4;\n"
		"       row1_8 *= (float8)0.0625f; row1_4 *= (float4)0.0625f;\n"
		"       op_pix1.s0 = amd_pack((float4)(row1_8.s0, row1_8.s1, row1_8.s2, 255.0f)); op_pix1.s1 = amd_pack((float4)(row1_8.s3, row1_8.s4, row1_8.s5, 255.0f));\n"
		"       op_pix1.s2 = amd_pack((float4)(row1_8.s6, row1_8.s7, row1_4.s0, 255.0f)); op_pix1.s3 = amd_pack((float4)(row1_4.s1, row1_4.s2, row1_4.s3, 255.0f));\n"
		"       if (outputValid) {\n"
		"         *(__global uint4 *)(op_buf + 16) = op_pix0; *(__global uint4 *)(op_buf + op_stride + 16) = op_pix1;\n"
		"       }\n"
		"	}\n"
		"}\n";
	return VX_SUCCESS;
}

//! \brief The OpenCL global work updater callback.
static vx_status VX_CALLBACK laplacian_reconstruct_opencl_global_work_update(
	vx_node node,                                  // [input] node
	const vx_reference parameters[],               // [input] parameters
	vx_uint32 num,                                 // [input] number of parameters
	vx_uint32 opencl_work_dim,                     // [input] work_dim for clEnqueueNDRangeKernel()
	vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
	const vx_size opencl_local_work[]              // [input] local_work[] for clEnqueueNDRangeKernel()
	)
{
	vx_uint32 arr_offset;
	// Get the number of elements in the stitchWarpRemapEntry array
	vx_size arr_numitems = 0;
	vx_array arr = (vx_array)avxGetNodeParamRef(node, 4);				// input array
	ERROR_CHECK_OBJECT(arr);
	vx_scalar scalar = (vx_scalar)avxGetNodeParamRef(node, 1);			// input scalar - grayscale compute method
	ERROR_CHECK_OBJECT(scalar);
	ERROR_CHECK_STATUS(vxReadScalarValue(scalar, &arr_offset));
	ERROR_CHECK_STATUS(vxReleaseScalar(&scalar));
	StitchBlendValidEntry *pBlendArr = nullptr;
	vx_size stride_blend_arr = sizeof(StitchBlendValidEntry);
	ERROR_CHECK_STATUS(vxAccessArrayRange(arr, arr_offset - 1, arr_offset, &stride_blend_arr, (void **)&pBlendArr, VX_READ_ONLY));
	arr_numitems = *((vx_uint32 *)pBlendArr);
	ERROR_CHECK_STATUS(vxCommitArrayRange(arr, arr_offset - 1, arr_offset, pBlendArr));
	ERROR_CHECK_STATUS(vxReleaseArray(&arr));
	opencl_global_work[0] = arr_numitems*opencl_local_work[0];
	opencl_global_work[1] = opencl_local_work[1]<<1;
	return VX_SUCCESS;
}

//! \macro kernel for building Laplacian pyramids (Gaussian filter with optional upscaling)
vx_status laplacian_reconstruct_publish(vx_context context)
{
	// add kernel to the context with callbacks
	vx_kernel kernel = vxAddKernel(context, "com.amd.loomsl.laplacian_reconstruct",
		AMDOVX_KERNEL_STITCHING_LAPLACIAN_RECONSTRUCT,
		laplacian_reconstruct_kernel,
		6,
		laplacian_reconstruct_input_validator,
		laplacian_reconstruct_output_validator,
		nullptr,
		nullptr);
	ERROR_CHECK_OBJECT(kernel);

	amd_kernel_opencl_global_work_update_callback_f opencl_global_work_update_callback_f = laplacian_reconstruct_opencl_global_work_update;
	amd_kernel_query_target_support_f query_target_support_f = laplacian_reconstruct_query_target_support;
	amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = laplacian_reconstruct_opencl_codegen;
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));
	ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_GLOBAL_WORK_UPDATE_CALLBACK, &opencl_global_work_update_callback_f, sizeof(opencl_global_work_update_callback_f)));
	// set kernel parameters
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
	ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
	// finalize and release kernel object
	ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
	ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	return VX_SUCCESS;
}
