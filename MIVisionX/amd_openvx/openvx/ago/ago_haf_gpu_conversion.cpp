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


#include "ago_haf_gpu.h"

#if ENABLE_OPENCL

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following channel combine kernels
//   VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_UYVY
//   VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_YUYV
//
int HafGpu_ChannelCombine_U32_422(AgoNode * node)
{
	int status = VX_SUCCESS;

	// configuration
	vx_enum kernel = node->akernel->id;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;
	int stride0 = node->paramList[0]->u.img.stride_in_bytes;
	int stride1 = node->paramList[1]->u.img.stride_in_bytes;
	int stride2 = node->paramList[2]->u.img.stride_in_bytes;
	int stride3 = node->paramList[3]->u.img.stride_in_bytes;
	int work_group_width = 16;
	int work_group_height = 4;

	char combineCode[1024];
	if (kernel == VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_UYVY) {
		snprintf(combineCode, sizeof(combineCode),
			OPENCL_FORMAT(
			"    out.s0 = amd_pack((float4)(amd_unpack0(pU), amd_unpack0(pY.s0), amd_unpack0(pV), amd_unpack1(pY.s0)));\n"
			"    out.s1 = amd_pack((float4)(amd_unpack1(pU), amd_unpack2(pY.s0), amd_unpack1(pV), amd_unpack3(pY.s0)));\n"
			"    out.s2 = amd_pack((float4)(amd_unpack2(pU), amd_unpack0(pY.s1), amd_unpack2(pV), amd_unpack1(pY.s1)));\n"
			"    out.s3 = amd_pack((float4)(amd_unpack3(pU), amd_unpack2(pY.s1), amd_unpack3(pV), amd_unpack3(pY.s1)));\n"
			));
	}
	else if (kernel == VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_YUYV) {
		snprintf(combineCode, sizeof(combineCode),
			OPENCL_FORMAT(
			"    out.s0 = amd_pack((float4)(amd_unpack0(pY.s0), amd_unpack0(pU), amd_unpack1(pY.s0), amd_unpack0(pV)));\n"
			"    out.s1 = amd_pack((float4)(amd_unpack2(pY.s0), amd_unpack1(pU), amd_unpack3(pY.s0), amd_unpack1(pV)));\n"
			"    out.s2 = amd_pack((float4)(amd_unpack0(pY.s1), amd_unpack2(pU), amd_unpack1(pY.s1), amd_unpack2(pV)));\n"
			"    out.s3 = amd_pack((float4)(amd_unpack2(pY.s1), amd_unpack3(pU), amd_unpack3(pY.s1), amd_unpack3(pV)));\n"
			));
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_ChannelCombine_U32_422 doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}

	// kernel body
	char item[8192];
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset,\n"
		"        uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset,\n"
		"        uint p2_width, uint p2_height, __global uchar * p2_buf, uint p2_stride, uint p2_offset,\n"
		"        uint p3_width, uint p3_height, __global uchar * p3_buf, uint p3_stride, uint p3_offset)\n"
		"{\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, height
		"    p0_buf += p0_offset;\n"
		"    p1_buf += p1_offset;\n"
		"    p2_buf += p2_offset;\n"
		"    p3_buf += p3_offset;\n"
		"    p0_buf += (gy * %d) + (gx << 4);\n" // stride0
		"    p1_buf += (gy * %d) + (gx << 3);\n" // stride1
		"    p2_buf += (gy * %d) + (gx << 2);\n" // stride2
		"    p3_buf += (gy * %d) + (gx << 2);\n" // stride3
		"    uint2 pY = *(__global uint2 *) p1_buf;\n"
		"    uint  pU = *(__global uint  *) p2_buf;\n"
		"    uint  pV = *(__global uint  *) p3_buf;\n"
		"    uint4 out;\n"
		"%s"
		"    *(__global uint4 *) p0_buf = out;\n"
		"  }\n"
		"}\n")
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, (width + 7) / 8, height, stride0, stride1, stride2, stride3, combineCode);
	node->opencl_code = item;

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_work_dim = 2;
	node->opencl_global_work[0] = (((width + 7) >> 3) + work_group_width - 1) & ~(work_group_width - 1);
	node->opencl_global_work[1] = (height + work_group_height - 1) & ~(work_group_height - 1);
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;

	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following channel extractions:
//   VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS0
//   VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS1
//   VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS2
//   VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS3
//
int HafGpu_ChannelExtract_U8_U32(AgoNode * node)
{
	int status = VX_SUCCESS;

	// configuration
	vx_enum kernel = node->akernel->id;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;
	int stride0 = node->paramList[0]->u.img.stride_in_bytes;
	int stride1 = node->paramList[1]->u.img.stride_in_bytes;

	int work_group_width = 16;
	int work_group_height = 4;

	char extractionCode[1024];
	if (kernel == VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS0)	{
		snprintf(extractionCode, sizeof(extractionCode),
			"	r.s0 = amd_pack((float4)(amd_unpack0(L.s0), amd_unpack0(L.s1), amd_unpack0(L.s2), amd_unpack0(L.s3)));\n"
			"	r.s1 = amd_pack((float4)(amd_unpack0(L.s4), amd_unpack0(L.s5), amd_unpack0(L.s6), amd_unpack0(L.s7)));\n"
			);
	}
	else if (kernel == VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS1)	{
		snprintf(extractionCode, sizeof(extractionCode),
			"	r.s0 = amd_pack((float4)(amd_unpack1(L.s0), amd_unpack1(L.s1), amd_unpack1(L.s2), amd_unpack1(L.s3)));\n"
			"	r.s1 = amd_pack((float4)(amd_unpack1(L.s4), amd_unpack1(L.s5), amd_unpack1(L.s6), amd_unpack1(L.s7)));\n"
			);
	}
	else if (kernel == VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS2)	{
		snprintf(extractionCode, sizeof(extractionCode),
			"	r.s0 = amd_pack((float4)(amd_unpack2(L.s0), amd_unpack2(L.s1), amd_unpack2(L.s2), amd_unpack2(L.s3)));\n"
			"	r.s1 = amd_pack((float4)(amd_unpack2(L.s4), amd_unpack2(L.s5), amd_unpack2(L.s6), amd_unpack2(L.s7)));\n"
			);
	}
	else if (kernel == VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS3)	{
		snprintf(extractionCode, sizeof(extractionCode),
			"	r.s0 = amd_pack((float4)(amd_unpack3(L.s0), amd_unpack3(L.s1), amd_unpack3(L.s2), amd_unpack3(L.s3)));\n"
			"	r.s1 = amd_pack((float4)(amd_unpack3(L.s4), amd_unpack3(L.s5), amd_unpack3(L.s6), amd_unpack3(L.s7)));\n"
			);
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_ChannelExtract_U8_U32 doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}

	// kernel body
	char item[8192];
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset,\n"
		"        uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset)\n"
		"{\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  if ((gx < %d) && (gy < %d)) {\n" // (width+3)/4, height
		"    p0_buf += p0_offset;\n"
		"    p1_buf += p1_offset;\n"
		"    p0_buf += (gy * %d) + (gx << 2);\n" // stride0
		"    p1_buf += (gy * %d) + (gx << 4);\n" // stride1
		"    uint8 L = *(__global uint8 *) p1_buf;\n"
		"	 uint2 r;\n"
		"%s"
		"	 *(__global uint2 *) p0_buf = r;\n"
		"  }\n"
		"}\n")
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, (width + 3) / 4, height, stride0, stride1, extractionCode
		);
	node->opencl_code = item;

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_work_dim = 2;
	node->opencl_global_work[0] = (((width + 3) >> 2) + work_group_width - 1) & ~(work_group_width - 1);
	node->opencl_global_work[1] = (height + work_group_height - 1) & ~(work_group_height - 1);
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;
	
	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following format conversions:
//   VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_UYVY
//   VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_YUYV
//   VX_KERNEL_AMD_FORMAT_CONVERT_NV12_UYVY
//   VX_KERNEL_AMD_FORMAT_CONVERT_NV12_YUYV
//
int HafGpu_FormatConvert_420_422(AgoNode * node)
{
	int status = VX_SUCCESS;

	// configuration
	vx_enum kernel = node->akernel->id;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;
	int stride0 = node->paramList[0]->u.img.stride_in_bytes;
	int stride1 = node->paramList[1]->u.img.stride_in_bytes;
	int stride2 = node->paramList[2]->u.img.stride_in_bytes;
	int stride3 = node->paramList[3] ? node->paramList[3]->u.img.stride_in_bytes : 0;
	int work_group_width = 16;
	int work_group_height = 4;

	char conversionCode[1024];
	if (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_YUYV) {
		snprintf(conversionCode, sizeof(conversionCode),
			OPENCL_FORMAT(
			"    pY0.s0 = amd_pack((float4)(amd_unpack0(L0.s0), amd_unpack2(L0.s0), amd_unpack0(L0.s1), amd_unpack2(L0.s1)));\n"
			"    pY0.s1 = amd_pack((float4)(amd_unpack0(L0.s2), amd_unpack2(L0.s2), amd_unpack0(L0.s3), amd_unpack2(L0.s3)));\n"
			"    pY1.s0 = amd_pack((float4)(amd_unpack0(L1.s0), amd_unpack2(L1.s0), amd_unpack0(L1.s1), amd_unpack2(L1.s1)));\n"
			"    pY1.s1 = amd_pack((float4)(amd_unpack0(L1.s2), amd_unpack2(L1.s2), amd_unpack0(L1.s3), amd_unpack2(L1.s3)));\n"
			"    L0.s0  = amd_lerp(L0.s0, L1.s0, 0);\n"
			"    L0.s1  = amd_lerp(L0.s1, L1.s1, 0);\n"
			"    L0.s2  = amd_lerp(L0.s2, L1.s2, 0);\n"
			"    L0.s3  = amd_lerp(L0.s3, L1.s3, 0);\n"
			"    pU     = amd_pack((float4)(amd_unpack1(L0.s0), amd_unpack1(L0.s1), amd_unpack1(L0.s2), amd_unpack1(L0.s3)));\n"
			"    pV     = amd_pack((float4)(amd_unpack3(L0.s0), amd_unpack3(L0.s1), amd_unpack3(L0.s2), amd_unpack3(L0.s3)));\n"
			));
	}
	else if (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_UYVY) {
		snprintf(conversionCode, sizeof(conversionCode),
			OPENCL_FORMAT(
			"    pY0.s0 = amd_pack((float4)(amd_unpack1(L0.s0), amd_unpack3(L0.s0), amd_unpack1(L0.s1), amd_unpack3(L0.s1)));\n"
			"    pY0.s1 = amd_pack((float4)(amd_unpack1(L0.s2), amd_unpack3(L0.s2), amd_unpack1(L0.s3), amd_unpack3(L0.s3)));\n"
			"    pY1.s0 = amd_pack((float4)(amd_unpack1(L1.s0), amd_unpack3(L1.s0), amd_unpack1(L1.s1), amd_unpack3(L1.s1)));\n"
			"    pY1.s1 = amd_pack((float4)(amd_unpack1(L1.s2), amd_unpack3(L1.s2), amd_unpack1(L1.s3), amd_unpack3(L1.s3)));\n"
			"    L0.s0  = amd_lerp(L0.s0, L1.s0, 0);\n"
			"    L0.s1  = amd_lerp(L0.s1, L1.s1, 0);\n"
			"    L0.s2  = amd_lerp(L0.s2, L1.s2, 0);\n"
			"    L0.s3  = amd_lerp(L0.s3, L1.s3, 0);\n"
			"    pU     = amd_pack((float4)(amd_unpack0(L0.s0), amd_unpack0(L0.s1), amd_unpack0(L0.s2), amd_unpack0(L0.s3)));\n"
			"    pV     = amd_pack((float4)(amd_unpack2(L0.s0), amd_unpack2(L0.s1), amd_unpack2(L0.s2), amd_unpack2(L0.s3)));\n"
			));
	}
	else if (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_NV12_YUYV) {
		snprintf(conversionCode, sizeof(conversionCode),
			OPENCL_FORMAT(
			"    pY0.s0 = amd_pack((float4)(amd_unpack0(L0.s0), amd_unpack2(L0.s0), amd_unpack0(L0.s1), amd_unpack2(L0.s1)));\n"
			"    pY0.s1 = amd_pack((float4)(amd_unpack0(L0.s2), amd_unpack2(L0.s2), amd_unpack0(L0.s3), amd_unpack2(L0.s3)));\n"
			"    pY1.s0 = amd_pack((float4)(amd_unpack0(L1.s0), amd_unpack2(L1.s0), amd_unpack0(L1.s1), amd_unpack2(L1.s1)));\n"
			"    pY1.s1 = amd_pack((float4)(amd_unpack0(L1.s2), amd_unpack2(L1.s2), amd_unpack0(L1.s3), amd_unpack2(L1.s3)));\n"
			"    L0.s0  = amd_lerp(L0.s0, L1.s0, 0);\n"
			"    L0.s1  = amd_lerp(L0.s1, L1.s1, 0);\n"
			"    L0.s2  = amd_lerp(L0.s2, L1.s2, 0);\n"
			"    L0.s3  = amd_lerp(L0.s3, L1.s3, 0);\n"
			"    pUV.s0 = amd_pack((float4)(amd_unpack1(L0.s0), amd_unpack3(L0.s0), amd_unpack1(L0.s1), amd_unpack3(L0.s1)));\n"
			"    pUV.s1 = amd_pack((float4)(amd_unpack1(L0.s2), amd_unpack3(L0.s2), amd_unpack1(L0.s3), amd_unpack3(L0.s3)));\n"
			));
	}
	else if (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_NV12_UYVY) {
		snprintf(conversionCode, sizeof(conversionCode),
			OPENCL_FORMAT(
			"    pY0.s0 = amd_pack((float4)(amd_unpack1(L0.s0), amd_unpack3(L0.s0), amd_unpack1(L0.s1), amd_unpack3(L0.s1)));\n"
			"    pY0.s1 = amd_pack((float4)(amd_unpack1(L0.s2), amd_unpack3(L0.s2), amd_unpack1(L0.s3), amd_unpack3(L0.s3)));\n"
			"    pY1.s0 = amd_pack((float4)(amd_unpack1(L1.s0), amd_unpack3(L1.s0), amd_unpack1(L1.s1), amd_unpack3(L1.s1)));\n"
			"    pY1.s1 = amd_pack((float4)(amd_unpack1(L1.s2), amd_unpack3(L1.s2), amd_unpack1(L1.s3), amd_unpack3(L1.s3)));\n"
			"    L0.s0  = amd_lerp(L0.s0, L1.s0, 0);\n"
			"    L0.s1  = amd_lerp(L0.s1, L1.s1, 0);\n"
			"    L0.s2  = amd_lerp(L0.s2, L1.s2, 0);\n"
			"    L0.s3  = amd_lerp(L0.s3, L1.s3, 0);\n"
			"    pUV.s0 = amd_pack((float4)(amd_unpack0(L0.s0), amd_unpack2(L0.s0), amd_unpack0(L0.s1), amd_unpack2(L0.s1)));\n"
			"    pUV.s1 = amd_pack((float4)(amd_unpack0(L0.s2), amd_unpack2(L0.s2), amd_unpack0(L0.s3), amd_unpack2(L0.s3)));\n"
			));
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_FormatConvert_420_422 doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}

	// kernel declaration
	char item[8192];
	if ((kernel == VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_YUYV) || (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_UYVY)) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset,\n"
			"        uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset,\n"
			"        uint p2_width, uint p2_height, __global uchar * p2_buf, uint p2_stride, uint p2_offset,\n"
			"        uint p3_width, uint p3_height, __global uchar * p3_buf, uint p3_stride, uint p3_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    p0_buf += p0_offset;\n"
			"    p1_buf += p1_offset;\n"
			"    p2_buf += p2_offset;\n"
			"    p3_buf += p3_offset;\n"
			"    p0_buf += (gy * %d) + (gx << 3);\n" // stride0 * 2
			"    p1_buf += (gy * %d) + (gx << 2);\n" // stride1
			"    p2_buf += (gy * %d) + (gx << 2);\n" // stride2
			"    p3_buf += (gy * %d) + (gx << 4);\n" // stride3 * 2
			"    uint4 L0 = *(__global uint4 *) p3_buf;\n"
			"    uint4 L1 = *(__global uint4 *)&p3_buf[%d];\n" // stride3
			"    uint2 pY0, pY1; uint pU, pV;\n"
			"%s"
			"    *(__global uint2 *) p0_buf = pY0;\n"
			"    *(__global uint2 *)&p0_buf[%d] = pY1;\n" // stride0
			"    *(__global uint  *) p1_buf = pU;\n"
			"    *(__global uint  *) p2_buf = pV;\n"
			"  }\n"
			"}\n")
			, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, (width + 7) / 8, (height + 1) / 2, stride0 * 2, stride1, stride2, stride3 * 2, stride3, conversionCode, stride0);
		node->opencl_code = item;
	}
	else if ((kernel == VX_KERNEL_AMD_FORMAT_CONVERT_NV12_YUYV) || (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_NV12_UYVY)) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset,\n"
			"        uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset,\n"
			"        uint p2_width, uint p2_height, __global uchar * p2_buf, uint p2_stride, uint p2_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    p0_buf += p0_offset;\n"
			"    p1_buf += p1_offset;\n"
			"    p2_buf += p2_offset;\n"
			"    p0_buf += (gy * %d) + (gx << 3);\n" // stride0 * 2
			"    p1_buf += (gy * %d) + (gx << 3);\n" // stride1
			"    p2_buf += (gy * %d) + (gx << 4);\n" // stride2 * 2
			"    uint4 L0 = *(__global uint4 *) p2_buf;\n"
			"    uint4 L1 = *(__global uint4 *)&p2_buf[%d];\n" // stride2
			"    uint2 pY0, pY1, pUV;\n"
			"%s"
			"    *(__global uint2 *) p0_buf = pY0;\n"
			"    *(__global uint2 *)&p0_buf[%d] = pY1;\n" // stride0
			"    *(__global uint2 *) p1_buf = pUV;\n"
			"  }\n"
			"}\n")
			, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, (width + 7) / 8, (height + 1) / 2, stride0 * 2, stride1, stride2 * 2, stride2, conversionCode, stride0);
		node->opencl_code = item;
	}

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_work_dim = 2;
	node->opencl_global_work[0] = (((width + 7) >> 3) + work_group_width - 1) & ~(work_group_width - 1);
	node->opencl_global_work[1] = (((height + 1) >> 1) + work_group_height - 1) & ~(work_group_height - 1);
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;

	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following format conversions:
//   VX_KERNEL_AMD_FORMAT_CONVERT_UV_UV12
//   VX_KERNEL_AMD_FORMAT_CONVERT_IUV_UV12
//   VX_KERNEL_AMD_FORMAT_CONVERT_UV12_IUV
//   VX_KERNEL_AMD_SCALE_UP_2x2_U8_U8
//
int HafGpu_FormatConvert_Chroma(AgoNode * node)
{
	int status = VX_SUCCESS;

	// configuration
	vx_enum kernel = node->akernel->id;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;
	int stride0 = node->paramList[0]->u.img.stride_in_bytes;
	int stride1 = node->paramList[1]->u.img.stride_in_bytes;
	int stride2 = node->paramList[2] ? node->paramList[2]->u.img.stride_in_bytes : 0;
	int work_group_width = 16;
	int work_group_height = 4;

	// kernel declaration
	char item[8192];
	if (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_UV_UV12) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset,\n"
			"        uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset,\n"
			"        uint p2_width, uint p2_height, __global uchar * p2_buf, uint p2_stride, uint p2_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    p0_buf += p0_offset;\n"
			"    p1_buf += p1_offset;\n"
			"    p2_buf += p2_offset;\n"
			"    p0_buf += (gy * %d) + (gx << 3);\n" // stride0 * 2
			"    p1_buf += (gy * %d) + (gx << 3);\n" // stride1 * 2
			"    p2_buf += (gy * %d) + (gx << 3);\n" // stride2
			"    uint2 L0 = *(__global uint2 *) p2_buf;\n"
			"    uint2 pU, pV;\n"
			"    pU.s0 = amd_pack((float4)(amd_unpack0(L0.s0), amd_unpack0(L0.s0), amd_unpack2(L0.s0), amd_unpack2(L0.s0)));\n"
			"    pU.s1 = amd_pack((float4)(amd_unpack0(L0.s1), amd_unpack0(L0.s1), amd_unpack2(L0.s1), amd_unpack2(L0.s1)));\n"
			"    pV.s0 = amd_pack((float4)(amd_unpack1(L0.s0), amd_unpack1(L0.s0), amd_unpack3(L0.s0), amd_unpack3(L0.s0)));\n"
			"    pV.s1 = amd_pack((float4)(amd_unpack1(L0.s1), amd_unpack1(L0.s1), amd_unpack3(L0.s1), amd_unpack3(L0.s1)));\n"
			"    *(__global uint2 *) p0_buf = pU;\n"
			"    *(__global uint2 *)&p0_buf[%d] = pU;\n" // stride0
			"    *(__global uint2 *) p1_buf = pV;\n"
			"    *(__global uint2 *)&p1_buf[%d] = pV;\n" // stride1
			"  }\n"
			"}\n")
			, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, (width + 7) / 8, (height + 1) / 2, stride0 * 2, stride1 * 2, stride2, stride0, stride1);
		node->opencl_code = item;
	}
	else if (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_IUV_UV12) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset,\n"
			"        uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset,\n"
			"        uint p2_width, uint p2_height, __global uchar * p2_buf, uint p2_stride, uint p2_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    p0_buf += p0_offset;\n"
			"    p1_buf += p1_offset;\n"
			"    p2_buf += p2_offset;\n"
			"    p0_buf += (gy * %d) + (gx << 3);\n" // stride0 * 2
			"    p1_buf += (gy * %d) + (gx << 3);\n" // stride1 * 2
			"    p2_buf += (gy * %d) + (gx << 4);\n" // stride2 * 2
			"    uint4 L0, L1;\n"
			"    L0 = *(__global uint4 *) p2_buf;\n"
			"    L1 = *(__global uint4 *) &p2_buf[%d];\n"	// stride2
			"    uint2 pU0, pV0, pU1, pV1;\n"
			"    pU0.s0 = amd_pack((float4)(amd_unpack0(L0.s0), amd_unpack2(L0.s0), amd_unpack0(L0.s1), amd_unpack2(L0.s1)));\n"
			"    pU0.s1 = amd_pack((float4)(amd_unpack0(L0.s2), amd_unpack2(L0.s2), amd_unpack0(L0.s3), amd_unpack2(L0.s3)));\n"
			"    pV0.s0 = amd_pack((float4)(amd_unpack1(L0.s0), amd_unpack3(L0.s0), amd_unpack1(L0.s1), amd_unpack3(L0.s1)));\n"
			"    pV0.s1 = amd_pack((float4)(amd_unpack1(L0.s2), amd_unpack3(L0.s2), amd_unpack1(L0.s3), amd_unpack3(L0.s3)));\n"
			"    pU1.s0 = amd_pack((float4)(amd_unpack0(L1.s0), amd_unpack2(L1.s0), amd_unpack0(L1.s1), amd_unpack2(L1.s1)));\n"
			"    pU1.s1 = amd_pack((float4)(amd_unpack0(L1.s2), amd_unpack2(L1.s2), amd_unpack0(L1.s3), amd_unpack2(L1.s3)));\n"
			"    pV1.s0 = amd_pack((float4)(amd_unpack1(L1.s0), amd_unpack3(L1.s0), amd_unpack1(L1.s1), amd_unpack3(L1.s1)));\n"
			"    pV1.s1 = amd_pack((float4)(amd_unpack1(L1.s2), amd_unpack3(L1.s2), amd_unpack1(L1.s3), amd_unpack3(L1.s3)));\n"
			"    *(__global uint2 *) p0_buf = pU0;\n"
			"    *(__global uint2 *)&p0_buf[%d] = pU1;\n" // stride0
			"    *(__global uint2 *) p1_buf = pV0;\n"
			"    *(__global uint2 *)&p1_buf[%d] = pV1;\n" // stride1
			"  }\n"
			"}\n")
			, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, (width + 7) / 8, (height + 1) / 2, stride0 * 2, stride1 * 2, stride2 * 2, stride2, stride0, stride1);
		node->opencl_code = item;
	}
	else if (kernel == VX_KERNEL_AMD_FORMAT_CONVERT_UV12_IUV) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset,\n"
			"        uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset,\n"
			"        uint p2_width, uint p2_height, __global uchar * p2_buf, uint p2_stride, uint p2_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    p0_buf += p0_offset;\n"
			"    p1_buf += p1_offset;\n"
			"    p2_buf += p2_offset;\n"
			"    p0_buf += (gy * %d) + (gx << 4);\n" // stride0 * 2
			"    p1_buf += (gy * %d) + (gx << 3);\n" // stride1 * 2
			"    p2_buf += (gy * %d) + (gx << 3);\n" // stride2 * 2
			"    uint2 pU0 = *(__global uint2 *) p1_buf;\n"
			"    uint2 pU1 = *(__global uint2 *)&p1_buf[%d];\n" // stride1
			"    uint2 pV0 = *(__global uint2 *) p2_buf;\n"
			"    uint2 pV1 = *(__global uint2 *)&p2_buf[%d];\n" // stride2
			"    uint4 L0, L1;\n"
			"    L0.s0 = amd_pack((float4)(amd_unpack0(pU0.s0), amd_unpack0(pV0.s0), amd_unpack1(pU0.s0), amd_unpack1(pV0.s0)));\n"
			"    L0.s1 = amd_pack((float4)(amd_unpack2(pU0.s0), amd_unpack2(pV0.s0), amd_unpack3(pU0.s0), amd_unpack3(pV0.s0)));\n"
			"    L0.s2 = amd_pack((float4)(amd_unpack0(pU0.s1), amd_unpack0(pV0.s1), amd_unpack1(pU0.s1), amd_unpack1(pV0.s1)));\n"
			"    L0.s3 = amd_pack((float4)(amd_unpack2(pU0.s1), amd_unpack2(pV0.s1), amd_unpack3(pU0.s1), amd_unpack3(pV0.s1)));\n"
			"    L1.s0 = amd_pack((float4)(amd_unpack0(pU1.s0), amd_unpack0(pV1.s0), amd_unpack1(pU1.s0), amd_unpack1(pV1.s0)));\n"
			"    L1.s1 = amd_pack((float4)(amd_unpack2(pU1.s0), amd_unpack2(pV1.s0), amd_unpack3(pU1.s0), amd_unpack3(pV1.s0)));\n"
			"    L1.s2 = amd_pack((float4)(amd_unpack0(pU1.s1), amd_unpack0(pV1.s1), amd_unpack1(pU1.s1), amd_unpack1(pV1.s1)));\n"
			"    L1.s3 = amd_pack((float4)(amd_unpack2(pU1.s1), amd_unpack2(pV1.s1), amd_unpack3(pU1.s1), amd_unpack3(pV1.s1)));\n"
			"    *(__global uint4 *) p0_buf = L0;\n"
			"    *(__global uint4 *)&p0_buf[%d] = L1;\n" // stride0
			"  }\n"
			"}\n")
			, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, (width + 7)/ 8, (height + 1)/ 2, stride0 * 2, stride1 * 2, stride2 * 2, stride1, stride2, stride0);
		node->opencl_code = item;
	}
	else if (kernel == VX_KERNEL_AMD_SCALE_UP_2x2_U8_U8) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
			"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
			"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset,\n"
			"        uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    p0_buf += p0_offset;\n"
			"    p1_buf += p1_offset;\n"
			"    p0_buf += (gy * %d) + (gx << 3);\n" // stride0 * 2
			"    p1_buf += (gy * %d) + (gx << 2);\n" // stride1
			"    uint L0 = *(__global uint *) p1_buf;\n"
			"    uint2 X2;\n"
			"    X2.s0 = amd_pack((float4)(amd_unpack0(L0), amd_unpack0(L0), amd_unpack1(L0), amd_unpack1(L0)));\n"
			"    X2.s1 = amd_pack((float4)(amd_unpack2(L0), amd_unpack2(L0), amd_unpack3(L0), amd_unpack3(L0)));\n"
			"    *(__global uint2 *) p0_buf = X2;\n"
			"    *(__global uint2 *)&p0_buf[%d] = X2;\n" // stride0
			"  }\n"
			"}\n")
			, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, (width + 7) / 8, (height + 1) / 2, stride0 * 2, stride1, stride0);
		node->opencl_code = item;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_FormatConvert_Chroma doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_work_dim = 2;
	node->opencl_global_work[0] = (((width + 7) >> 3) + work_group_width - 1) & ~(work_group_width - 1);
	node->opencl_global_work[1] = (((height + 1) >> 1) + work_group_height - 1) & ~(work_group_height - 1);
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;
	
	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following color conversions:
//   VX_KERNEL_AMD_COLOR_CONVERT_IU_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_IU_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_IV_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_IV_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGB
//   VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGBX
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_IYUV
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV12
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV21
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_UYVY
//   VX_KERNEL_AMD_COLOR_CONVERT_RGB_YUYV
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_IYUV
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV12
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV21
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_UYVY
//   VX_KERNEL_AMD_COLOR_CONVERT_RGBX_YUYV
//
int HafGpu_ColorConvert(AgoNode * node)
{
	int status = VX_SUCCESS;

	// configuration
	vx_enum kernel = node->akernel->id;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;
	vx_color_space_e input_color_space = node->paramList[1]->u.img.color_space;
	vx_color_space_e output_color_space = node->paramList[0]->u.img.color_space;
	vx_channel_range_e input_channel_range = node->paramList[1]->u.img.channel_range;
	vx_channel_range_e output_channel_range = node->paramList[0]->u.img.channel_range;
	int pRGB_stride = 0, p422_stride = 0, pY_stride = 0, pU_stride = 0, pV_stride = 0, pUV_stride = 0;
	int work_group_width = 16;
	int work_group_height = 4;
	bool isSourceRGB =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IU_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IV_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGB;
	bool isSourceRGBX =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IU_RGBX ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGBX ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IV_RGBX ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGBX ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGBX ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGBX;
	bool isSourceUYVY =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_UYVY ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_UYVY;
	bool isSourceYUYV =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_YUYV ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_YUYV;
	bool isSourceIYUV =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_IYUV ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_IYUV;
	bool isSourceNV12 =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV12 ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV12;
	bool isSourceNV21 =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV21 ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV21;
	bool isDestinationRGB =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_IYUV ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV12 ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV21 ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_UYVY ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGB_YUYV;
	bool isDestinationRGBX =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_IYUV ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV12 ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV21 ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_UYVY ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_RGBX_YUYV;
	bool destinationHasY =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGBX ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGBX;
	bool destinationHasUV12 =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGBX ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGBX;
	bool destinationNoU =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IV_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IV_RGBX;
	bool destinationNoV =
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IU_RGB ||
		kernel == VX_KERNEL_AMD_COLOR_CONVERT_IU_RGBX;

	// kernel header and reading
	char item[8192];
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"typedef uint2   U8x8;\n"
		"typedef uint8  U24x8;\n"
		"typedef uint8  U32x8;\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(")
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME);
	node->opencl_code = item;
	int argCount = 0;
	if (isDestinationRGB) {
		node->opencl_code += "uint pRGB_width, uint pRGB_height, __global uchar * pRGB_buf, uint pRGB_stride, uint pRGB_offset,\n";
		pRGB_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
	}
	else if (isDestinationRGBX) {
		node->opencl_code += "uint pRGB_width, uint pRGB_height, __global uchar * pRGB_buf, uint pRGB_stride, uint pRGB_offset,\n";
		pRGB_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
	}
	else {
		if (destinationHasY) {
			node->opencl_code += "uint pY_width, uint pY_height, __global uchar * pY_buf, uint pY_stride, uint pY_offset,\n    ";
			pY_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		}
		if (destinationHasUV12) {
			node->opencl_code += "uint pUV_width, uint pUV_height, __global uchar * pUV_buf, uint pUV_stride, uint pUV_offset,\n    ";
			pUV_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		}
		else {
			if (!destinationNoU) {
				node->opencl_code += "uint pU_width, uint pU_height, __global uchar * pU_buf, uint pU_stride, uint pU_offset,\n    ";
				pU_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
			}
			if (!destinationNoV) {
				node->opencl_code += "uint pV_width, uint pV_height, __global uchar * pV_buf, uint pV_stride, uint pV_offset,\n    ";
				pV_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
			}
		}
	}
	if (isSourceRGB) {
		pRGB_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"uint pRGB_width, uint pRGB_height, __global uchar * pRGB_buf, uint pRGB_stride, uint pRGB_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    pRGB_buf += pRGB_offset + (gy * %d) + (gx * 24);\n" // pRGB_stride * 2
			"    U24x8 pRGB0, pRGB1;\n"
			"    pRGB0.s0123 = *(__global uint4 *) pRGB_buf;\n"
			"    pRGB0.s45 = *(__global uint2 *)&pRGB_buf[16];\n"
			"    pRGB1.s0123 = *(__global uint4 *)&pRGB_buf[%d];\n" // pRGB_stride
			"    pRGB1.s45 = *(__global uint2 *)&pRGB_buf[%d+16];\n" // pRGB_stride
			), (width + 7) / 8, (height + 1) / 2, pRGB_stride * 2, pRGB_stride, pRGB_stride);
		node->opencl_code += item;
	}
	else if (isSourceRGBX) {
		pRGB_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"uint pRGB_width, uint pRGB_height, __global uchar * pRGB_buf, uint pRGB_stride, uint pRGB_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    pRGB_buf += pRGB_offset + (gy * %d) + (gx << 5);\n" // pRGB_stride * 2
			"    U32x8 pRGBX0, pRGBX1;\n"
			"    pRGBX0 = *(__global U32x8 *) pRGB_buf;\n"
			"    pRGBX1 = *(__global U32x8 *)&pRGB_buf[%d];\n" // pRGB_stride
			), (width + 7) / 8, (height + 1) / 2, pRGB_stride * 2, pRGB_stride);
		node->opencl_code += item;
	}
	else if (isSourceUYVY || isSourceYUYV) {
		p422_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"uint p422_width, uint p422_height, __global uchar * p422_buf, uint p422_stride, uint p422_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    p422_buf += p422_offset + (gy * %d) + (gx << 4);\n" // p422_stride * 2
			"    uint4 L0, L1;\n"
			"    L0 = *(__global uint4 *) p422_buf;\n"
			"    L1 = *(__global uint4 *)&p422_buf[%d];\n" // p422_stride
			), (width + 7) / 8, (height + 1) / 2, p422_stride * 2, p422_stride);
		node->opencl_code += item;
	}
	else if (isSourceIYUV) {
		pY_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		pU_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		pV_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"uint pY_width, uint pY_height, __global uchar * pY_buf, uint pY_stride, uint pY_offset,\n    "
			"uint pU_width, uint pU_height, __global uchar * pU_buf, uint pU_stride, uint pU_offset,\n    "
			"uint pV_width, uint pV_height, __global uchar * pV_buf, uint pV_stride, uint pV_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    pY_buf += pY_offset + (gy * %d) + (gx << 3);\n" // pY_stride * 2
			"    pU_buf += pU_offset + (gy * %d) + (gx << 2);\n" // pU_stride
			"    pV_buf += pV_offset + (gy * %d) + (gx << 2);\n" // pV_stride
			"    U8x8 pY0, pY1, pUV;\n"
			"    pY0 = *(__global U8x8 *) pY_buf;\n"
			"    pY1 = *(__global U8x8 *)&pY_buf[%d];\n" // pY_stride
			"    pUV.s0 = *(__global uint *) pU_buf;\n"
			"    pUV.s1 = *(__global uint *) pV_buf;\n"
			), (width + 7) / 8, (height + 1) / 2, pY_stride * 2, pU_stride, pV_stride, pY_stride);
		node->opencl_code += item;
	}
	else {
		pY_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		pUV_stride = node->paramList[argCount++]->u.img.stride_in_bytes;
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"uint pY_width, uint pY_height, __global uchar * pY_buf, uint pY_stride, uint pY_offset,\n    "
			"uint pUV_width, uint pUV_height, __global uchar * pUV_buf, uint pUV_stride, uint pUV_offset)\n"
			"{\n"
			"  int gx = get_global_id(0);\n"
			"  int gy = get_global_id(1);\n"
			"  if ((gx < %d) && (gy < %d)) {\n" // (width+7)/8, (height+1)/2
			"    pY_buf += pY_offset + (gy * %d) + (gx << 3);\n" // pY_stride * 2
			"    pUV_buf += pUV_offset + (gy * %d) + (gx << 3);\n" // pUV_stride
			"    U8x8 pY0, pY1, pUV;\n"
			"    pY0 = *(__global U8x8 *) pY_buf;\n"
			"    pY1 = *(__global U8x8 *)&pY_buf[%d];\n" // pY_stride
			"    pUV = *(__global U8x8 *) pUV_buf;\n"
			), (width + 7) / 8, (height + 1) / 2, pY_stride * 2, pUV_stride, pY_stride);
		node->opencl_code += item;
	}

	// color conversion part
	node->opencl_code +=
		"    float4 f;\n";
	if (isSourceRGB || isSourceRGBX) {
		if (isSourceRGB) {
			if (destinationHasY) {
				if (output_color_space == VX_COLOR_SPACE_BT601_525 || output_color_space == VX_COLOR_SPACE_BT601_625) {
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float4 cY = (float4)(0.2990f*219f/256f, 0.5870f*219f/256f, 0.1140f*219f/256f, 16.0f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float4 cY = (float4)(0.2990f, 0.5870f, 0.1140f, 0.0f);\n"
							);
					}
				}
				else { // VX_COLOR_SPACE_BT709
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float4 cY = (float4)(0.2126f*219f/256f, 0.7152f*219f/256f, 0.0722f*219f/256f, 16.0f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float4 cY = (float4)(0.2126f, 0.7152f, 0.0722f, 0.0f);\n"
							);
					}
				}
				node->opencl_code +=
					OPENCL_FORMAT(
					"    U8x8 pY0, pY1;\n"
					"    f.s0 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGB0.s0), amd_unpack1(pRGB0.s0), amd_unpack2(pRGB0.s0)));\n"
					"    f.s1 = cY.s3 + dot(cY.s012, (float3)(amd_unpack3(pRGB0.s0), amd_unpack0(pRGB0.s1), amd_unpack1(pRGB0.s1)));\n"
					"    f.s2 = cY.s3 + dot(cY.s012, (float3)(amd_unpack2(pRGB0.s1), amd_unpack3(pRGB0.s1), amd_unpack0(pRGB0.s2)));\n"
					"    f.s3 = cY.s3 + dot(cY.s012, (float3)(amd_unpack1(pRGB0.s2), amd_unpack2(pRGB0.s2), amd_unpack3(pRGB0.s2)));\n"
					"    pY0.s0 = amd_pack(f);\n"
					"    f.s0 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGB0.s3), amd_unpack1(pRGB0.s3), amd_unpack2(pRGB0.s3)));\n"
					"    f.s1 = cY.s3 + dot(cY.s012, (float3)(amd_unpack3(pRGB0.s3), amd_unpack0(pRGB0.s4), amd_unpack1(pRGB0.s4)));\n"
					"    f.s2 = cY.s3 + dot(cY.s012, (float3)(amd_unpack2(pRGB0.s4), amd_unpack3(pRGB0.s4), amd_unpack0(pRGB0.s5)));\n"
					"    f.s3 = cY.s3 + dot(cY.s012, (float3)(amd_unpack1(pRGB0.s5), amd_unpack2(pRGB0.s5), amd_unpack3(pRGB0.s5)));\n"
					"    pY0.s1 = amd_pack(f);\n"
					"    f.s0 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGB1.s0), amd_unpack1(pRGB1.s0), amd_unpack2(pRGB1.s0)));\n"
					"    f.s1 = cY.s3 + dot(cY.s012, (float3)(amd_unpack3(pRGB1.s0), amd_unpack0(pRGB1.s1), amd_unpack1(pRGB1.s1)));\n"
					"    f.s2 = cY.s3 + dot(cY.s012, (float3)(amd_unpack2(pRGB1.s1), amd_unpack3(pRGB1.s1), amd_unpack0(pRGB1.s2)));\n"
					"    f.s3 = cY.s3 + dot(cY.s012, (float3)(amd_unpack1(pRGB1.s2), amd_unpack2(pRGB1.s2), amd_unpack3(pRGB1.s2)));\n"
					"    pY1.s0 = amd_pack(f);\n"
					"    f.s0 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGB1.s3), amd_unpack1(pRGB1.s3), amd_unpack2(pRGB1.s3)));\n"
					"    f.s1 = cY.s3 + dot(cY.s012, (float3)(amd_unpack3(pRGB1.s3), amd_unpack0(pRGB1.s4), amd_unpack1(pRGB1.s4)));\n"
					"    f.s2 = cY.s3 + dot(cY.s012, (float3)(amd_unpack2(pRGB1.s4), amd_unpack3(pRGB1.s4), amd_unpack0(pRGB1.s5)));\n"
					"    f.s3 = cY.s3 + dot(cY.s012, (float3)(amd_unpack1(pRGB1.s5), amd_unpack2(pRGB1.s5), amd_unpack3(pRGB1.s5)));\n"
					"    pY1.s1 = amd_pack(f);\n"
					);
			}
			if (!destinationNoU) {
				if (output_color_space == VX_COLOR_SPACE_BT601_525 || output_color_space == VX_COLOR_SPACE_BT601_625) {
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cU = (float3)(-0.1690f*224f/256f, -0.3310f*224f/256f, 0.5f*224f/256f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cU = (float3)(-0.1690f, -0.3310f, 0.5f);\n"
							);
					}
				}
				else { // VX_COLOR_SPACE_BT709
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cU = (float3)(-0.1146f*224f/256f, -0.3854f*224f/256f, 0.5f*224f/256f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cU = (float3)(-0.1146f, -0.3854f, 0.5f);\n"
							);
					}
				}
				node->opencl_code +=
					OPENCL_FORMAT(
					"    U8x8 pU0, pU1;\n"
					"    f.s0 = dot(cU, (float3)(amd_unpack0(pRGB0.s0), amd_unpack1(pRGB0.s0), amd_unpack2(pRGB0.s0)));\n"
					"    f.s1 = dot(cU, (float3)(amd_unpack2(pRGB0.s1), amd_unpack3(pRGB0.s1), amd_unpack0(pRGB0.s2)));\n"
					"    f.s2 = dot(cU, (float3)(amd_unpack0(pRGB0.s3), amd_unpack1(pRGB0.s3), amd_unpack2(pRGB0.s3)));\n"
					"    f.s3 = dot(cU, (float3)(amd_unpack2(pRGB0.s4), amd_unpack3(pRGB0.s4), amd_unpack0(pRGB0.s5)));\n"
					"    pU0.s0 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cU, (float3)(amd_unpack3(pRGB0.s0), amd_unpack0(pRGB0.s1), amd_unpack1(pRGB0.s1)));\n"
					"    f.s1 = dot(cU, (float3)(amd_unpack1(pRGB0.s2), amd_unpack2(pRGB0.s2), amd_unpack3(pRGB0.s2)));\n"
					"    f.s2 = dot(cU, (float3)(amd_unpack3(pRGB0.s3), amd_unpack0(pRGB0.s4), amd_unpack1(pRGB0.s4)));\n"
					"    f.s3 = dot(cU, (float3)(amd_unpack1(pRGB0.s5), amd_unpack2(pRGB0.s5), amd_unpack3(pRGB0.s5)));\n"
					"    pU0.s1 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cU, (float3)(amd_unpack0(pRGB1.s0), amd_unpack1(pRGB1.s0), amd_unpack2(pRGB1.s0)));\n"
					"    f.s1 = dot(cU, (float3)(amd_unpack2(pRGB1.s1), amd_unpack3(pRGB1.s1), amd_unpack0(pRGB1.s2)));\n"
					"    f.s2 = dot(cU, (float3)(amd_unpack0(pRGB1.s3), amd_unpack1(pRGB1.s3), amd_unpack2(pRGB1.s3)));\n"
					"    f.s3 = dot(cU, (float3)(amd_unpack2(pRGB1.s4), amd_unpack3(pRGB1.s4), amd_unpack0(pRGB1.s5)));\n"
					"    pU1.s0 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cU, (float3)(amd_unpack3(pRGB1.s0), amd_unpack0(pRGB1.s1), amd_unpack1(pRGB1.s1)));\n"
					"    f.s1 = dot(cU, (float3)(amd_unpack1(pRGB1.s2), amd_unpack2(pRGB1.s2), amd_unpack3(pRGB1.s2)));\n"
					"    f.s2 = dot(cU, (float3)(amd_unpack3(pRGB1.s3), amd_unpack0(pRGB1.s4), amd_unpack1(pRGB1.s4)));\n"
					"    f.s3 = dot(cU, (float3)(amd_unpack1(pRGB1.s5), amd_unpack2(pRGB1.s5), amd_unpack3(pRGB1.s5)));\n"
					"    pU1.s1 = amd_pack(f + (float4)(128));\n"
					"    pU0.s0 = amd_lerp(pU0.s0, pU0.s1, 0);\n"
					"    pU1.s0 = amd_lerp(pU1.s0, pU1.s1, 0);\n"
					"    pU0.s0 = amd_lerp(pU0.s0, pU1.s0, 0);\n"
					);
			}
			if (!destinationNoV) {
				if (output_color_space == VX_COLOR_SPACE_BT601_525 || output_color_space == VX_COLOR_SPACE_BT601_625) {
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cV = (float3)(0.5f*224f/256f, -0.4190f*224f/256f, -0.0810f*224f/256f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cV = (float3)(0.5f, -0.4190f, -0.0810f);\n"
							);
					}
				}
				else { // VX_COLOR_SPACE_BT709
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cV = (float3)(0.5f*224f/256f, -0.4542f*224f/256f, -0.0458f*224f/256f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cV = (float3)(0.5f, -0.4542f, -0.0458f);\n"
							);
					}
				}
				node->opencl_code +=
					OPENCL_FORMAT(
					"    U8x8 pV0, pV1;\n"
					"    f.s0 = dot(cV, (float3)(amd_unpack0(pRGB0.s0), amd_unpack1(pRGB0.s0), amd_unpack2(pRGB0.s0)));\n"
					"    f.s1 = dot(cV, (float3)(amd_unpack2(pRGB0.s1), amd_unpack3(pRGB0.s1), amd_unpack0(pRGB0.s2)));\n"
					"    f.s2 = dot(cV, (float3)(amd_unpack0(pRGB0.s3), amd_unpack1(pRGB0.s3), amd_unpack2(pRGB0.s3)));\n"
					"    f.s3 = dot(cV, (float3)(amd_unpack2(pRGB0.s4), amd_unpack3(pRGB0.s4), amd_unpack0(pRGB0.s5)));\n"
					"    pV0.s0 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cV, (float3)(amd_unpack3(pRGB0.s0), amd_unpack0(pRGB0.s1), amd_unpack1(pRGB0.s1)));\n"
					"    f.s1 = dot(cV, (float3)(amd_unpack1(pRGB0.s2), amd_unpack2(pRGB0.s2), amd_unpack3(pRGB0.s2)));\n"
					"    f.s2 = dot(cV, (float3)(amd_unpack3(pRGB0.s3), amd_unpack0(pRGB0.s4), amd_unpack1(pRGB0.s4)));\n"
					"    f.s3 = dot(cV, (float3)(amd_unpack1(pRGB0.s5), amd_unpack2(pRGB0.s5), amd_unpack3(pRGB0.s5)));\n"
					"    pV0.s1 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cV, (float3)(amd_unpack0(pRGB1.s0), amd_unpack1(pRGB1.s0), amd_unpack2(pRGB1.s0)));\n"
					"    f.s1 = dot(cV, (float3)(amd_unpack2(pRGB1.s1), amd_unpack3(pRGB1.s1), amd_unpack0(pRGB1.s2)));\n"
					"    f.s2 = dot(cV, (float3)(amd_unpack0(pRGB1.s3), amd_unpack1(pRGB1.s3), amd_unpack2(pRGB1.s3)));\n"
					"    f.s3 = dot(cV, (float3)(amd_unpack2(pRGB1.s4), amd_unpack3(pRGB1.s4), amd_unpack0(pRGB1.s5)));\n"
					"    pV1.s0 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cV, (float3)(amd_unpack3(pRGB1.s0), amd_unpack0(pRGB1.s1), amd_unpack1(pRGB1.s1)));\n"
					"    f.s1 = dot(cV, (float3)(amd_unpack1(pRGB1.s2), amd_unpack2(pRGB1.s2), amd_unpack3(pRGB1.s2)));\n"
					"    f.s2 = dot(cV, (float3)(amd_unpack3(pRGB1.s3), amd_unpack0(pRGB1.s4), amd_unpack1(pRGB1.s4)));\n"
					"    f.s3 = dot(cV, (float3)(amd_unpack1(pRGB1.s5), amd_unpack2(pRGB1.s5), amd_unpack3(pRGB1.s5)));\n"
					"    pV1.s1 = amd_pack(f + (float4)(128));\n"
					"    pV0.s0 = amd_lerp(pV0.s0, pV0.s1, 0x01010101u);\n"
					"    pV1.s0 = amd_lerp(pV1.s0, pV1.s1, 0x01010101u);\n"
					"    pV0.s0 = amd_lerp(pV0.s0, pV1.s0, 0x01010101u);\n"
					);
			}
		}
		else if (isSourceRGBX) {
			if (destinationHasY) {
				if (output_color_space == VX_COLOR_SPACE_BT601_525 || output_color_space == VX_COLOR_SPACE_BT601_625) {
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float4 cY = (float4)(0.2990f*219f/256f, 0.5870f*219f/256f, 0.1140f*219f/256f, 16.0f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float4 cY = (float4)(0.2990f, 0.5870f, 0.1140f, 0.0f);\n"
							);
					}
				}
				else { // VX_COLOR_SPACE_BT709
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float4 cY = (float4)(0.2126f*219f/256f, 0.7152f*219f/256f, 0.0722f*219f/256f, 16.0f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float4 cY = (float4)(0.2126f, 0.7152f, 0.0722f, 0.0f);\n"
							);
					}
				}
				node->opencl_code +=
					OPENCL_FORMAT(
					"    U8x8 pY0, pY1;\n"
					"    f.s0 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX0.s0), amd_unpack1(pRGBX0.s0), amd_unpack2(pRGBX0.s0)));\n"
					"    f.s1 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX0.s1), amd_unpack1(pRGBX0.s1), amd_unpack2(pRGBX0.s1)));\n"
					"    f.s2 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX0.s2), amd_unpack1(pRGBX0.s2), amd_unpack2(pRGBX0.s2)));\n"
					"    f.s3 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX0.s3), amd_unpack1(pRGBX0.s3), amd_unpack2(pRGBX0.s3)));\n"
					"    pY0.s0 = amd_pack(f);\n"
					"    f.s0 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX0.s4), amd_unpack1(pRGBX0.s4), amd_unpack2(pRGBX0.s4)));\n"
					"    f.s1 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX0.s5), amd_unpack1(pRGBX0.s5), amd_unpack2(pRGBX0.s5)));\n"
					"    f.s2 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX0.s6), amd_unpack1(pRGBX0.s6), amd_unpack2(pRGBX0.s6)));\n"
					"    f.s3 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX0.s7), amd_unpack1(pRGBX0.s7), amd_unpack2(pRGBX0.s7)));\n"
					"    pY0.s1 = amd_pack(f);\n"
					"    f.s0 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX1.s0), amd_unpack1(pRGBX1.s0), amd_unpack2(pRGBX1.s0)));\n"
					"    f.s1 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX1.s1), amd_unpack1(pRGBX1.s1), amd_unpack2(pRGBX1.s1)));\n"
					"    f.s2 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX1.s2), amd_unpack1(pRGBX1.s2), amd_unpack2(pRGBX1.s2)));\n"
					"    f.s3 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX1.s3), amd_unpack1(pRGBX1.s3), amd_unpack2(pRGBX1.s3)));\n"
					"    pY1.s0 = amd_pack(f);\n"
					"    f.s0 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX1.s4), amd_unpack1(pRGBX1.s4), amd_unpack2(pRGBX1.s4)));\n"
					"    f.s1 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX1.s5), amd_unpack1(pRGBX1.s5), amd_unpack2(pRGBX1.s5)));\n"
					"    f.s2 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX1.s6), amd_unpack1(pRGBX1.s6), amd_unpack2(pRGBX1.s6)));\n"
					"    f.s3 = cY.s3 + dot(cY.s012, (float3)(amd_unpack0(pRGBX1.s7), amd_unpack1(pRGBX1.s7), amd_unpack2(pRGBX1.s7)));\n"
					"    pY1.s1 = amd_pack(f);\n"
					);
			}
			if (!destinationNoU) {
				if (output_color_space == VX_COLOR_SPACE_BT601_525 || output_color_space == VX_COLOR_SPACE_BT601_625) {
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cU = (float3)(-0.1690f*224f/256f, -0.3310f*224f/256f, 0.5f*224f/256f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cU = (float3)(-0.1690f, -0.3310f, 0.5f);\n"
							);
					}
				}
				else { // VX_COLOR_SPACE_BT709
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cU = (float3)(-0.1146f*224f/256f, -0.3854f*224f/256f, 0.5f*224f/256f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cU = (float3)(-0.1146f, -0.3854f, 0.5f);\n"
							);
					}
				}
				node->opencl_code +=
					OPENCL_FORMAT(
					"    U8x8 pU0, pU1;\n"
					"    f.s0 = dot(cU, (float3)(amd_unpack0(pRGBX0.s0), amd_unpack1(pRGBX0.s0), amd_unpack2(pRGBX0.s0)));\n"
					"    f.s1 = dot(cU, (float3)(amd_unpack0(pRGBX0.s2), amd_unpack1(pRGBX0.s2), amd_unpack2(pRGBX0.s2)));\n"
					"    f.s2 = dot(cU, (float3)(amd_unpack0(pRGBX0.s4), amd_unpack1(pRGBX0.s4), amd_unpack2(pRGBX0.s4)));\n"
					"    f.s3 = dot(cU, (float3)(amd_unpack0(pRGBX0.s6), amd_unpack1(pRGBX0.s6), amd_unpack2(pRGBX0.s6)));\n"
					"    pU0.s0 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cU, (float3)(amd_unpack0(pRGBX0.s1), amd_unpack1(pRGBX0.s1), amd_unpack2(pRGBX0.s1)));\n"
					"    f.s1 = dot(cU, (float3)(amd_unpack0(pRGBX0.s3), amd_unpack1(pRGBX0.s3), amd_unpack2(pRGBX0.s3)));\n"
					"    f.s2 = dot(cU, (float3)(amd_unpack0(pRGBX0.s5), amd_unpack1(pRGBX0.s5), amd_unpack2(pRGBX0.s5)));\n"
					"    f.s3 = dot(cU, (float3)(amd_unpack0(pRGBX0.s7), amd_unpack1(pRGBX0.s7), amd_unpack2(pRGBX0.s7)));\n"
					"    pU0.s1 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cU, (float3)(amd_unpack0(pRGBX1.s0), amd_unpack1(pRGBX1.s0), amd_unpack2(pRGBX1.s0)));\n"
					"    f.s1 = dot(cU, (float3)(amd_unpack0(pRGBX1.s2), amd_unpack1(pRGBX1.s2), amd_unpack2(pRGBX1.s2)));\n"
					"    f.s2 = dot(cU, (float3)(amd_unpack0(pRGBX1.s4), amd_unpack1(pRGBX1.s4), amd_unpack2(pRGBX1.s4)));\n"
					"    f.s3 = dot(cU, (float3)(amd_unpack0(pRGBX1.s6), amd_unpack1(pRGBX1.s6), amd_unpack2(pRGBX1.s6)));\n"
					"    pU1.s0 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cU, (float3)(amd_unpack0(pRGBX1.s1), amd_unpack1(pRGBX1.s1), amd_unpack2(pRGBX1.s1)));\n"
					"    f.s1 = dot(cU, (float3)(amd_unpack0(pRGBX1.s3), amd_unpack1(pRGBX1.s3), amd_unpack2(pRGBX1.s3)));\n"
					"    f.s2 = dot(cU, (float3)(amd_unpack0(pRGBX1.s5), amd_unpack1(pRGBX1.s5), amd_unpack2(pRGBX1.s5)));\n"
					"    f.s3 = dot(cU, (float3)(amd_unpack0(pRGBX1.s7), amd_unpack1(pRGBX1.s7), amd_unpack2(pRGBX1.s7)));\n"
					"    pU1.s1 = amd_pack(f + (float4)(128));\n"
					"    pU0.s0 = amd_lerp(pU0.s0, pU0.s1, 0);\n"
					"    pU1.s0 = amd_lerp(pU1.s0, pU1.s1, 0);\n"
					"    pU0.s0 = amd_lerp(pU0.s0, pU1.s0, 0x01010101u);\n"
					);
			}
			if (!destinationNoV) {
				if (output_color_space == VX_COLOR_SPACE_BT601_525 || output_color_space == VX_COLOR_SPACE_BT601_625) {
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cV = (float3)(0.5f*224f/256f, -0.4190f*224f/256f, -0.0810f*224f/256f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cV = (float3)(0.5f, -0.4190f, -0.0810f);\n"
							);
					}
				}
				else { // VX_COLOR_SPACE_BT709
					if (output_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cV = (float3)(0.5f*224f/256f, -0.4542f*224f/256f, -0.0458f*224f/256f);\n"
							);
					}
					else { // VX_CHANNEL_RANGE_FULL
						node->opencl_code +=
							OPENCL_FORMAT(
							"    float3 cV = (float3)(0.5f, -0.4542f, -0.0458f);\n"
							);
					}
				}
				node->opencl_code +=
					OPENCL_FORMAT(
					"    U8x8 pV0, pV1;\n"
					"    f.s0 = dot(cV, (float3)(amd_unpack0(pRGBX0.s0), amd_unpack1(pRGBX0.s0), amd_unpack2(pRGBX0.s0)));\n"
					"    f.s1 = dot(cV, (float3)(amd_unpack0(pRGBX0.s2), amd_unpack1(pRGBX0.s2), amd_unpack2(pRGBX0.s2)));\n"
					"    f.s2 = dot(cV, (float3)(amd_unpack0(pRGBX0.s4), amd_unpack1(pRGBX0.s4), amd_unpack2(pRGBX0.s4)));\n"
					"    f.s3 = dot(cV, (float3)(amd_unpack0(pRGBX0.s6), amd_unpack1(pRGBX0.s6), amd_unpack2(pRGBX0.s6)));\n"
					"    pV0.s0 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cV, (float3)(amd_unpack0(pRGBX0.s1), amd_unpack1(pRGBX0.s1), amd_unpack2(pRGBX0.s1)));\n"
					"    f.s1 = dot(cV, (float3)(amd_unpack0(pRGBX0.s3), amd_unpack1(pRGBX0.s3), amd_unpack2(pRGBX0.s3)));\n"
					"    f.s2 = dot(cV, (float3)(amd_unpack0(pRGBX0.s5), amd_unpack1(pRGBX0.s5), amd_unpack2(pRGBX0.s5)));\n"
					"    f.s3 = dot(cV, (float3)(amd_unpack0(pRGBX0.s7), amd_unpack1(pRGBX0.s7), amd_unpack2(pRGBX0.s7)));\n"
					"    pV0.s1 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cV, (float3)(amd_unpack0(pRGBX1.s0), amd_unpack1(pRGBX1.s0), amd_unpack2(pRGBX1.s0)));\n"
					"    f.s1 = dot(cV, (float3)(amd_unpack0(pRGBX1.s2), amd_unpack1(pRGBX1.s2), amd_unpack2(pRGBX1.s2)));\n"
					"    f.s2 = dot(cV, (float3)(amd_unpack0(pRGBX1.s4), amd_unpack1(pRGBX1.s4), amd_unpack2(pRGBX1.s4)));\n"
					"    f.s3 = dot(cV, (float3)(amd_unpack0(pRGBX1.s6), amd_unpack1(pRGBX1.s6), amd_unpack2(pRGBX1.s6)));\n"
					"    pV1.s0 = amd_pack(f + (float4)(128));\n"
					"    f.s0 = dot(cV, (float3)(amd_unpack0(pRGBX1.s1), amd_unpack1(pRGBX1.s1), amd_unpack2(pRGBX1.s1)));\n"
					"    f.s1 = dot(cV, (float3)(amd_unpack0(pRGBX1.s3), amd_unpack1(pRGBX1.s3), amd_unpack2(pRGBX1.s3)));\n"
					"    f.s2 = dot(cV, (float3)(amd_unpack0(pRGBX1.s5), amd_unpack1(pRGBX1.s5), amd_unpack2(pRGBX1.s5)));\n"
					"    f.s3 = dot(cV, (float3)(amd_unpack0(pRGBX1.s7), amd_unpack1(pRGBX1.s7), amd_unpack2(pRGBX1.s7)));\n"
					"    pV1.s1 = amd_pack(f + (float4)(128));\n"
					"    pV0.s0 = amd_lerp(pV0.s0, pV0.s1, 0);\n"
					"    pV1.s0 = amd_lerp(pV1.s0, pV1.s1, 0);\n"
					"    pV0.s0 = amd_lerp(pV0.s0, pV1.s0, 0);\n"
					);
			}
		}
		if (destinationHasUV12) {
			node->opencl_code +=
				OPENCL_FORMAT(
				"    U8x8 pUV;\n"
				"    f.s0 = amd_unpack0(pU0.s0);\n"
				"    f.s1 = amd_unpack0(pV0.s0);\n"
				"    f.s2 = amd_unpack1(pU0.s0);\n"
				"    f.s3 = amd_unpack1(pV0.s0);\n"
				"    pUV.s0 = amd_pack(f);\n"
				"    f.s0 = amd_unpack2(pU0.s0);\n"
				"    f.s1 = amd_unpack2(pV0.s0);\n"
				"    f.s2 = amd_unpack3(pU0.s0);\n"
				"    f.s3 = amd_unpack3(pV0.s0);\n"
				"    pUV.s1 = amd_pack(f);\n"
				);
			if (destinationHasY) {
				snprintf(item, sizeof(item),
					OPENCL_FORMAT(
					"    pY_buf += pY_offset + (gy * %d) + (gx << 3);\n" // pY_stride * 2
					"    pUV_buf += pUV_offset + (gy * %d) + (gx << 3);\n" // pUV_stride
					"    *(__global U8x8 *) pY_buf = pY0;\n"
					"    *(__global U8x8 *)&pY_buf[%d] = pY1;\n" // pY_stride
					"    *(__global U8x8 *) pUV_buf = pUV;\n"
					), pY_stride * 2, pUV_stride, pY_stride);
				node->opencl_code += item;
			}
			else {
				snprintf(item, sizeof(item),
					OPENCL_FORMAT(
					"    pUV_buf += pUV_offset + (gy * %d) + (gx << 3);\n" // pUV_stride
					"    *(__global U8x8 *) pUV_buf = pUV;\n"
					), pUV_stride);
				node->opencl_code += item;
			}
		}
		else if (destinationHasY) {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"    pY_buf += pY_offset + (gy * %d) + (gx << 3);\n" // pY_stride * 2
				"    pU_buf += pU_offset + (gy * %d) + (gx << 2);\n" // pU_stride
				"    pV_buf += pV_offset + (gy * %d) + (gx << 2);\n" // pV_stride
				"    *(__global U8x8 *) pY_buf = pY0;\n"
				"    *(__global U8x8 *)&pY_buf[%d] = pY1;\n" // pY_stride
				"    *(__global uint *) pU_buf = pU0.s0;\n"
				"    *(__global uint *) pV_buf = pV0.s0;\n"
				), pY_stride * 2, pU_stride, pV_stride, pY_stride);
			node->opencl_code += item;
		}
		else {
			if (!destinationNoU) {
				snprintf(item, sizeof(item),
					OPENCL_FORMAT(
					"    pU_buf += pU_offset + (gy * %d) + (gx << 2);\n" // pU_stride
					"    *(__global uint *) pU_buf = pU0.s0;\n"
					), pU_stride);
				node->opencl_code += item;
			}
			if (!destinationNoV) {
				snprintf(item, sizeof(item),
					OPENCL_FORMAT(
					"    pV_buf += pV_offset + (gy * %d) + (gx << 2);\n" // pV_stride
					"    *(__global uint *) pV_buf = pV0.s0;\n"
					), pV_stride);
				node->opencl_code += item;
			}
		}
	}
	else {
		if (isSourceUYVY) {
			node->opencl_code +=
				OPENCL_FORMAT(
				"    U8x8 pY0, pY1;\n"
				"    U8x8 pU0, pU1;\n"
				"    U8x8 pV0, pV1;\n"
				"    pY0.s0 = amd_pack((float4)(amd_unpack1(L0.s0), amd_unpack3(L0.s0), amd_unpack1(L0.s1), amd_unpack3(L0.s1)));\n"
				"    pY0.s1 = amd_pack((float4)(amd_unpack1(L0.s2), amd_unpack3(L0.s2), amd_unpack1(L0.s3), amd_unpack3(L0.s3)));\n"
				"    pY1.s0 = amd_pack((float4)(amd_unpack1(L1.s0), amd_unpack3(L1.s0), amd_unpack1(L1.s1), amd_unpack3(L1.s1)));\n"
				"    pY1.s1 = amd_pack((float4)(amd_unpack1(L1.s2), amd_unpack3(L1.s2), amd_unpack1(L1.s3), amd_unpack3(L1.s3)));\n"
				"    pU0.s0 = amd_pack((float4)(amd_unpack0(L0.s0), amd_unpack0(L0.s0), amd_unpack0(L0.s1), amd_unpack0(L0.s1)));\n"
				"    pU0.s1 = amd_pack((float4)(amd_unpack0(L0.s2), amd_unpack0(L0.s2), amd_unpack0(L0.s3), amd_unpack0(L0.s3)));\n"
				"    pU1.s0 = amd_pack((float4)(amd_unpack0(L1.s0), amd_unpack0(L1.s0), amd_unpack0(L1.s1), amd_unpack0(L1.s1)));\n"
				"    pU1.s1 = amd_pack((float4)(amd_unpack0(L1.s2), amd_unpack0(L1.s2), amd_unpack0(L1.s3), amd_unpack0(L1.s3)));\n"
				"    pV0.s0 = amd_pack((float4)(amd_unpack2(L0.s0), amd_unpack2(L0.s0), amd_unpack2(L0.s1), amd_unpack2(L0.s1)));\n"
				"    pV0.s1 = amd_pack((float4)(amd_unpack2(L0.s2), amd_unpack2(L0.s2), amd_unpack2(L0.s3), amd_unpack2(L0.s3)));\n"
				"    pV1.s0 = amd_pack((float4)(amd_unpack2(L1.s0), amd_unpack2(L1.s0), amd_unpack2(L1.s1), amd_unpack2(L1.s1)));\n"
				"    pV1.s1 = amd_pack((float4)(amd_unpack2(L1.s2), amd_unpack2(L1.s2), amd_unpack2(L1.s3), amd_unpack2(L1.s3)));\n"
				);
		}
		else if (isSourceYUYV) {
			node->opencl_code +=
				OPENCL_FORMAT(
				"    U8x8 pY0, pY1;\n"
				"    U8x8 pU0, pU1;\n"
				"    U8x8 pV0, pV1;\n"
				"    pY0.s0 = amd_pack((float4)(amd_unpack0(L0.s0), amd_unpack2(L0.s0), amd_unpack0(L0.s1), amd_unpack2(L0.s1)));\n"
				"    pY0.s1 = amd_pack((float4)(amd_unpack0(L0.s2), amd_unpack2(L0.s2), amd_unpack0(L0.s3), amd_unpack2(L0.s3)));\n"
				"    pY1.s0 = amd_pack((float4)(amd_unpack0(L1.s0), amd_unpack2(L1.s0), amd_unpack0(L1.s1), amd_unpack2(L1.s1)));\n"
				"    pY1.s1 = amd_pack((float4)(amd_unpack0(L1.s2), amd_unpack2(L1.s2), amd_unpack0(L1.s3), amd_unpack2(L1.s3)));\n"
				"    pU0.s0 = amd_pack((float4)(amd_unpack1(L0.s0), amd_unpack1(L0.s0), amd_unpack1(L0.s1), amd_unpack1(L0.s1)));\n"
				"    pU0.s1 = amd_pack((float4)(amd_unpack1(L0.s2), amd_unpack1(L0.s2), amd_unpack1(L0.s3), amd_unpack1(L0.s3)));\n"
				"    pU1.s0 = amd_pack((float4)(amd_unpack1(L1.s0), amd_unpack1(L1.s0), amd_unpack1(L1.s1), amd_unpack1(L1.s1)));\n"
				"    pU1.s1 = amd_pack((float4)(amd_unpack1(L1.s2), amd_unpack1(L1.s2), amd_unpack1(L1.s3), amd_unpack1(L1.s3)));\n"
				"    pV0.s0 = amd_pack((float4)(amd_unpack3(L0.s0), amd_unpack3(L0.s0), amd_unpack3(L0.s1), amd_unpack3(L0.s1)));\n"
				"    pV0.s1 = amd_pack((float4)(amd_unpack3(L0.s2), amd_unpack3(L0.s2), amd_unpack3(L0.s3), amd_unpack3(L0.s3)));\n"
				"    pV1.s0 = amd_pack((float4)(amd_unpack3(L1.s0), amd_unpack3(L1.s0), amd_unpack3(L1.s1), amd_unpack3(L1.s1)));\n"
				"    pV1.s1 = amd_pack((float4)(amd_unpack3(L1.s2), amd_unpack3(L1.s2), amd_unpack3(L1.s3), amd_unpack3(L1.s3)));\n"
				);
		}
		else if (isSourceIYUV) {
			node->opencl_code +=
				OPENCL_FORMAT(
				"    U8x8 pU0, pU1;\n"
				"    U8x8 pV0, pV1;\n"
				"    f.s0 = amd_unpack0(pUV.s0); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack1(pUV.s0); f.s3 = f.s2;\n"
				"    pU0.s0 = amd_pack(f);\n"
				"    f.s0 = amd_unpack2(pUV.s0); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack3(pUV.s0); f.s3 = f.s2;\n"
				"    pU0.s1 = amd_pack(f);\n"
				"    pU1.s0 = pU0.s0;\n"
				"    pU1.s1 = pU0.s1;\n"
				"    f.s0 = amd_unpack0(pUV.s1); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack1(pUV.s1); f.s3 = f.s2;\n"
				"    pV0.s0 = amd_pack(f);\n"
				"    f.s0 = amd_unpack2(pUV.s1); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack3(pUV.s1); f.s3 = f.s2;\n"
				"    pV0.s1 = amd_pack(f);\n"
				"    pV1.s0 = pV0.s0;\n"
				"    pV1.s1 = pV0.s1;\n"
				);
		}
		else if (isSourceNV12) {
			node->opencl_code +=
				OPENCL_FORMAT(
				"    U8x8 pU0, pU1;\n"
				"    U8x8 pV0, pV1;\n"
				"    f.s0 = amd_unpack0(pUV.s0); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack2(pUV.s0); f.s3 = f.s2;\n"
				"    pU0.s0 = amd_pack(f);\n"
				"    f.s0 = amd_unpack0(pUV.s1); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack2(pUV.s1); f.s3 = f.s2;\n"
				"    pU0.s1 = amd_pack(f);\n"
				"    pU1.s0 = pU0.s0;\n"
				"    pU1.s1 = pU0.s1;\n"
				"    f.s0 = amd_unpack1(pUV.s0); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack3(pUV.s0); f.s3 = f.s2;\n"
				"    pV0.s0 = amd_pack(f);\n"
				"    f.s0 = amd_unpack1(pUV.s1); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack3(pUV.s1); f.s3 = f.s2;\n"
				"    pV0.s1 = amd_pack(f);\n"
				"    pV1.s0 = pV0.s0;\n"
				"    pV1.s1 = pV0.s1;\n"
				);
		}
		else if (isSourceNV21) {
			node->opencl_code +=
				OPENCL_FORMAT(
				"    U8x8 pU0, pU1;\n"
				"    U8x8 pV0, pV1;\n"
				"    f.s0 = amd_unpack1(pUV.s0); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack3(pUV.s0); f.s3 = f.s2;\n"
				"    pU0.s0 = amd_pack(f);\n"
				"    f.s0 = amd_unpack1(pUV.s1); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack3(pUV.s1); f.s3 = f.s2;\n"
				"    pU0.s1 = amd_pack(f);\n"
				"    pU1.s0 = pU0.s0;\n"
				"    pU1.s1 = pU0.s1;\n"
				"    f.s0 = amd_unpack0(pUV.s0); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack2(pUV.s0); f.s3 = f.s2;\n"
				"    pV0.s0 = amd_pack(f);\n"
				"    f.s0 = amd_unpack0(pUV.s1); f.s1 = f.s0;\n"
				"    f.s2 = amd_unpack2(pUV.s1); f.s3 = f.s2;\n"
				"    pV0.s1 = amd_pack(f);\n"
				"    pV1.s0 = pV0.s0;\n"
				"    pV1.s1 = pV0.s1;\n"
				);
		}
		else {
			agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_ColorConvert doesn't support kernel %s\n", node->akernel->name);
			return -1;
		}
		if (isDestinationRGB || isDestinationRGBX) {
			if (input_color_space == VX_COLOR_SPACE_BT601_525 || input_color_space == VX_COLOR_SPACE_BT601_625) {
				node->opencl_code += OPENCL_FORMAT(
					"    float2 cR = (float2)( 0.0000f,  1.4030f);\n"
					"    float2 cG = (float2)(-0.3440f, -0.7140f);\n"
					"    float2 cB = (float2)( 1.7730f,  0.0000f);\n"
					);
			}
			else { // VX_COLOR_SPACE_BT709
				node->opencl_code += OPENCL_FORMAT(
					"    float2 cR = (float2)( 0.0000f,  1.5748f);\n"
					"    float2 cG = (float2)(-0.1873f, -0.4681f);\n"
					"    float2 cB = (float2)( 1.8556f,  0.0000f);\n"
					);
			}
			if (input_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
				node->opencl_code += OPENCL_FORMAT(
					"    float4 r2f = (float4)(256f/219f, -16f*256f/219f, 256f/224f, -128f*256f/224f);\n"
					);
			}
		}
		if (isDestinationRGB) {
			if (input_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
				snprintf(item, sizeof(item),
					OPENCL_FORMAT(
					"    float3 yuv; U24x8 pRGB0, pRGB1;\n"
					"    yuv.s0 = mad(amd_unpack0(pY0.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack0(pU0.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack0(pV0.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack1(pY0.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack1(pU0.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack1(pV0.s0),r2f.s2,r2f.s3);\n"
					"    f.s3 = mad(cR.s1, yuv.s2, yuv.s0); pRGB0.s0 = amd_pack(trunc(f)); f.s0 = mad(cG.s0, yuv.s1, yuv.s0); f.s0 = mad(cG.s1, yuv.s2, f.s0); f.s1 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack2(pY0.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack2(pU0.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack2(pV0.s0),r2f.s2,r2f.s3);\n"
					"    f.s2 = mad(cR.s1, yuv.s2, yuv.s0); f.s3 = mad(cG.s0, yuv.s1, yuv.s0); f.s3 = mad(cG.s1, yuv.s2, f.s3); pRGB0.s1 = amd_pack(trunc(f)); f.s0 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack3(pY0.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack3(pU0.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack3(pV0.s0),r2f.s2,r2f.s3);\n"
					"    f.s1 = mad(cR.s1, yuv.s2, yuv.s0); f.s2 = mad(cG.s0, yuv.s1, yuv.s0); f.s2 = mad(cG.s1, yuv.s2, f.s2); f.s3 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s2 = amd_pack(trunc(f));\n"
					"    yuv.s0 = mad(amd_unpack0(pY0.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack0(pU0.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack0(pV0.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack1(pY0.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack1(pU0.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack1(pV0.s1),r2f.s2,r2f.s3);\n"
					"    f.s3 = mad(cR.s1, yuv.s2, yuv.s0); pRGB0.s3 = amd_pack(trunc(f)); f.s0 = mad(cG.s0, yuv.s1, yuv.s0); f.s0 = mad(cG.s1, yuv.s2, f.s0); f.s1 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack2(pY0.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack2(pU0.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack2(pV0.s1),r2f.s2,r2f.s3);\n"
					"    f.s2 = mad(cR.s1, yuv.s2, yuv.s0); f.s3 = mad(cG.s0, yuv.s1, yuv.s0); f.s3 = mad(cG.s1, yuv.s2, f.s3); pRGB0.s4 = amd_pack(trunc(f)); f.s0 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack3(pY0.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack3(pU0.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack3(pV0.s1),r2f.s2,r2f.s3);\n"
					"    f.s1 = mad(cR.s1, yuv.s2, yuv.s0); f.s2 = mad(cG.s0, yuv.s1, yuv.s0); f.s2 = mad(cG.s1, yuv.s2, f.s2); f.s3 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s5 = amd_pack(trunc(f));\n"
					"    yuv.s0 = mad(amd_unpack0(pY1.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack0(pU1.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack0(pV1.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack1(pY1.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack1(pU1.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack1(pV1.s0),r2f.s2,r2f.s3);\n"
					"    f.s3 = mad(cR.s1, yuv.s2, yuv.s0); pRGB1.s0 = amd_pack(trunc(f)); f.s0 = mad(cG.s0, yuv.s1, yuv.s0); f.s0 = mad(cG.s1, yuv.s2, f.s0); f.s1 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack2(pY1.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack2(pU1.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack2(pV1.s0),r2f.s2,r2f.s3);\n"
					"    f.s2 = mad(cR.s1, yuv.s2, yuv.s0); f.s3 = mad(cG.s0, yuv.s1, yuv.s0); f.s3 = mad(cG.s1, yuv.s2, f.s3); pRGB1.s1 = amd_pack(trunc(f)); f.s0 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack3(pY1.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack3(pU1.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack3(pV1.s0),r2f.s2,r2f.s3);\n"
					"    f.s1 = mad(cR.s1, yuv.s2, yuv.s0); f.s2 = mad(cG.s0, yuv.s1, yuv.s0); f.s2 = mad(cG.s1, yuv.s2, f.s2); f.s3 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s2 = amd_pack(trunc(f));\n"
					"    yuv.s0 = mad(amd_unpack0(pY1.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack0(pU1.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack0(pV1.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack1(pY1.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack1(pU1.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack1(pV1.s1),r2f.s2,r2f.s3);\n"
					"    f.s3 = mad(cR.s1, yuv.s2, yuv.s0); pRGB1.s3 = amd_pack(trunc(f)); f.s0 = mad(cG.s0, yuv.s1, yuv.s0); f.s0 = mad(cG.s1, yuv.s2, f.s0); f.s1 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack2(pY1.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack2(pU1.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack2(pV1.s1),r2f.s2,r2f.s3);\n"
					"    f.s2 = mad(cR.s1, yuv.s2, yuv.s0); f.s3 = mad(cG.s0, yuv.s1, yuv.s0); f.s3 = mad(cG.s1, yuv.s2, f.s3); pRGB1.s4 = amd_pack(trunc(f)); f.s0 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = mad(amd_unpack3(pY1.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack3(pU1.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack3(pV1.s1),r2f.s2,r2f.s3);\n"
					"    f.s1 = mad(cR.s1, yuv.s2, yuv.s0); f.s2 = mad(cG.s0, yuv.s1, yuv.s0); f.s2 = mad(cG.s1, yuv.s2, f.s2); f.s3 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s5 = amd_pack(trunc(f));\n"
					"    pRGB_buf += pRGB_offset + (gy * %d) + (gx * 24);\n" // pRGB_stride * 2
					"    *(__global uint4 *) pRGB_buf = pRGB0.s0123;\n"
					"    *(__global uint2 *)&pRGB_buf[16] = pRGB0.s45;\n"
					"    *(__global uint4 *)&pRGB_buf[%d] = pRGB1.s0123;\n" // pRGB_stride
					"    *(__global uint2 *)&pRGB_buf[%d+16] = pRGB1.s45;\n" // pRGB_stride
					), pRGB_stride * 2, pRGB_stride, pRGB_stride);
			}
			else { // VX_CHANNEL_RANGE_FULL
				snprintf(item, sizeof(item),
					OPENCL_FORMAT(
					"    float3 yuv; U24x8 pRGB0, pRGB1;\n"
					"    yuv.s0 = amd_unpack0(pY0.s0); yuv.s1 = amd_unpack0(pU0.s0); yuv.s2 = amd_unpack0(pV0.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack1(pY0.s0); yuv.s1 = amd_unpack1(pU0.s0); yuv.s2 = amd_unpack1(pV0.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s3 = mad(cR.s1, yuv.s2, yuv.s0); pRGB0.s0 = amd_pack(trunc(f)); f.s0 = mad(cG.s0, yuv.s1, yuv.s0); f.s0 = mad(cG.s1, yuv.s2, f.s0); f.s1 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack2(pY0.s0); yuv.s1 = amd_unpack2(pU0.s0); yuv.s2 = amd_unpack2(pV0.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s2 = mad(cR.s1, yuv.s2, yuv.s0); f.s3 = mad(cG.s0, yuv.s1, yuv.s0); f.s3 = mad(cG.s1, yuv.s2, f.s3); pRGB0.s1 = amd_pack(trunc(f)); f.s0 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack3(pY0.s0); yuv.s1 = amd_unpack3(pU0.s0); yuv.s2 = amd_unpack3(pV0.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s1 = mad(cR.s1, yuv.s2, yuv.s0); f.s2 = mad(cG.s0, yuv.s1, yuv.s0); f.s2 = mad(cG.s1, yuv.s2, f.s2); f.s3 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s2 = amd_pack(trunc(f));\n"
					"    yuv.s0 = amd_unpack0(pY0.s1); yuv.s1 = amd_unpack0(pU0.s1); yuv.s2 = amd_unpack0(pV0.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack1(pY0.s1); yuv.s1 = amd_unpack1(pU0.s1); yuv.s2 = amd_unpack1(pV0.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s3 = mad(cR.s1, yuv.s2, yuv.s0); pRGB0.s3 = amd_pack(trunc(f)); f.s0 = mad(cG.s0, yuv.s1, yuv.s0); f.s0 = mad(cG.s1, yuv.s2, f.s0); f.s1 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack2(pY0.s1); yuv.s1 = amd_unpack2(pU0.s1); yuv.s2 = amd_unpack2(pV0.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s2 = mad(cR.s1, yuv.s2, yuv.s0); f.s3 = mad(cG.s0, yuv.s1, yuv.s0); f.s3 = mad(cG.s1, yuv.s2, f.s3); pRGB0.s4 = amd_pack(trunc(f)); f.s0 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack3(pY0.s1); yuv.s1 = amd_unpack3(pU0.s1); yuv.s2 = amd_unpack3(pV0.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s1 = mad(cR.s1, yuv.s2, yuv.s0); f.s2 = mad(cG.s0, yuv.s1, yuv.s0); f.s2 = mad(cG.s1, yuv.s2, f.s2); f.s3 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s5 = amd_pack(trunc(f));\n"
					"    yuv.s0 = amd_unpack0(pY1.s0); yuv.s1 = amd_unpack0(pU1.s0); yuv.s2 = amd_unpack0(pV1.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack1(pY1.s0); yuv.s1 = amd_unpack1(pU1.s0); yuv.s2 = amd_unpack1(pV1.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s3 = mad(cR.s1, yuv.s2, yuv.s0); pRGB1.s0 = amd_pack(trunc(f)); f.s0 = mad(cG.s0, yuv.s1, yuv.s0); f.s0 = mad(cG.s1, yuv.s2, f.s0); f.s1 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack2(pY1.s0); yuv.s1 = amd_unpack2(pU1.s0); yuv.s2 = amd_unpack2(pV1.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s2 = mad(cR.s1, yuv.s2, yuv.s0); f.s3 = mad(cG.s0, yuv.s1, yuv.s0); f.s3 = mad(cG.s1, yuv.s2, f.s3); pRGB1.s1 = amd_pack(trunc(f)); f.s0 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack3(pY1.s0); yuv.s1 = amd_unpack3(pU1.s0); yuv.s2 = amd_unpack3(pV1.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s1 = mad(cR.s1, yuv.s2, yuv.s0); f.s2 = mad(cG.s0, yuv.s1, yuv.s0); f.s2 = mad(cG.s1, yuv.s2, f.s2); f.s3 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s2 = amd_pack(trunc(f));\n"
					"    yuv.s0 = amd_unpack0(pY1.s1); yuv.s1 = amd_unpack0(pU1.s1); yuv.s2 = amd_unpack0(pV1.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack1(pY1.s1); yuv.s1 = amd_unpack1(pU1.s1); yuv.s2 = amd_unpack1(pV1.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s3 = mad(cR.s1, yuv.s2, yuv.s0); pRGB1.s3 = amd_pack(trunc(f)); f.s0 = mad(cG.s0, yuv.s1, yuv.s0); f.s0 = mad(cG.s1, yuv.s2, f.s0); f.s1 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack2(pY1.s1); yuv.s1 = amd_unpack2(pU1.s1); yuv.s2 = amd_unpack2(pV1.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s2 = mad(cR.s1, yuv.s2, yuv.s0); f.s3 = mad(cG.s0, yuv.s1, yuv.s0); f.s3 = mad(cG.s1, yuv.s2, f.s3); pRGB1.s4 = amd_pack(trunc(f)); f.s0 = mad(cB.s0, yuv.s1, yuv.s0);\n"
					"    yuv.s0 = amd_unpack3(pY1.s1); yuv.s1 = amd_unpack3(pU1.s1); yuv.s2 = amd_unpack3(pV1.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s1 = mad(cR.s1, yuv.s2, yuv.s0); f.s2 = mad(cG.s0, yuv.s1, yuv.s0); f.s2 = mad(cG.s1, yuv.s2, f.s2); f.s3 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s5 = amd_pack(trunc(f));\n"
					"    pRGB_buf += pRGB_offset + (gy * %d) + (gx * 24);\n" // pRGB_stride * 2
					"    *(__global uint4 *) pRGB_buf = pRGB0.s0123;\n"
					"    *(__global uint2 *)&pRGB_buf[16] = pRGB0.s45;\n"
					"    *(__global uint4 *)&pRGB_buf[%d] = pRGB1.s0123;\n" // pRGB_stride
					"    *(__global uint2 *)&pRGB_buf[%d+16] = pRGB1.s45;\n" // pRGB_stride
					), pRGB_stride * 2, pRGB_stride, pRGB_stride);
			}
			node->opencl_code += item;
		}
		else if (isDestinationRGBX) {
			if (input_channel_range == VX_CHANNEL_RANGE_RESTRICTED) {
				snprintf(item, sizeof(item),
					OPENCL_FORMAT(
					"    float3 yuv; f.s3 = 255.0f; U32x8 pRGB0, pRGB1;\n"
					"    yuv.s0 = mad(amd_unpack0(pY0.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack0(pU0.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack0(pV0.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s0 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack1(pY0.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack1(pU0.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack1(pV0.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s1 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack2(pY0.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack2(pU0.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack2(pV0.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s2 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack3(pY0.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack3(pU0.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack3(pV0.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s3 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack0(pY0.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack0(pU0.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack0(pV0.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s4 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack1(pY0.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack1(pU0.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack1(pV0.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s5 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack2(pY0.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack2(pU0.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack2(pV0.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s6 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack3(pY0.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack3(pU0.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack3(pV0.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s7 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack0(pY1.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack0(pU1.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack0(pV1.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s0 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack1(pY1.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack1(pU1.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack1(pV1.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s1 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack2(pY1.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack2(pU1.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack2(pV1.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s2 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack3(pY1.s0),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack3(pU1.s0),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack3(pV1.s0),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s3 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack0(pY1.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack0(pU1.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack0(pV1.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s4 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack1(pY1.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack1(pU1.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack1(pV1.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s5 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack2(pY1.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack2(pU1.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack2(pV1.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s6 = amd_pack(f);\n"
					"    yuv.s0 = mad(amd_unpack3(pY1.s1),r2f.s0,r2f.s1); yuv.s1 = mad(amd_unpack3(pU1.s1),r2f.s2,r2f.s3); yuv.s2 = mad(amd_unpack3(pV1.s1),r2f.s2,r2f.s3);\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s7 = amd_pack(f);\n"
					"    pRGB_buf += pRGB_offset + (gy * %d) + (gx << 5);\n" // pRGB_stride * 2
					"    *(__global U32x8 *) pRGB_buf = pRGB0;\n"
					"    *(__global U32x8 *)&pRGB_buf[%d] = pRGB1;\n" // pRGB_stride
					), pRGB_stride * 2, pRGB_stride);
				node->opencl_code += item;
			}
			else { // VX_CHANNEL_RANGE_FULL
				snprintf(item, sizeof(item),
					OPENCL_FORMAT(
					"    float3 yuv; f.s3 = 255.0f; U32x8 pRGB0, pRGB1;\n"
					"    yuv.s0 = amd_unpack0(pY0.s0); yuv.s1 = amd_unpack0(pU0.s0); yuv.s2 = amd_unpack0(pV0.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s0 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack1(pY0.s0); yuv.s1 = amd_unpack1(pU0.s0); yuv.s2 = amd_unpack1(pV0.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s1 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack2(pY0.s0); yuv.s1 = amd_unpack2(pU0.s0); yuv.s2 = amd_unpack2(pV0.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s2 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack3(pY0.s0); yuv.s1 = amd_unpack3(pU0.s0); yuv.s2 = amd_unpack3(pV0.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s3 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack0(pY0.s1); yuv.s1 = amd_unpack0(pU0.s1); yuv.s2 = amd_unpack0(pV0.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s4 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack1(pY0.s1); yuv.s1 = amd_unpack1(pU0.s1); yuv.s2 = amd_unpack1(pV0.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s5 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack2(pY0.s1); yuv.s1 = amd_unpack2(pU0.s1); yuv.s2 = amd_unpack2(pV0.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s6 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack3(pY0.s1); yuv.s1 = amd_unpack3(pU0.s1); yuv.s2 = amd_unpack3(pV0.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB0.s7 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack0(pY1.s0); yuv.s1 = amd_unpack0(pU1.s0); yuv.s2 = amd_unpack0(pV1.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s0 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack1(pY1.s0); yuv.s1 = amd_unpack1(pU1.s0); yuv.s2 = amd_unpack1(pV1.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s1 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack2(pY1.s0); yuv.s1 = amd_unpack2(pU1.s0); yuv.s2 = amd_unpack2(pV1.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s2 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack3(pY1.s0); yuv.s1 = amd_unpack3(pU1.s0); yuv.s2 = amd_unpack3(pV1.s0); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s3 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack0(pY1.s1); yuv.s1 = amd_unpack0(pU1.s1); yuv.s2 = amd_unpack0(pV1.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s4 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack1(pY1.s1); yuv.s1 = amd_unpack1(pU1.s1); yuv.s2 = amd_unpack1(pV1.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s5 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack2(pY1.s1); yuv.s1 = amd_unpack2(pU1.s1); yuv.s2 = amd_unpack2(pV1.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s6 = amd_pack(f);\n"
					"    yuv.s0 = amd_unpack3(pY1.s1); yuv.s1 = amd_unpack3(pU1.s1); yuv.s2 = amd_unpack3(pV1.s1); yuv.s1 -= 128.0f;; yuv.s2 -= 128.0f;\n"
					"    f.s0 = mad(cR.s1, yuv.s2, yuv.s0); f.s1 = mad(cG.s0, yuv.s1, yuv.s0); f.s1 = mad(cG.s1, yuv.s2, f.s1); f.s2 = mad(cB.s0, yuv.s1, yuv.s0); pRGB1.s7 = amd_pack(f);\n"
					"    pRGB_buf += pRGB_offset + (gy * %d) + (gx << 5);\n" // pRGB_stride * 2
					"    *(__global U32x8 *) pRGB_buf = pRGB0;\n"
					"    *(__global U32x8 *)&pRGB_buf[%d] = pRGB1;\n" // pRGB_stride
					), pRGB_stride * 2, pRGB_stride);
				node->opencl_code += item;
			}
		}
	}
	node->opencl_code +=
		"  }\n"
		"}\n"
		;

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_work_dim = 2;
	node->opencl_global_work[0] = (((width + 7) >> 3) + work_group_width - 1) & ~(work_group_width - 1);
	node->opencl_global_work[1] = (((height + 1) >> 1) + work_group_height - 1) & ~(work_group_height - 1);
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;

	return status;
}

#endif
