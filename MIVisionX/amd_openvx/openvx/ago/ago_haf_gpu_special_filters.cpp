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

#define ENABLE_FAST_MEDIAN_3x3       0   // 0:disable 1:enable fast shortcut for median 3x3

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Useful pre-defined filters
//
static float scharrFilter_3x3_x[3][3] = {
	{ -3, 0,  3 },
	{-10, 0, 10 },
	{ -3, 0,  3 },
};
static float scharrFilter_3x3_y[3][3] = {
	{ -3, -10, -3 },
	{  0,   0,  0 },
	{  3,  10,  3 },
};
static float sobelFilter_3x3_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1},
};
static float sobelFilter_3x3_y[3][3] = {
        {-1,-2,-1},
        { 0, 0, 0},
        { 1, 2, 1},
};
static float sobelFilter_5x5_x[5][5] = {
        {-1, -2, 0, 2, 1},
        {-4, -8, 0, 8, 4},
        {-6,-12, 0,12, 6},
        {-4, -8, 0, 8, 4},
        {-1, -2, 0, 2, 1},
};
static float sobelFilter_5x5_y[5][5] = {
        {-1,-4, -6,-4,-1},
        {-2,-8,-12,-8,-2},
        { 0, 0,  0, 0, 0},
        { 2, 8, 12, 8, 2},
        { 1, 4,  6, 4, 1},
};
static float sobelFilter_7x7_x[7][7] = {
        {  -1,  -4,  -5, 0,   5,  4,  1},
        {  -6, -24, -30, 0,  30, 24,  6},
        { -15, -60, -75, 0,  75, 60, 15},
        { -20, -80,-100, 0, 100, 80, 20},
        { -15, -60, -75, 0,  75, 60, 15},
        {  -6, -24, -30, 0,  30, 24,  6},
        {  -1,  -4,  -5, 0,   5,  4,  1},
};
static float sobelFilter_7x7_y[7][7] = {
        {-1, -6,-15, -20,-15, -6,-1},
        {-4,-24,-60, -80,-60,-24,-4},
        {-5,-30,-75,-100,-75,-30,-5},
        { 0,  0,  0,   0,  0,  0, 0},
        { 5, 30, 75, 100, 75, 30, 5},
        { 4, 24, 60,  80, 60, 24, 4},
        { 1,  6, 15,  20, 15,  6, 1},
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for following non-linear filter kernels:
//   VX_KERNEL_AMD_DILATE_U8_U8_3x3, VX_KERNEL_AMD_DILATE_U1_U8_3x3,
//   VX_KERNEL_AMD_ERODE_U8_U8_3x3, VX_KERNEL_AMD_ERODE_U1_U8_3x3, 
//   VX_KERNEL_AMD_MEDIAN_U8_U8_3x3
//
int HafGpu_NonLinearFilter_3x3_ANY_U8(AgoNode * node)
{
	int status = VX_SUCCESS;
	// get destination type
	const char * dstRegType = "U8";
	bool dstIsU1 = false;
	if (node->paramList[0]->u.img.format == VX_DF_IMAGE_U1_AMD) {
		dstRegType = "U1";
		dstIsU1 = true;
	}
	else if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8) {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_NonLinearFilter_3x3_ANY_U8 doesn't support non-U8/U1 destinations for kernel %s\n", node->akernel->name);
		return -1;
	}
	// function declaration
	char item[8192];
	snprintf(item, sizeof(item), "void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n", node->opencl_name, dstRegType);
	std::string code = item;

	// configuration
	vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;
	vx_uint32 LMemWidth = AGO_OPENCL_WORKGROUP_SIZE_0 * 8;
	vx_uint32 LMemSideLR = 4;
	vx_uint32 LMemSideTB = 1;
	vx_uint32 LMemStride = LMemWidth + 2 * LMemSideLR;
	vx_uint32 LMemSize = (LMemHeight + 2 * LMemSideTB) * LMemStride;
	node->opencl_param_discard_mask = 0;
	node->opencl_local_buffer_usage_mask = (1 << 1);
	node->opencl_local_buffer_size_in_bytes = LMemSize;

	// generate local memory load
	code +=
		"  int lx = get_local_id(0);\n"
		"  int ly = get_local_id(1);\n"
		"  int gx = x >> 3;\n"
		"  int gy = y;\n"
		"  int gstride = stride;\n"
		"  __global uchar * gbuf = p;\n";
	if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemStride, LMemHeight + 2 * LMemSideTB, LMemSideLR, LMemSideTB, code) < 0) {
		return -1;
	}

	// generate computation
	if (node->akernel->id == VX_KERNEL_AMD_DILATE_U8_U8_3x3 || node->akernel->id == VX_KERNEL_AMD_DILATE_U1_U8_3x3) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			"  F32x8 sum; uint4 pix; float4 val;\n"
			"  pix.s01 = lbufptr[0];\n"
			"  pix.s23 = lbufptr[1];\n"
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  sum.s0 = amd_max3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  sum.s1 = amd_max3(val.s0, val.s1, val.s2);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  sum.s2 = amd_max3(val.s0, val.s1, val.s2);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  sum.s3 = amd_max3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  sum.s4 = amd_max3(val.s0, val.s1, val.s2);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  sum.s5 = amd_max3(val.s0, val.s1, val.s2);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  sum.s6 = amd_max3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  sum.s7 = amd_max3(val.s0, val.s1, val.s2);\n"
			"  pix.s01 = lbufptr[%d];\n" // LMemStride / 8
			"  pix.s23 = lbufptr[%d];\n" // LMemStride / 8 + 1
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s0 = max(sum.s0, val.s3);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s1 = max(sum.s1, val.s3);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s2 = max(sum.s2, val.s3);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s3 = max(sum.s3, val.s3);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s4 = max(sum.s4, val.s3);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s5 = max(sum.s5, val.s3);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s6 = max(sum.s6, val.s3);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s7 = max(sum.s7, val.s3);\n"
			"  pix.s01 = lbufptr[%d];\n" // 2 * LMemStride / 8
			"  pix.s23 = lbufptr[%d];\n" // 2 * LMemStride / 8 + 1
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s0 = max(sum.s0, val.s3);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s1 = max(sum.s1, val.s3);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s2 = max(sum.s2, val.s3);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s3 = max(sum.s3, val.s3);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s4 = max(sum.s4, val.s3);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s5 = max(sum.s5, val.s3);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s6 = max(sum.s6, val.s3);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s7 = max(sum.s7, val.s3);\n"
			)
		, LMemStride, LMemStride / 8, LMemStride / 8 + 1, LMemStride * 2 / 8, LMemStride * 2 / 8 + 1);
		code += item;
	}
	else if (node->akernel->id == VX_KERNEL_AMD_ERODE_U8_U8_3x3 || node->akernel->id == VX_KERNEL_AMD_ERODE_U1_U8_3x3) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			"  F32x8 sum; uint4 pix; float4 val;\n"
			"  pix.s01 = lbufptr[0];\n"
			"  pix.s23 = lbufptr[1];\n"
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  sum.s0 = amd_min3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  sum.s1 = amd_min3(val.s0, val.s1, val.s2);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  sum.s2 = amd_min3(val.s0, val.s1, val.s2);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  sum.s3 = amd_min3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  sum.s4 = amd_min3(val.s0, val.s1, val.s2);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  sum.s5 = amd_min3(val.s0, val.s1, val.s2);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  sum.s6 = amd_min3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  sum.s7 = amd_min3(val.s0, val.s1, val.s2);\n"
			"  pix.s01 = lbufptr[%d];\n" // LMemStride / 8
			"  pix.s23 = lbufptr[%d];\n" // LMemStride / 8 + 1
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s0 = min(sum.s0, val.s3);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s1 = min(sum.s1, val.s3);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s2 = min(sum.s2, val.s3);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s3 = min(sum.s3, val.s3);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s4 = min(sum.s4, val.s3);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s5 = min(sum.s5, val.s3);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s6 = min(sum.s6, val.s3);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s7 = min(sum.s7, val.s3);\n"
			"  pix.s01 = lbufptr[%d];\n" // 2 * LMemStride / 8
			"  pix.s23 = lbufptr[%d];\n" // 2 * LMemStride / 8 + 1
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s0 = min(sum.s0, val.s3);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s1 = min(sum.s1, val.s3);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s2 = min(sum.s2, val.s3);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s3 = min(sum.s3, val.s3);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s4 = min(sum.s4, val.s3);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s5 = min(sum.s5, val.s3);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s6 = min(sum.s6, val.s3);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  val.s3 = amd_min3(val.s0, val.s1, val.s2); sum.s7 = min(sum.s7, val.s3);\n"
			)
			, LMemStride, LMemStride / 8, LMemStride / 8 + 1, LMemStride * 2 / 8, LMemStride * 2 / 8 + 1);
		code += item;
	}
	else if (node->akernel->id == VX_KERNEL_AMD_MEDIAN_U8_U8_3x3) {
#if ENABLE_FAST_MEDIAN_3x3
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			"  F32x8 sum, tum; uint4 pix; float4 val;\n"
			"  pix.s01 = lbufptr[0];\n"
			"  pix.s23 = lbufptr[1];\n"
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  sum.s0 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  sum.s1 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  sum.s2 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  sum.s3 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  sum.s4 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  sum.s5 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  sum.s6 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  sum.s7 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  pix.s01 = lbufptr[%d];\n" // LMemStride / 8
			"  pix.s23 = lbufptr[%d];\n" // LMemStride / 8 + 1
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  tum.s0 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  tum.s1 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  tum.s2 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  tum.s3 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  tum.s4 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  tum.s5 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  tum.s6 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  tum.s7 = amd_median3(val.s0, val.s1, val.s2);\n"
			"  pix.s01 = lbufptr[%d];\n" // 2 * LMemStride / 8
			"  pix.s23 = lbufptr[%d];\n" // 2 * LMemStride / 8 + 1
			"  val.s0 = amd_unpack3(pix.s0);\n"
			"  val.s1 = amd_unpack0(pix.s1);\n"
			"  val.s2 = amd_unpack1(pix.s1);\n"
			"  val.s3 = amd_median3(val.s0, val.s1, val.s2); sum.s0 = amd_median3(sum.s0, tum.s0, val.s3);\n"
			"  val.s0 = amd_unpack2(pix.s1);\n"
			"  val.s3 = amd_median3(val.s0, val.s1, val.s2); sum.s1 = amd_median3(sum.s1, tum.s1, val.s3);\n"
			"  val.s1 = amd_unpack3(pix.s1);\n"
			"  val.s3 = amd_median3(val.s0, val.s1, val.s2); sum.s2 = amd_median3(sum.s2, tum.s2, val.s3);\n"
			"  val.s2 = amd_unpack0(pix.s2);\n"
			"  val.s3 = amd_median3(val.s0, val.s1, val.s2); sum.s3 = amd_median3(sum.s3, tum.s3, val.s3);\n"
			"  val.s0 = amd_unpack1(pix.s2);\n"
			"  val.s3 = amd_median3(val.s0, val.s1, val.s2); sum.s4 = amd_median3(sum.s4, tum.s4, val.s3);\n"
			"  val.s1 = amd_unpack2(pix.s2);\n"
			"  val.s3 = amd_median3(val.s0, val.s1, val.s2); sum.s5 = amd_median3(sum.s5, tum.s5, val.s3);\n"
			"  val.s2 = amd_unpack3(pix.s2);\n"
			"  val.s3 = amd_median3(val.s0, val.s1, val.s2); sum.s6 = amd_median3(sum.s6, tum.s6, val.s3);\n"
			"  val.s0 = amd_unpack0(pix.s3);\n"
			"  val.s3 = amd_median3(val.s0, val.s1, val.s2); sum.s7 = amd_median3(sum.s7, tum.s7, val.s3);\n"
			)
			, LMemStride, LMemStride / 8, LMemStride / 8 + 1, LMemStride * 2 / 8, LMemStride * 2 / 8 + 1);
		code += item;
#else
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			"  F32x8 sum;\n"
			"  float4 val0, val1, val2, valz;\n"
			"  uint4 pix0, pix1, pix2;\n"
			"  pix0.s01 = lbufptr[0];\n"
			"  pix0.s23 = lbufptr[1];\n"
			"  pix1.s01 = lbufptr[%d];\n" //     LMemStride / 8
			"  pix1.s23 = lbufptr[%d];\n" //     LMemStride / 8 + 1
			"  pix2.s01 = lbufptr[%d];\n" // 2 * LMemStride / 8
			"  pix2.s23 = lbufptr[%d];\n" // 2 * LMemStride / 8 + 1
			), LMemStride, LMemStride / 8, LMemStride / 8 + 1, LMemStride * 2 / 8, LMemStride * 2 / 8 + 1);
		code += item;
		code +=
			OPENCL_FORMAT(
			"  // pixel 0\n"
			"  valz.s0 = amd_unpack3(pix0.s0);\n"
			"  valz.s1 = amd_unpack0(pix0.s1);\n"
			"  valz.s2 = amd_unpack1(pix0.s1);\n"
			"  val0.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val0.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val0.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack3(pix1.s0);\n"
			"  valz.s1 = amd_unpack0(pix1.s1);\n"
			"  valz.s2 = amd_unpack1(pix1.s1);\n"
			"  val1.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val1.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val1.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack3(pix2.s0);\n"
			"  valz.s1 = amd_unpack0(pix2.s1);\n"
			"  valz.s2 = amd_unpack1(pix2.s1);\n"
			"  val2.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val2.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val2.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_max3   (val0.s0, val1.s0, val2.s0);\n"
			"  valz.s1 = amd_median3(val0.s1, val1.s1, val2.s1);\n"
			"  valz.s2 = amd_min3   (val0.s2, val1.s2, val2.s2);\n"
			"  sum.s0  = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  // pixel 1\n"
			"  valz.s0 = amd_unpack0(pix0.s1);\n"
			"  valz.s1 = amd_unpack1(pix0.s1);\n"
			"  valz.s2 = amd_unpack2(pix0.s1);\n"
			"  val0.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val0.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val0.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack0(pix1.s1);\n"
			"  valz.s1 = amd_unpack1(pix1.s1);\n"
			"  valz.s2 = amd_unpack2(pix1.s1);\n"
			"  val1.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val1.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val1.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack0(pix2.s1);\n"
			"  valz.s1 = amd_unpack1(pix2.s1);\n"
			"  valz.s2 = amd_unpack2(pix2.s1);\n"
			"  val2.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val2.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val2.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_max3   (val0.s0, val1.s0, val2.s0);\n"
			"  valz.s1 = amd_median3(val0.s1, val1.s1, val2.s1);\n"
			"  valz.s2 = amd_min3   (val0.s2, val1.s2, val2.s2);\n"
			"  sum.s1  = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  // pixel 2\n"
			"  valz.s0 = amd_unpack1(pix0.s1);\n"
			"  valz.s1 = amd_unpack2(pix0.s1);\n"
			"  valz.s2 = amd_unpack3(pix0.s1);\n"
			"  val0.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val0.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val0.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack1(pix1.s1);\n"
			"  valz.s1 = amd_unpack2(pix1.s1);\n"
			"  valz.s2 = amd_unpack3(pix1.s1);\n"
			"  val1.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val1.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val1.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack1(pix2.s1);\n"
			"  valz.s1 = amd_unpack2(pix2.s1);\n"
			"  valz.s2 = amd_unpack3(pix2.s1);\n"
			"  val2.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val2.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val2.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_max3   (val0.s0, val1.s0, val2.s0);\n"
			"  valz.s1 = amd_median3(val0.s1, val1.s1, val2.s1);\n"
			"  valz.s2 = amd_min3   (val0.s2, val1.s2, val2.s2);\n"
			"  sum.s2  = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  // pixel 3\n"
			"  valz.s0 = amd_unpack2(pix0.s1);\n"
			"  valz.s1 = amd_unpack3(pix0.s1);\n"
			"  valz.s2 = amd_unpack0(pix0.s2);\n"
			"  val0.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val0.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val0.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack2(pix1.s1);\n"
			"  valz.s1 = amd_unpack3(pix1.s1);\n"
			"  valz.s2 = amd_unpack0(pix1.s2);\n"
			"  val1.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val1.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val1.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack2(pix2.s1);\n"
			"  valz.s1 = amd_unpack3(pix2.s1);\n"
			"  valz.s2 = amd_unpack0(pix2.s2);\n"
			"  val2.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val2.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val2.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_max3   (val0.s0, val1.s0, val2.s0);\n"
			"  valz.s1 = amd_median3(val0.s1, val1.s1, val2.s1);\n"
			"  valz.s2 = amd_min3   (val0.s2, val1.s2, val2.s2);\n"
			"  sum.s3  = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  // pixel 4\n"
			"  valz.s0 = amd_unpack3(pix0.s1);\n"
			"  valz.s1 = amd_unpack0(pix0.s2);\n"
			"  valz.s2 = amd_unpack1(pix0.s2);\n"
			"  val0.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val0.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val0.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack3(pix1.s1);\n"
			"  valz.s1 = amd_unpack0(pix1.s2);\n"
			"  valz.s2 = amd_unpack1(pix1.s2);\n"
			"  val1.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val1.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val1.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack3(pix2.s1);\n"
			"  valz.s1 = amd_unpack0(pix2.s2);\n"
			"  valz.s2 = amd_unpack1(pix2.s2);\n"
			"  val2.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val2.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val2.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_max3   (val0.s0, val1.s0, val2.s0);\n"
			"  valz.s1 = amd_median3(val0.s1, val1.s1, val2.s1);\n"
			"  valz.s2 = amd_min3   (val0.s2, val1.s2, val2.s2);\n"
			"  sum.s4  = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  // pixel 5\n"
			"  valz.s0 = amd_unpack0(pix0.s2);\n"
			"  valz.s1 = amd_unpack1(pix0.s2);\n"
			"  valz.s2 = amd_unpack2(pix0.s2);\n"
			"  val0.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val0.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val0.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack0(pix1.s2);\n"
			"  valz.s1 = amd_unpack1(pix1.s2);\n"
			"  valz.s2 = amd_unpack2(pix1.s2);\n"
			"  val1.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val1.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val1.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack0(pix2.s2);\n"
			"  valz.s1 = amd_unpack1(pix2.s2);\n"
			"  valz.s2 = amd_unpack2(pix2.s2);\n"
			"  val2.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val2.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val2.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_max3   (val0.s0, val1.s0, val2.s0);\n"
			"  valz.s1 = amd_median3(val0.s1, val1.s1, val2.s1);\n"
			"  valz.s2 = amd_min3   (val0.s2, val1.s2, val2.s2);\n"
			"  sum.s5  = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  // pixel 6\n"
			"  valz.s0 = amd_unpack1(pix0.s2);\n"
			"  valz.s1 = amd_unpack2(pix0.s2);\n"
			"  valz.s2 = amd_unpack3(pix0.s2);\n"
			"  val0.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val0.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val0.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack1(pix1.s2);\n"
			"  valz.s1 = amd_unpack2(pix1.s2);\n"
			"  valz.s2 = amd_unpack3(pix1.s2);\n"
			"  val1.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val1.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val1.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack1(pix2.s2);\n"
			"  valz.s1 = amd_unpack2(pix2.s2);\n"
			"  valz.s2 = amd_unpack3(pix2.s2);\n"
			"  val2.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val2.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val2.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_max3   (val0.s0, val1.s0, val2.s0);\n"
			"  valz.s1 = amd_median3(val0.s1, val1.s1, val2.s1);\n"
			"  valz.s2 = amd_min3   (val0.s2, val1.s2, val2.s2);\n"
			"  sum.s6  = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  // pixel 7\n"
			"  valz.s0 = amd_unpack2(pix0.s2);\n"
			"  valz.s1 = amd_unpack3(pix0.s2);\n"
			"  valz.s2 = amd_unpack0(pix0.s3);\n"
			"  val0.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val0.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val0.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack2(pix1.s2);\n"
			"  valz.s1 = amd_unpack3(pix1.s2);\n"
			"  valz.s2 = amd_unpack0(pix1.s3);\n"
			"  val1.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val1.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val1.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_unpack2(pix2.s2);\n"
			"  valz.s1 = amd_unpack3(pix2.s2);\n"
			"  valz.s2 = amd_unpack0(pix2.s3);\n"
			"  val2.s0 = amd_min3   (valz.s0, valz.s1, valz.s2);\n"
			"  val2.s1 = amd_median3(valz.s0, valz.s1, valz.s2);\n"
			"  val2.s2 = amd_max3   (valz.s0, valz.s1, valz.s2);\n"
			"  valz.s0 = amd_max3   (val0.s0, val1.s0, val2.s0);\n"
			"  valz.s1 = amd_median3(val0.s1, val1.s1, val2.s1);\n"
			"  valz.s2 = amd_min3   (val0.s2, val1.s2, val2.s2);\n"
			"  sum.s7  = amd_median3(valz.s0, valz.s1, valz.s2);\n");
#endif
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_NonLinearFilter_3x3_ANY_U8 doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}

	if (dstIsU1) {
		code +=
			OPENCL_FORMAT(
			"  U8x8 rv;\n"
			"  rv.s0 = amd_pack(sum.s0123);\n"
			"  rv.s1 = amd_pack(sum.s4567);\n"
			"  Convert_U1_U8(r, rv);\n"
			"}\n");
	}
	else {
		code +=
			OPENCL_FORMAT(
			"  U8x8 rv;\n"
			"  rv.s0 = amd_pack(sum.s0123);\n"
			"  rv.s1 = amd_pack(sum.s4567);\n"
			"  *r = rv;\n"
			"}\n");
	}

	node->opencl_code = code;
	node->opencl_type = NODE_OPENCL_TYPE_MEM2REG;

	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for following non-linear filter kernels:
//   VX_KERNEL_AMD_DILATE_U8_U1_3x3, VX_KERNEL_AMD_DILATE_U1_U1_3x3,
//   VX_KERNEL_AMD_ERODE_U8_U1_3x3, VX_KERNEL_AMD_ERODE_U1_U1_3x3, 
//
int HafGpu_NonLinearFilter_3x3_ANY_U1(AgoNode * node)
{
	int status = VX_SUCCESS;
	// get destination type
	const char * dstRegType = "U8";
	bool dstIsU1 = false;
	if (node->paramList[0]->u.img.format == VX_DF_IMAGE_U1_AMD) {
		dstRegType = "U1";
		dstIsU1 = true;
	}
	else if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8) {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_NonLinearFilter_3x3_ANY_U1 doesn't support non-U8/U1 destinations for kernel %s\n", node->akernel->name);
		return -1;
	}
	// function declaration
	char item[8192];
	snprintf(item, sizeof(item), "void %s(%sx8 * r, uint x, uint y, __global uchar * p, uint stride) {\n", node->opencl_name, dstRegType);
	std::string code = item;

	// configuration
	int stride = node->paramList[1]->u.img.stride_in_bytes;
	node->opencl_param_discard_mask = 0;
	node->opencl_local_buffer_usage_mask = 0;
	node->opencl_local_buffer_size_in_bytes = 0;

	// generate computation
	if (node->akernel->id == VX_KERNEL_AMD_DILATE_U8_U1_3x3 || 
		node->akernel->id == VX_KERNEL_AMD_DILATE_U1_U1_3x3 ||
		node->akernel->id == VX_KERNEL_AMD_ERODE_U8_U1_3x3 ||
		node->akernel->id == VX_KERNEL_AMD_ERODE_U1_U1_3x3) {
		int op = (node->akernel->id == VX_KERNEL_AMD_DILATE_U8_U1_3x3 || node->akernel->id == VX_KERNEL_AMD_DILATE_U1_U1_3x3) ? '|' : '&';
		snprintf(item, sizeof(item),
			// TBD: this code segment uses risky 32-bit loads without 32-bit alignment
			//      it works great on our hardware though it doesn't follow OpenCL rules
			OPENCL_FORMAT(
			"  x = (x >> 3) - 1;\n"
			"  p += y * %d + x;\n" // stride
			"  uint L0 = *(__global uint *)&p[-%d];\n" //  stride
			"  uint L1 = *(__global uint *) p;\n"
			"  uint L2 = *(__global uint *)&p[%d];\n" //  stride
			"  L0 = L0 %c (L0 >> 1) %c (L0 << 1);\n" // op, op
			"  L1 = L1 %c (L1 >> 1) %c (L1 << 1);\n" // op, op
			"  L2 = L2 %c (L2 >> 1) %c (L2 << 1);\n" // op, op
			"  L0 = L0 %c  L1       %c  L2;\n" // op, op
			"  L0 = L0 >> 8;\n"
			), stride, stride, stride, op, op, op, op, op, op, op, op);
		code += item;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_NonLinearFilter_3x3_ANY_U1 doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}

	if (dstIsU1) {
		code +=
			OPENCL_FORMAT(
			"  *r = (U1x8)L0;\n"
			"}\n");
	}
	else {
		code +=
			OPENCL_FORMAT(
			"  Convert_U8_U1(r, (U1x8)L0);\n"
			"}\n");
	}

	node->opencl_code = code;
	node->opencl_type = NODE_OPENCL_TYPE_MEM2REG;

	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following special case Sobel filter kernels:
//   VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GX
//   VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GY
//   VX_KERNEL_AMD_SOBEL_S16S16_U8_3x3_GXY
//   VX_KERNEL_AMD_SOBEL_MAGNITUDE_PHASE_S16U8_U8_3x3
//   VX_KERNEL_AMD_SOBEL_MAGNITUDE_S16_U8_3x3
//   VX_KERNEL_AMD_SOBEL_PHASE_U8_U8_3x3
//
int HafGpu_SobelSpecialCases(AgoNode * node)
{
	int status = VX_SUCCESS;

	if (node->akernel->id == VX_KERNEL_AMD_SOBEL_S16S16_U8_3x3_GXY) {
		AgoData filterGX, filterGY;
		filterGX.ref.type = VX_TYPE_MATRIX; filterGX.u.mat.type = VX_TYPE_FLOAT32; filterGX.u.mat.columns = filterGX.u.mat.rows = 3; filterGX.buffer = (vx_uint8 *)&sobelFilter_3x3_x[0][0]; filterGX.ref.read_only = true;
		filterGY.ref.type = VX_TYPE_MATRIX; filterGY.u.mat.type = VX_TYPE_FLOAT32; filterGY.u.mat.columns = filterGY.u.mat.rows = 3; filterGY.buffer = (vx_uint8 *)&sobelFilter_3x3_y[0][0]; filterGY.ref.read_only = true;
		status = HafGpu_LinearFilter_ANYx2_U8(node, VX_DF_IMAGE_S16, &filterGX, &filterGY, false);
	}
	else if (node->akernel->id == VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GX) {
		AgoData filterGX;
		filterGX.ref.type = VX_TYPE_MATRIX; filterGX.u.mat.type = VX_TYPE_FLOAT32; filterGX.u.mat.columns = filterGX.u.mat.rows = 3; filterGX.buffer = (vx_uint8 *)&sobelFilter_3x3_x[0][0]; filterGX.ref.read_only = true;
		status = HafGpu_LinearFilter_ANY_U8(node, VX_DF_IMAGE_S16, &filterGX, false);
	}
	else if (node->akernel->id == VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GY) {
		AgoData filterGY;
		filterGY.ref.type = VX_TYPE_MATRIX; filterGY.u.mat.type = VX_TYPE_FLOAT32; filterGY.u.mat.columns = filterGY.u.mat.rows = 3; filterGY.buffer = (vx_uint8 *)&sobelFilter_3x3_y[0][0]; filterGY.ref.read_only = true;
		status = HafGpu_LinearFilter_ANY_U8(node, VX_DF_IMAGE_S16, &filterGY, false);
	}
	else {
		// for other special cases
		// re-use LinearFilter_ANYx2_U8 for computing GX & GY
		char opencl_name[VX_MAX_KERNEL_NAME];
		strcpy(opencl_name, node->opencl_name);
		snprintf(node->opencl_name, sizeof(node->opencl_name), "%s_GXY", opencl_name);
		AgoData filterGX, filterGY;
		filterGX.ref.type = VX_TYPE_MATRIX; filterGX.u.mat.type = VX_TYPE_FLOAT32; filterGX.u.mat.columns = filterGX.u.mat.rows = 3; filterGX.buffer = (vx_uint8 *)&sobelFilter_3x3_x[0][0]; filterGX.ref.read_only = true;
		filterGY.ref.type = VX_TYPE_MATRIX; filterGY.u.mat.type = VX_TYPE_FLOAT32; filterGY.u.mat.columns = filterGY.u.mat.rows = 3; filterGY.buffer = (vx_uint8 *)&sobelFilter_3x3_y[0][0]; filterGY.ref.read_only = true;
		status = HafGpu_LinearFilter_ANYx2_U8(node, VX_DF_IMAGE_S16, &filterGX, &filterGY, false);
		strcpy(node->opencl_name, opencl_name);
		if (status) {
			return status;
		}

		// actual function using pre-defined functions
		char item[8192];
		snprintf(item, sizeof(item), OPENCL_FORMAT(
			"#define Magnitude_S16_S16S16 Magnitude_S16_S16S16_%s\n"
			"#define Phase_U8_S16S16 Phase_U8_S16S16_%s\n"
			"void Magnitude_S16_S16S16 (S16x8 * p0, S16x8 p1, S16x8 p2)\n"
			"{\n"
			"	S16x8 r;\n"
			"	float2 f;\n"
			"	f.s0 = (float)((((int)(p1.s0)) << 16) >> 16); f.s1 = (float)((((int)(p2.s0)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s0  = (uint)(f.s0);\n"
			"	f.s0 = (float)(( (int)(p1.s0))        >> 16); f.s1 = (float)(( (int)(p2.s0))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s0 |= (uint)(f.s0) << 16;\n"
			"	f.s0 = (float)((((int)(p1.s1)) << 16) >> 16); f.s1 = (float)((((int)(p2.s1)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s1  = (uint)(f.s0);\n"
			"	f.s0 = (float)(( (int)(p1.s1))        >> 16); f.s1 = (float)(( (int)(p2.s1))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s1 |= (uint)(f.s0) << 16;\n"
			"	f.s0 = (float)((((int)(p1.s2)) << 16) >> 16); f.s1 = (float)((((int)(p2.s2)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s2  = (uint)(f.s0);\n"
			"	f.s0 = (float)(( (int)(p1.s2))        >> 16); f.s1 = (float)(( (int)(p2.s2))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s2 |= (uint)(f.s0) << 16;\n"
			"	f.s0 = (float)((((int)(p1.s3)) << 16) >> 16); f.s1 = (float)((((int)(p2.s3)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s3  = (uint)(f.s0);\n"
			"	f.s0 = (float)(( (int)(p1.s3))        >> 16); f.s1 = (float)(( (int)(p2.s3))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s3 |= (uint)(f.s0) << 16;\n"
			"	*p0 = r;\n"
			"}\n"
			"\n"
			"void Phase_U8_S16S16 (U8x8 * p0, S16x8 p1, S16x8 p2)\n"
			"{\n"
			"	U8x8 r;\n"
			"	float2 f; float4 p4;\n"
			"	f.s0 = (float)((((int)(p1.s0)) << 16) >> 16); f.s1 = (float)((((int)(p2.s0)) << 16) >> 16); p4.s0 = atan2pi(f.s1, f.s0); p4.s0 += (p4.s0 < 0.0) ? 2.0f : 0.0; p4.s0 *= 128.0f;\n"
			"	f.s0 = (float)(( (int)(p1.s0))        >> 16); f.s1 = (float)(( (int)(p2.s0))        >> 16); p4.s1 = atan2pi(f.s1, f.s0); p4.s1 += (p4.s1 < 0.0) ? 2.0f : 0.0; p4.s1 *= 128.0f;\n"
			"	f.s0 = (float)((((int)(p1.s1)) << 16) >> 16); f.s1 = (float)((((int)(p2.s1)) << 16) >> 16); p4.s2 = atan2pi(f.s1, f.s0); p4.s2 += (p4.s2 < 0.0) ? 2.0f : 0.0; p4.s2 *= 128.0f;\n"
			"	f.s0 = (float)(( (int)(p1.s1))        >> 16); f.s1 = (float)(( (int)(p2.s1))        >> 16); p4.s3 = atan2pi(f.s1, f.s0); p4.s3 += (p4.s3 < 0.0) ? 2.0f : 0.0; p4.s3 *= 128.0f;\n"
			"	p4 = select(p4, (float4) 0.0f, p4 > 255.5f);\n"
			"	r.s0 = amd_pack(p4);\n"
			"	f.s0 = (float)((((int)(p1.s2)) << 16) >> 16); f.s1 = (float)((((int)(p2.s2)) << 16) >> 16); p4.s0 = atan2pi(f.s1, f.s0); p4.s0 += (p4.s0 < 0.0) ? 2.0f : 0.0; p4.s0 *= 128.0f;\n"
			"	f.s0 = (float)(( (int)(p1.s2))        >> 16); f.s1 = (float)(( (int)(p2.s2))        >> 16); p4.s1 = atan2pi(f.s1, f.s0); p4.s1 += (p4.s1 < 0.0) ? 2.0f : 0.0; p4.s1 *= 128.0f;\n"
			"	f.s0 = (float)((((int)(p1.s3)) << 16) >> 16); f.s1 = (float)((((int)(p2.s3)) << 16) >> 16); p4.s2 = atan2pi(f.s1, f.s0); p4.s2 += (p4.s2 < 0.0) ? 2.0f : 0.0; p4.s2 *= 128.0f;\n"
			"	f.s0 = (float)(( (int)(p1.s3))        >> 16); f.s1 = (float)(( (int)(p2.s3))        >> 16); p4.s3 = atan2pi(f.s1, f.s0); p4.s3 += (p4.s3 < 0.0) ? 2.0f : 0.0; p4.s3 *= 128.0f;\n"
			"	p4 = select(p4, (float4) 0.0f, p4 > 255.5f);\n"
			"	r.s1 = amd_pack(p4);\n"
			"	*p0 = r;\n"
			"}\n"
			), node->opencl_name, node->opencl_name);
		node->opencl_code += item;
		if (node->akernel->id == VX_KERNEL_AMD_SOBEL_MAGNITUDE_PHASE_S16U8_U8_3x3) {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"void %s(S16x8 * mag, U8x8 * phase, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				"  S16x8 gx, gy;\n"
				"  %s_GXY(&gx, &gy, x, y, lbuf, p, stride); // LinearFilter_ANYx2_U8\n"
				"  Magnitude_S16_S16S16(mag, gx, gy);\n"
				"  Phase_U8_S16S16(phase, gx, gy);\n"
				"}\n"
				), node->opencl_name, node->opencl_name);
			node->opencl_param_discard_mask = 0;
			node->opencl_local_buffer_usage_mask = (2 << 1);
		}
		else if (node->akernel->id == VX_KERNEL_AMD_SOBEL_MAGNITUDE_S16_U8_3x3) {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"void %s(S16x8 * mag, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				"  S16x8 gx, gy;\n"
				"  %s_GXY(&gx, &gy, x, y, lbuf, p, stride); // LinearFilter_ANYx2_U8\n"
				"  Magnitude_S16_S16S16(mag, gx, gy);\n"
				"}\n"
				), node->opencl_name, node->opencl_name);
			node->opencl_param_discard_mask = 0;
			node->opencl_local_buffer_usage_mask = (1 << 1);
		}
		else if (node->akernel->id == VX_KERNEL_AMD_SOBEL_PHASE_U8_U8_3x3) {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"void %s(U8x8 * phase, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				"  S16x8 gx, gy;\n"
				"  %s_GXY(&gx, &gy, x, y, lbuf, p, stride); // LinearFilter_ANYx2_U8\n"
				"  Phase_U8_S16S16(phase, gx, gy);\n"
				"}\n"
				), node->opencl_name, node->opencl_name);
			node->opencl_param_discard_mask = 0;
			node->opencl_local_buffer_usage_mask = (1 << 1);
		}
		else {
			agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_SobelSpecialCases doesn't support kernel %s\n", node->akernel->name);
			return -1;
		}
		node->opencl_code += item;
		node->opencl_code += OPENCL_FORMAT(
			"#undef Magnitude_S16_S16S16\n"
			"#undef Phase_U8_S16S16\n"
			);
	}

	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following canny sobel filter kernels:
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L1NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L2NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L1NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L2NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L1NORM
//   VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L2NORM
//
int HafGpu_CannySobelFilters(AgoNode * node)
{
	int status = VX_SUCCESS;

	// re-use LinearFilter_ANYx2_U8 for computing GX & GY
	char opencl_name[VX_MAX_KERNEL_NAME];
	strcpy(opencl_name, node->opencl_name);
	snprintf(node->opencl_name, sizeof(node->opencl_name), "%s_GXY", opencl_name);
	AgoData filterGX, filterGY;
	filterGX.ref.type = VX_TYPE_MATRIX; filterGX.u.mat.type = VX_TYPE_FLOAT32; filterGX.ref.read_only = true;
	filterGY.ref.type = VX_TYPE_MATRIX; filterGY.u.mat.type = VX_TYPE_FLOAT32; filterGY.ref.read_only = true;
	int N = 0;
	if (node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L1NORM || node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L2NORM) {
		filterGX.u.mat.columns = filterGX.u.mat.rows = 3; filterGX.buffer = (vx_uint8 *)&sobelFilter_3x3_x[0][0];
		filterGY.u.mat.columns = filterGY.u.mat.rows = 3; filterGY.buffer = (vx_uint8 *)&sobelFilter_3x3_y[0][0];
		N = 3;
	}
	else if (node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L1NORM || node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L2NORM) {
		filterGX.u.mat.columns = filterGX.u.mat.rows = 5; filterGX.buffer = (vx_uint8 *)&sobelFilter_5x5_x[0][0];
		filterGY.u.mat.columns = filterGY.u.mat.rows = 5; filterGY.buffer = (vx_uint8 *)&sobelFilter_5x5_y[0][0];
		N = 5;
	}
	else if (node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L1NORM || node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L2NORM) {
		filterGX.u.mat.columns = filterGX.u.mat.rows = 7; filterGX.buffer = (vx_uint8 *)&sobelFilter_7x7_x[0][0];
		filterGY.u.mat.columns = filterGY.u.mat.rows = 7; filterGY.buffer = (vx_uint8 *)&sobelFilter_7x7_y[0][0];
		N = 7;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_CannySobelFilters doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}
	status = HafGpu_LinearFilter_ANYx2_U8(node, VX_DF_IMAGE_F32_AMD, &filterGX, &filterGY, false);
	strcpy(node->opencl_name, opencl_name);
	if (status) {
		return status;
	}
	node->opencl_param_discard_mask = 0;
	node->opencl_local_buffer_usage_mask = (1 << 1);

	// actual function using pre-defined functions
	char item[8192];
	if (node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L1NORM ||
		node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L1NORM ||
		node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L1NORM)
	{ // L1NORM
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"uint CannyMagPhase(float gx, float gy) {\n"
			"  float dx = fabs(gx), dy = fabs(gy);\n"
			"  float dr = amd_min3((dx + dy)%s, 16383.0f, 16383.0f);\n" // magnitude /= 4 for gradient_size = 7
			"  float d1 = dx * 0.4142135623730950488016887242097f;\n"
			"  float d2 = dx * 2.4142135623730950488016887242097f;\n"
			"  uint mp = select(1u, 3u, (gx * gy) < 0.0f);\n"
			"       mp = select(mp, 0u, dy <= d1);\n"
			"       mp = select(mp, 2u, dy >= d2);\n"
			"  mp += (((uint)dr) << 2);\n"
			"  return mp;\n"
			"}\n")
			, node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L1NORM ? "*0.25f" : "");
		node->opencl_code += item;
	}
	else
	{ // L2NORM
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"uint CannyMagPhase(float gx, float gy) {\n"
			"  float dx = fabs(gx), dy = fabs(gy);\n"
			"  float dr = amd_min3(native_sqrt(mad(gy, gy, gx * gx)%s), 16383.0f, 16383.0f);\n" // magnitude /= (4^2) for gradient_size = 7
			"  float d1 = dx * 0.4142135623730950488016887242097f;\n"
			"  float d2 = dx * 2.4142135623730950488016887242097f;\n"
			"  uint mp = select(1u, 3u, (gx * gy) < 0.0f);\n"
			"       mp = select(mp, 0u, dy <= d1);\n"
			"       mp = select(mp, 2u, dy >= d2);\n"
			"  mp += (((uint)dr) << 2);\n"
			"  return mp;\n"
			"}\n")
			, node->akernel->id == VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L2NORM ? "*0.0625f" : ""); 
		node->opencl_code += item;
	}
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"void %s(U16x8 * magphase, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
		"  F32x8 gx, gy;\n"
		"  %s_GXY(&gx, &gy, x, y, lbuf, p, stride); // LinearFilter_ANYx2_U8\n"
		"  uint mask = select(0xffffu, 0u, y < %d); mask = select(0u, mask, y < %d);\n" // (N >> 1), height - (N >> 1)
		"  U16x8 r; uint mp;\n"
		"  mp = CannyMagPhase(gx.s0, gy.s0) & mask; mp = select(mp, 0u, (int)x < %d);                           r.s0  =  mp;\n"         // (N>>1)-0
		"  mp = CannyMagPhase(gx.s1, gy.s1) & mask; mp = select(mp, 0u, (int)x < %d);                           r.s0 |= (mp << 16);\n"  // (N>>1)-1
		"  mp = CannyMagPhase(gx.s2, gy.s2) & mask; mp = select(mp, 0u, (int)x < %d);                           r.s1  =  mp;\n"         // (N > 5) ? (N>>1)-2 : 0
		"  mp = CannyMagPhase(gx.s3, gy.s3) & mask;                                                             r.s1 |= (mp << 16);\n"  // 
		"  mp = CannyMagPhase(gx.s4, gy.s4) & mask;                                                             r.s2  =  mp;\n"         // 
		"  mp = CannyMagPhase(gx.s5, gy.s5) & mask;                               mp = select(0u, mp, x < %du); r.s2 |= (mp << 16);\n"  //           width-(N>>1)-5
		"  mp = CannyMagPhase(gx.s6, gy.s6) & mask;                               mp = select(0u, mp, x < %du); r.s3  =  mp;\n"         //           width-(N>>1)-6
		"  mp = CannyMagPhase(gx.s7, gy.s7) & mask;                               mp = select(0u, mp, x < %du); r.s3 |= (mp << 16);\n"  //           width-(N>>1)-7
		"  *magphase = r;\n"
		"}\n"
		)
		, node->opencl_name, node->opencl_name, (N >> 1), height - (N >> 1), (N >> 1) - 0, (N >> 1) - 1, (N > 5) ? ((N >> 1) - 2) : 0, width - (N >> 1) - 5, width - (N >> 1) - 6, width - (N >> 1) - 7);
	node->opencl_code += item;

	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following canny non-max supression filter kernels:
//   VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8_U16_3x3
//   VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_3x3
//
int HafGpu_CannySuppThreshold(AgoNode * node)
{
	int status = VX_SUCCESS;
	// configuration
	int work_group_width = 16;
	int work_group_height = 16;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;

	// local memory usage
	int LMemSideLR = 4;
	int LMemStride = work_group_width * (4 * 2) + 2 * 2 * 2;
	int LMemSize = LMemStride * (work_group_height + 2);

	// kernel declaration
	char item[8192];
	const char * xyarg = "";
	int ioffset = 1;
	if (node->akernel->id == VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_3x3 || node->akernel->id == VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_7x7) {
		xyarg = "__global char * p1_buf, uint p1_offset, uint p1_count, ";
		ioffset = 2;
	}
	else if (node->akernel->id != VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8_U16_3x3) {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_CannySuppThreshold doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset, %suint p2_width, uint p2_height, __global uchar * p2_buf, uint p2_stride, uint p2_offset, uint2 p3)\n" // xyarg
		"{\n"
		"  __local uchar lbuf[%d];\n" // LMemSize
		"  int lx = get_local_id(0);\n"
		"  int ly = get_local_id(1);\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  bool valid = (gx < %d) && (gy < %d);\n" // (width+3)/4, height
		"  p0_buf += p0_offset + (gy * p0_stride) + (gx << 2);\n"
		"  p2_buf += p2_offset;\n"
		"  int gstride = p2_stride;\n"
		"  __global uchar * gbuf = p2_buf;\n"
		)
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, xyarg, LMemSize, (width + 3) / 4, height);
	node->opencl_code = item;
	// load U16 into local
	if (HafGpu_Load_Local(work_group_width, work_group_height, LMemStride, work_group_height + 2, 2 * 2, 1, node->opencl_code) < 0) {
		return -1;
	}
	// load U16 pixels from local and perform non-max supression
	vx_uint32 gradient_size = node->paramList[ioffset+2] ? node->paramList[ioffset+2]->u.scalar.u.u : 3;
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"  __local uchar * lbuf_ptr = lbuf + ly * %d + (lx << 3);\n" // LMemStride
		"  uint4 L0 = vload4(0, (__local uint *) lbuf_ptr);\n"
		"  uint4 L1 = vload4(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride
		"  uint4 L2 = vload4(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride * 2
		"  uint3 NA, NB, NC; uint T, M1, M2; uint4 M;\n"
		"  NA.s0 =         L0.s0  >> 18 ; NA.s1 =         L1.s0  >> 18 ; NA.s2 =         L2.s0  >> 18 ;\n"
		"  NB.s0 = amd_bfe(L0.s1, 2, 14); NB.s1 = amd_bfe(L1.s1, 2, 14); NB.s2 = amd_bfe(L2.s1, 2, 14);\n"
		"  NC.s0 =         L0.s1  >> 18 ; NC.s1 =         L1.s1  >> 18 ; NC.s2 =         L2.s1  >> 18 ;\n"
		"  T = amd_bfe(L1.s1,  0, 2); M1 = select(NA.s1, NA.s0, T > 0); M1 = select(M1, NB.s0, T > 1); M1 = select(M1, NA.s2, T > 2); M2 = select(NC.s1, NC.s2+1, T > 0); M2 = select(M2, NB.s2, T > 1); M2 = select(M2, NC.s0+1, T > 2); M.s0 = select(0u, NB.s1, NB.s1 > M1); M.s0 = select(0u, M.s0, NB.s1 >= M2);\n"
		"  NA.s0 = amd_bfe(L0.s2, 2, 14); NA.s1 = amd_bfe(L1.s2, 2, 14); NA.s2 = amd_bfe(L2.s2, 2, 14);\n"
		"  T = amd_bfe(L1.s1, 16, 2); M1 = select(NB.s1, NB.s0, T > 0); M1 = select(M1, NC.s0, T > 1); M1 = select(M1, NB.s2, T > 2); M2 = select(NA.s1, NA.s2+1, T > 0); M2 = select(M2, NC.s2, T > 1); M2 = select(M2, NA.s0+1, T > 2); M.s1 = select(0u, NC.s1, NC.s1 > M1); M.s1 = select(0u, M.s1, NC.s1 >= M2);\n"
		"  NB.s0 =         L0.s2  >> 18 ; NB.s1 =         L1.s2  >> 18 ; NB.s2 =         L2.s2  >> 18 ;\n"
		"  T = amd_bfe(L1.s2,  0, 2); M1 = select(NC.s1, NC.s0, T > 0); M1 = select(M1, NA.s0, T > 1); M1 = select(M1, NC.s2, T > 2); M2 = select(NB.s1, NB.s2+1, T > 0); M2 = select(M2, NA.s2, T > 1); M2 = select(M2, NB.s0+1, T > 2); M.s2 = select(0u, NA.s1, NA.s1 > M1); M.s2 = select(0u, M.s2, NA.s1 >= M2);\n"
		"  NC.s0 = amd_bfe(L0.s3, 2, 14); NC.s1 = amd_bfe(L1.s3, 2, 14); NC.s2 = amd_bfe(L2.s3, 2, 14);\n"
		"  T = amd_bfe(L1.s2, 16, 2); M1 = select(NA.s1, NA.s0, T > 0); M1 = select(M1, NB.s0, T > 1); M1 = select(M1, NA.s2, T > 2); M2 = select(NC.s1, NC.s2+1, T > 0); M2 = select(M2, NB.s2, T > 1); M2 = select(M2, NC.s0+1, T > 2); M.s3 = select(0u, NB.s1, NB.s1 > M1); M.s3 = select(0u, M.s3, NB.s1 >= M2);\n"
		"  uint mask = select(0u, 0xffffffffu, gx < %du); mask = select(0u, mask, gy < %du);\n" // (width+3)/4, height
		"  M.s0 &= mask;\n"
		"  M.s1 &= mask;\n"
		"  M.s2 &= mask;\n"
		"  M.s3 &= mask;\n"
		"  uint4 P;\n"
		"%s" // THRESHOLD /= 2 when gradient_size = 7
		"  P.s0 = select(  0u, 127u, M.s0 > p3.s0);\n"
		"  P.s1 = select(  0u, 127u, M.s1 > p3.s0);\n"
		"  P.s2 = select(  0u, 127u, M.s2 > p3.s0);\n"
		"  P.s3 = select(  0u, 127u, M.s3 > p3.s0);\n"
		"  P.s0 = select(P.s0, 255u, M.s0 > p3.s1);\n"
		"  P.s1 = select(P.s1, 255u, M.s1 > p3.s1);\n"
		"  P.s2 = select(P.s2, 255u, M.s2 > p3.s1);\n"
		"  P.s3 = select(P.s3, 255u, M.s3 > p3.s1);\n"
		"  uint p0 = P.s0;\n"
		"  p0 += P.s1 << 8;\n"
		"  p0 += P.s2 << 16;\n"
		"  p0 += P.s3 << 24;\n"
		"  if (valid)  *(__global uint *)p0_buf = p0;\n"
		)
		, LMemStride, LMemStride, LMemStride * 2, (width + 3) / 4, height, (gradient_size == 7) ? "  p3.s0 = p3.s0 >> 2; p3.s1 = p3.s1 >> 2;\n" : "");
	node->opencl_code += item;
	if (node->akernel->id == VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_3x3 || node->akernel->id == VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_7x7) {
		node->opencl_code +=
			OPENCL_FORMAT(
			"  if (valid) {\n"
			"    uint stack_icount;\n"
			"    stack_icount  = select(0u, 1u, P.s0 == 255u);\n"
			"    stack_icount += select(0u, 1u, P.s1 == 255u);\n"
			"    stack_icount += select(0u, 1u, P.s2 == 255u);\n"
			"    stack_icount += select(0u, 1u, P.s3 == 255u);\n"
			"    if (stack_icount > 0) {\n"
			"      uint pos = atomic_add((__global uint *)p1_buf, stack_icount);\n"
			"      __global uint * p1_buf_ptr = (__global uint *)&p1_buf[p1_offset];\n"
			"      uint xyloc = (gy << 16) + (gx << 2);\n"
			"      if(pos < p1_count && P.s0 == 255u) p1_buf_ptr[pos++] = xyloc;\n"
			"      if(pos < p1_count && P.s1 == 255u) p1_buf_ptr[pos++] = xyloc+1;\n"
			"      if(pos < p1_count && P.s2 == 255u) p1_buf_ptr[pos++] = xyloc+2;\n"
			"      if(pos < p1_count && P.s3 == 255u) p1_buf_ptr[pos++] = xyloc+3;\n"
			"    }\n"
			"  }\n"
			);
	}
	node->opencl_code += "}\n";

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_param_discard_mask = 1 << 4;
	node->opencl_param_atomic_mask = (ioffset > 1) ? (1 << 1) : 0;
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
// Generate OpenCL code for the following harris sobel filter kernels:
//   VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_3x3
//   VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_5x5
//   VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_7x7
//
int HafGpu_HarrisSobelFilters(AgoNode * node)
{
	int status = VX_SUCCESS;
	// configuration
	int N = 0;
	if (node->akernel->id == VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_3x3) N = 3;
	else if (node->akernel->id == VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_5x5) N = 5;
	else if (node->akernel->id == VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_7x7) N = 7;
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_HarrisSobelFilters doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}
	int work_group_width = 16;
	int work_group_height = 16;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;

	// use completely separate kernel
	node->opencl_work_dim = 2;
	node->opencl_global_work[0] = (((width + 7) >> 3) + work_group_width - 1) & ~(work_group_width - 1);
	node->opencl_global_work[1] = (height + work_group_height - 1) & ~(work_group_height - 1);
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;

	// headers
	char item[8192];
	node->opencl_code =
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"typedef float8 F32x8;\n"
		);

	// re-use LinearFilter_ANYx2_U8 for computing GX & GY
	snprintf(node->opencl_name, sizeof(node->opencl_name), "LinearFilter_ANYx2_U8");
	AgoData filterGX, filterGY;
	filterGX.ref.type = VX_TYPE_MATRIX; filterGX.u.mat.type = VX_TYPE_FLOAT32; filterGX.ref.read_only = true;
	filterGY.ref.type = VX_TYPE_MATRIX; filterGY.u.mat.type = VX_TYPE_FLOAT32; filterGY.ref.read_only = true;
	if (N == 3) {
		filterGX.u.mat.columns = filterGX.u.mat.rows = 3; filterGX.buffer = (vx_uint8 *)&sobelFilter_3x3_x[0][0];
		filterGY.u.mat.columns = filterGY.u.mat.rows = 3; filterGY.buffer = (vx_uint8 *)&sobelFilter_3x3_y[0][0];
	}
	else if (N == 5) {
		filterGX.u.mat.columns = filterGX.u.mat.rows = 5; filterGX.buffer = (vx_uint8 *)&sobelFilter_5x5_x[0][0];
		filterGY.u.mat.columns = filterGY.u.mat.rows = 5; filterGY.buffer = (vx_uint8 *)&sobelFilter_5x5_y[0][0];
	}
	else if (N == 7) {
		filterGX.u.mat.columns = filterGX.u.mat.rows = 7; filterGX.buffer = (vx_uint8 *)&sobelFilter_7x7_x[0][0];
		filterGY.u.mat.columns = filterGY.u.mat.rows = 7; filterGY.buffer = (vx_uint8 *)&sobelFilter_7x7_y[0][0];
	}
	status = HafGpu_LinearFilter_ANYx2_U8(node, VX_DF_IMAGE_F32_AMD, &filterGX, &filterGY, false);
	if (status) {
		return status;
	}

	// kernel body
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset, uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset)\n"
		"{\n"
		"  uint x = get_global_id(0) << 3;\n"
		"  uint y = get_global_id(1);\n"
		"  __local uchar lbuf[%d];\n"
		"  F32x8 gx, gy;\n"
		"  LinearFilter_ANYx2_U8(&gx, &gy, x, y, lbuf, p1_buf + p1_offset, p1_stride); // LinearFilter_ANYx2_U8\n"
		"  if ((x < %d) && (y < %d)) {\n" // width, height
		"    p0_buf += p0_offset + y * p0_stride + (x << 2);\n"
		"    vstore8(gx * gx, 0, (__global float *)&p0_buf[0]);\n"
		"    vstore8(gx * gy, 0, (__global float *)&p0_buf[%d]);\n" // width * 4
		"    vstore8(gy * gy, 0, (__global float *)&p0_buf[%d]);\n" // width * 4 * 2
		"  }\n"
		"}\n"
		)
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, node->opencl_local_buffer_size_in_bytes, width, height, width * 4, width * 4 * 2);
	node->opencl_code += item;

	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_param_discard_mask = 0;
	node->opencl_local_buffer_usage_mask = (1 << 1);

	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following harris score filter kernels:
//   VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_3x3
//   VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_5x5
//   VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_7x7
//
int HafGpu_HarrisScoreFilters(AgoNode * node)
{
	int status = VX_SUCCESS;
	// configuration
	int N = 0;
	if (node->akernel->id == VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_3x3) N = 3;
	else if (node->akernel->id == VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_5x5) N = 5;
	else if (node->akernel->id == VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_7x7) N = 7;
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_HarrisScoreFilters doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}
	int work_group_width = 16;
	int work_group_height = 16;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;
	vx_float32 sensitivity = node->paramList[2]->u.scalar.u.f;
	vx_float32 strength_threshold = node->paramList[3]->u.scalar.u.f;
	vx_int32 gradient_size = node->paramList[4]->u.scalar.u.i;
	vx_float32 normFactor = 255.0f * (1 << (gradient_size - 1)) * N;

	// local memory usage
	int kO = (N >> 1) & 1;
	int LMemSideLR = (N >> 1) + kO;
	int LMemStride = (16 * 4 + LMemSideLR * 2) * 4;
	int LMemSize = LMemStride * (16 + N - 1);

	// kernel declaration
	char item[8192];
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"typedef float8 F32x8;\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset, uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset, float p2, float p3)\n"
		"{\n"
		"  __local uchar lbuf[%d];\n"
		"  int lx = get_local_id(0);\n"
		"  int ly = get_local_id(1);\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  int gstride = p1_stride;\n"
		"  p0_buf += p0_offset + (gy * p0_stride) + (gx << 4); p1_buf += p1_offset;\n"
		)
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, LMemSize);
	node->opencl_code = item;

	for (int component = 0; component < 3; component++) {
		// load component into LDS
		if (component == 0) {
			snprintf(item, sizeof(item), "  __global uchar * gbuf = p1_buf; __local uchar * lbuf_ptr; float2 v2;\n");
		}
		else {
			snprintf(item, sizeof(item),
				"  barrier(CLK_LOCAL_MEM_FENCE);\n"
				"  gbuf = p1_buf + %d;\n" // width * 4 * component
				, width * 4 * component);
		}
		node->opencl_code += item;
		if (HafGpu_Load_Local(work_group_width, work_group_height, LMemStride, 16 + N - 1, LMemSideLR * 4, (N >> 1), node->opencl_code) < 0) {
			return -1;
		}
		// horizontal sum
		snprintf(item, sizeof(item),
			"  float4 sum%d;\n" // component
			"  lbuf_ptr = &lbuf[ly * %d + (lx << 4)];\n" // LMemStride
			, component, LMemStride);
		node->opencl_code += item;
		for (int i = 0; i < 2 + (N >> 1); i++) {
			if (kO) {
				snprintf(item, sizeof(item), "  v2 = vload2(0, (__local float *)&lbuf_ptr[%d]);\n", (i * 2 + kO) * 4);
			}
			else {
				snprintf(item, sizeof(item), "  v2 = *(__local float2 *)&lbuf_ptr[%d];\n", (i * 2) * 4);
			}
			node->opencl_code += item;
			for (int k = i*2; k < i*2+2; k++) {
				for (int j = max(k-N+1,0); j <= min(k,3); j++) {
					snprintf(item, sizeof(item), "  sum%d.s%d %c= v2.s%d;\n", component, j, (k == j) ? ' ' : '+', k & 1);
					node->opencl_code += item;
				}
			}
		}
		snprintf(item, sizeof(item), "  *(__local float4 *)lbuf_ptr = sum%d;\n", component);
		node->opencl_code += item;
		snprintf(item, sizeof(item), "  if (ly < %d) {\n", N - 1);
		node->opencl_code += item;
		for (int i = 0; i < 2 + (N >> 1); i++) {
			if (kO) {
				snprintf(item, sizeof(item), "    v2 = vload2(0, (__local float *)&lbuf_ptr[%d]);\n", LMemStride * work_group_height + (i * 2 + kO) * 4);
			}
			else {
				snprintf(item, sizeof(item), "    v2 = *(__local float2 *)&lbuf_ptr[%d];\n", LMemStride * work_group_height + (i * 2) * 4);
			}
			node->opencl_code += item;
			for (int k = i * 2; k < i * 2 + 2; k++) {
				for (int j = max(k-N+1, 0); j <= min(k, 3); j++) {
					snprintf(item, sizeof(item), "    sum%d.s%d %c= v2.s%d;\n", component, j, (k == j) ? ' ' : '+', k & 1);
					node->opencl_code += item;
				}
			}
		}
		snprintf(item, sizeof(item), 
			"    *(__local float4 *)&lbuf_ptr[%d] = sum%d;\n"
			"  }\n"
			"  barrier(CLK_LOCAL_MEM_FENCE);\n"
			, LMemStride * work_group_height, component);
		node->opencl_code += item;
		// vertical sum
		snprintf(item, sizeof(item), "  sum%d = *(__local float4 *)lbuf_ptr;\n", component);
		node->opencl_code += item;
		for (int i = 1; i < N; i++) {
			snprintf(item, sizeof(item), "  sum%d += *(__local float4 *)&lbuf_ptr[%d];\n", component, i * LMemStride);
			node->opencl_code += item;
		}
	}

	int border = (gradient_size >> 1) + (N >> 1);
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"  gx = gx << 2;\n"
		"  if ((gx < %d) && (gy < %d)) {\n" // width, height
		"    float4 score = (float4)0.0f;\n"
		"    if ((gy >= %d) && (gy < %d)) {\n" // border, height - border
		"      score = sum0 * sum2 - sum1 * sum1;\n"
		"      sum0 += sum2;\n"
		"      sum0 *= sum0;\n"
		"      score = mad(sum0, (float4)-p2, score);\n"
		"      score *= (float4)%.12ef;\n" // (1/normFactor)^4
		"      score = select((float4)0.0f, score, score > (float4)p3);\n"
		"      score.s0 = select(score.s0, 0.0f, gx < %d);\n" // border
		"      score.s1 = select(score.s1, 0.0f, gx < %d);\n" // border-1
		"      score.s2 = select(score.s2, 0.0f, gx < %d);\n" // border-2
		"      score.s3 = select(score.s3, 0.0f, gx < %d);\n" // border-3
		"      score.s0 = select(score.s0, 0.0f, gx > %d);\n" // width-1-border
		"      score.s1 = select(score.s1, 0.0f, gx > %d);\n" // width-2-border
		"      score.s2 = select(score.s2, 0.0f, gx > %d);\n" // width-3-border
		"      score.s3 = select(score.s3, 0.0f, gx > %d);\n" // width-4-border
		"    }\n"
		"    *(__global float4 *)p0_buf = score;\n"
		"  }\n"
		"}\n"
		),
		width, height, border, height - border, (float)(1.0 / (normFactor*normFactor*normFactor*normFactor)),
		border, border - 1, border - 2, border - 3, width - 1 - border, width - 2 - border, width - 3 - border, width - 4 - border);
	node->opencl_code += item;

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_param_discard_mask = (1 << 4);
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
// Generate OpenCL code for the following non-max supression filter kernels:
//   VX_KERNEL_AMD_NON_MAX_SUPP_XY_ANY_3x3
//
int HafGpu_NonMaxSupp_XY_ANY_3x3(AgoNode * node)
{
	int status = VX_SUCCESS;
	// configuration
	int work_group_width = 16;
	int work_group_height = 16;
	int width = node->paramList[1]->u.img.width;
	int height = node->paramList[1]->u.img.height;

	// local memory usage
	int LMemSideLR = 1 * 4;
	int LMemSideTB = 1;
	int LMemStride = work_group_width * 2 * 4 + LMemSideLR * 2;
	int LMemSize = LMemStride * (work_group_height + 2);

	// kernel declaration
	char item[8192];
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(__global char * p0_buf, uint p0_offset, uint p0_count, uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset)\n"
		"{\n"
		"  int lx = get_local_id(0);\n"
		"  int ly = get_local_id(1);\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  int gstride = p1_stride;\n"
		"  __global uchar * gbuf = p1_buf + p1_offset;\n"
		"  __local uchar lbuf[%d];\n" // LMemSize
		)
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, LMemSize);
	node->opencl_code = item;
	// load into local
	if (HafGpu_Load_Local(work_group_width, work_group_height, LMemStride, work_group_height + 2, LMemSideLR, LMemSideTB, node->opencl_code) < 0) {
		return -1;
	}
	// load pixels from local and perform non-max supression
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"  __local uchar * lbuf_ptr = lbuf + ly * %d + (lx << 3);\n" // LMemStride
		"  float4 L0 = vload4(0, (__local float *) lbuf_ptr);\n"
		"  float4 L1 = vload4(0, (__local float *)&lbuf_ptr[%d]);\n" // LMemStride
		"  float4 L2 = vload4(0, (__local float *)&lbuf_ptr[%d]);\n" // LMemStride * 2
		"  float2 T = L1.s12;\n"
		"  T.s0 = select(0.0f, T.s0, T.s0 >= L1.s0);\n"
		"  T.s0 = select(0.0f, T.s0, T.s0 >  L1.s2);\n"
		"  T.s0 = select(0.0f, T.s0, T.s0 >= L0.s0);\n"
		"  T.s0 = select(0.0f, T.s0, T.s0 >= L0.s1);\n"
		"  T.s0 = select(0.0f, T.s0, T.s0 >= L0.s2);\n"
		"  T.s0 = select(0.0f, T.s0, T.s0 >  L2.s0);\n"
		"  T.s0 = select(0.0f, T.s0, T.s0 >  L2.s1);\n"
		"  T.s0 = select(0.0f, T.s0, T.s0 >  L2.s2);\n"
		"  T.s1 = select(0.0f, T.s1, T.s1 >= L1.s1);\n"
		"  T.s1 = select(0.0f, T.s1, T.s1 >  L1.s3);\n"
		"  T.s1 = select(0.0f, T.s1, T.s1 >= L0.s1);\n"
		"  T.s1 = select(0.0f, T.s1, T.s1 >= L0.s2);\n"
		"  T.s1 = select(0.0f, T.s1, T.s1 >= L0.s3);\n"
		"  T.s1 = select(0.0f, T.s1, T.s1 >  L2.s1);\n"
		"  T.s1 = select(0.0f, T.s1, T.s1 >  L2.s2);\n"
		"  T.s1 = select(0.0f, T.s1, T.s1 >  L2.s3);\n"
		"  T.s0 = select(0.0f, T.s0, gx < %d);\n" // (width+1)/2
		"  T.s1 = select(0.0f, T.s1, gx < %d);\n" // width/2
		"  T.s0 = select(0.0f, T.s0, gy < %d);\n" // height
		"  T.s1 = select(0.0f, T.s1, gy < %d);\n" // height
		"  gx = gx + gx + select(0, 1, T.s1 > 0.0f);\n"
		"  T.s0 = select(T.s0, T.s1, T.s1 > 0.0f);\n"
		"  if (T.s0 > 0.0f) {\n"
		"    uint pos = atomic_inc((__global uint *)p0_buf);\n"
		"    if(pos < p0_count) {\n"
		"      *(__global uint2 *)&p0_buf[p0_offset + (pos << 3)] = (uint2)(gx | (gy << 16), as_uint(T.s0));\n"
		"    }\n"
		"  }\n"
		"}\n"
		)
		, LMemStride, LMemStride, LMemStride * 2, (width + 1) / 2, width / 2, height, height);
	node->opencl_code += item;

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_param_discard_mask = 0;
	node->opencl_param_atomic_mask = (1 << 0);
	node->opencl_work_dim = 2;
	node->opencl_global_work[0] = (((width + 1)/2) + work_group_width - 1) & ~(work_group_width - 1);
	node->opencl_global_work[1] = (height + work_group_height - 1) & ~(work_group_height - 1);
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;

	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for the following half scale gaussian filters:
//   VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_3x3
//   VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_5x5
//
int HafGpu_ScaleGaussianHalf(AgoNode * node)
{
	int status = VX_SUCCESS;
	// configuration
	int work_group_width = 16;
	int work_group_height = 16;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height - 1;
	int N = 0;
	if (node->akernel->id == VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_3x3) {
		N = 3;
	}
	else if (node->akernel->id == VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_5x5) {
		N = 5;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_ScaleGaussianHalf doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}

	// local memory usage
	int LMemSideLR = ((N >> 1) + 3) & ~3;
	int LMemSideTB =  (N >> 1);
	int LMemStride = work_group_width * 8 + LMemSideLR * 2;
	int LMemSize = LMemStride * (work_group_height * 2 - 1 + LMemSideTB * 2);

	// kernel declaration
	char item[8192];
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset, uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset)\n"
		"{\n"
		"  __local uchar lbuf[%d];\n" // LMemSize
		"  int lx = get_local_id(0);\n"
		"  int ly = get_local_id(1);\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  p0_buf += p0_offset + (gy * p0_stride) + (gx << 2);\n"
		"  int gstride = p1_stride;\n"
		"  __global uchar * gbuf = p1_buf + p1_offset + (((gy - ly) << 1) + 1) * gstride + ((gx - lx) << 3);\n"
		"  bool valid = ((gx < %d) && (gy < %d)) ? true : false;\n" // (width+3)/4, height
		"  gx = lx; gy = ly;\n"
		)
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, LMemSize, (width + 3) / 4, height);
	node->opencl_code = item;
	int srcImageBufferSize = (int) node->paramList[1]->size;
	// load input image into local
	if (HafGpu_Load_Local_ImageSize(work_group_width, work_group_height, LMemStride, work_group_height * 2 - 1 + LMemSideTB * 2, LMemSideLR, LMemSideTB, node->opencl_code, srcImageBufferSize) < 0) {
		return -1;
	}
	// perform filtering
	if (N == 3) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  __local uchar * lbuf_ptr = lbuf + ly * %d + (lx << 3);\n" // LMemStride * 2
			"  uint3 L0 = vload3(0, (__local uint *)&lbuf_ptr[4]);\n"
			"  uint3 L1 = vload3(0, (__local uint *)&lbuf_ptr[%d+4]);\n" // LMemStride
			"  uint3 L2 = vload3(0, (__local uint *)&lbuf_ptr[%d+4]);\n" // LMemStride * 2
			"  float4 sum; float v;\n"
			"  v = amd_unpack0(L0.s0); v = mad(amd_unpack0(L1.s0), 2.0f, v); v += amd_unpack0(L2.s0); sum.s0 = v;\n"
			"  v = amd_unpack1(L0.s0); v = mad(amd_unpack1(L1.s0), 2.0f, v); v += amd_unpack1(L2.s0); sum.s0 = mad(v, 2.0f, sum.s0);\n"
			"  v = amd_unpack2(L0.s0); v = mad(amd_unpack2(L1.s0), 2.0f, v); v += amd_unpack2(L2.s0); sum.s1 = v; sum.s0 += v;\n"
			"  v = amd_unpack3(L0.s0); v = mad(amd_unpack3(L1.s0), 2.0f, v); v += amd_unpack3(L2.s0); sum.s1 = mad(v, 2.0f, sum.s1);\n"
			"  v = amd_unpack0(L0.s1); v = mad(amd_unpack0(L1.s1), 2.0f, v); v += amd_unpack0(L2.s1); sum.s2 = v; sum.s1 += v;\n"
			"  v = amd_unpack1(L0.s1); v = mad(amd_unpack1(L1.s1), 2.0f, v); v += amd_unpack1(L2.s1); sum.s2 = mad(v, 2.0f, sum.s2);\n"
			"  v = amd_unpack2(L0.s1); v = mad(amd_unpack2(L1.s1), 2.0f, v); v += amd_unpack2(L2.s1); sum.s3 = v; sum.s2 += v;\n"
			"  v = amd_unpack3(L0.s1); v = mad(amd_unpack3(L1.s1), 2.0f, v); v += amd_unpack3(L2.s1); sum.s3 = mad(v, 2.0f, sum.s3);\n"
			"  v = amd_unpack0(L0.s2); v = mad(amd_unpack0(L1.s2), 2.0f, v); v += amd_unpack0(L2.s2); sum.s3 += v;\n"
			"  sum = sum * (float4)0.0625f;\n"
			"  if (valid) {;\n"
			"    *(__global uint *)p0_buf = amd_pack(sum);\n"
			"  }\n"
			"}\n"
			)
			, LMemStride * 2, LMemStride, LMemStride * 2);
		node->opencl_code += item;
	}
	else if (N == 5) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  __local uchar * lbuf_ptr = lbuf + ly * %d + (lx << 3);\n" // LMemStride
			"  float4 sum; float v;\n"
			"  uint4 L0 = vload4(0, (__local uint *) lbuf_ptr);\n"
			"  v = amd_unpack3(L0.s0);                                             sum.s0 = v;\n"
			"  v = amd_unpack0(L0.s1);                                             sum.s0 = mad(v, 4.0f, sum.s0);\n"
			"  v = amd_unpack1(L0.s1);              sum.s0 = mad(v, 6.0f, sum.s0); sum.s1 = v;\n"
			"  v = amd_unpack2(L0.s1);              sum.s0 = mad(v, 4.0f, sum.s0); sum.s1 = mad(v, 4.0f, sum.s1);\n"
			"  v = amd_unpack3(L0.s1); sum.s0 += v; sum.s1 = mad(v, 6.0f, sum.s1); sum.s2 = v;\n"
			"  v = amd_unpack0(L0.s2);              sum.s1 = mad(v, 4.0f, sum.s1); sum.s2 = mad(v, 4.0f, sum.s2);\n"
			"  v = amd_unpack1(L0.s2); sum.s1 += v; sum.s2 = mad(v, 6.0f, sum.s2); sum.s3 = v;\n"
			"  v = amd_unpack2(L0.s2);              sum.s2 = mad(v, 4.0f, sum.s2); sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"  v = amd_unpack3(L0.s2); sum.s2 += v; sum.s3 = mad(v, 6.0f, sum.s3);\n"
			"  v = amd_unpack0(L0.s3);              sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"  v = amd_unpack1(L0.s3); sum.s3 += v;\n"
			"  L0.s0 = (uint)sum.s0 + (((uint)sum.s1) << 16);\n"
			"  L0.s1 = (uint)sum.s2 + (((uint)sum.s3) << 16);\n"
			"  *(__local uint2 *)lbuf_ptr = L0.s01;\n"
			"  L0 = vload4(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride*16
			"  v = amd_unpack3(L0.s0);                                             sum.s0 = v;\n"
			"  v = amd_unpack0(L0.s1);                                             sum.s0 = mad(v, 4.0f, sum.s0);\n"
			"  v = amd_unpack1(L0.s1);              sum.s0 = mad(v, 6.0f, sum.s0); sum.s1 = v;\n"
			"  v = amd_unpack2(L0.s1);              sum.s0 = mad(v, 4.0f, sum.s0); sum.s1 = mad(v, 4.0f, sum.s1);\n"
			"  v = amd_unpack3(L0.s1); sum.s0 += v; sum.s1 = mad(v, 6.0f, sum.s1); sum.s2 = v;\n"
			"  v = amd_unpack0(L0.s2);              sum.s1 = mad(v, 4.0f, sum.s1); sum.s2 = mad(v, 4.0f, sum.s2);\n"
			"  v = amd_unpack1(L0.s2); sum.s1 += v; sum.s2 = mad(v, 6.0f, sum.s2); sum.s3 = v;\n"
			"  v = amd_unpack2(L0.s2);              sum.s2 = mad(v, 4.0f, sum.s2); sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"  v = amd_unpack3(L0.s2); sum.s2 += v; sum.s3 = mad(v, 6.0f, sum.s3);\n"
			"  v = amd_unpack0(L0.s3);              sum.s3 = mad(v, 4.0f, sum.s3);\n"
			"  v = amd_unpack1(L0.s3); sum.s3 += v;\n"
			"  L0.s0 = (uint)sum.s0 + (((uint)sum.s1) << 16);\n"
			"  L0.s1 = (uint)sum.s2 + (((uint)sum.s3) << 16);\n"
			"  *(__local uint2 *)&lbuf_ptr[%d] = L0.s01;\n" // LMemStride*16
			"  if (ly < 3) {\n"
			"    L0 = vload4(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride*32
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
			"    *(__local uint2 *)&lbuf_ptr[%d] = L0.s01;\n" // LMemStride*32
			"  }\n"
			"  barrier(CLK_LOCAL_MEM_FENCE);\n"
			"  lbuf_ptr += ly * %d;\n" // LMemStride
			"  L0.s01 = vload2(0, (__local uint *) lbuf_ptr);\n"
			"  sum.s0 = (float)(L0.s0 & 0xffff); sum.s1 = (float)(L0.s0 >> 16); sum.s2 = (float)(L0.s1 & 0xffff); sum.s3 = (float)(L0.s1 >> 16);\n"
			"  L0.s01 = vload2(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride
			"  sum.s0 = mad((float)(L0.s0 & 0xffff), 4.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 4.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 4.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 4.0f, sum.s3);\n"
			"  L0.s01 = vload2(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride * 2
			"  sum.s0 = mad((float)(L0.s0 & 0xffff), 6.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 6.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 6.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 6.0f, sum.s3);\n"
			"  L0.s01 = vload2(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride * 3
			"  sum.s0 = mad((float)(L0.s0 & 0xffff), 4.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 4.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 4.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 4.0f, sum.s3);\n"
			"  L0.s01 = vload2(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride * 4
			"  sum.s0 += (float)(L0.s0 & 0xffff); sum.s1 += (float)(L0.s0 >> 16); sum.s2 += (float)(L0.s1 & 0xffff); sum.s3 += (float)(L0.s1 >> 16);\n"
			"  sum = sum * (float4)0.00390625f;\n"
			"  if (valid) {\n"
			"    *(__global uint *)p0_buf = amd_pack(sum);\n"
			"  }\n"
			"}\n"
			)
			, LMemStride, LMemStride * 16, LMemStride * 16, LMemStride * 32, LMemStride * 32, LMemStride, LMemStride, LMemStride * 2, LMemStride * 3, LMemStride * 4);
		node->opencl_code += item;
	}

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_param_discard_mask = 0;
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
// Generate OpenCL code for the following gaussian scale filters:
//   VX_KERNEL_AMD_SCALE_GAUSSIAN_ORB_U8_U8_5x5 (interpolation = VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR)
//
int HafGpu_ScaleGaussianOrb(AgoNode * node, vx_interpolation_type_e interpolation)
{
	int status = VX_SUCCESS;
	// configuration
	int work_group_width = 16;
	int work_group_height = 16;
	int width = node->paramList[0]->u.img.width;
	int height = node->paramList[0]->u.img.height;
	float xscale = (float)node->paramList[1]->u.img.width / (float)width, xoffset = xscale * 0.5f;
	float yscale = (float)node->paramList[1]->u.img.height / (float)height, yoffset = yscale * 0.5f;
	int N = 0;
	if (node->akernel->id == VX_KERNEL_AMD_SCALE_GAUSSIAN_ORB_U8_U8_5x5) {
		N = 5;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_ScaleGaussian doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}

	// local memory usage
	int LMemStride = 128;
	int LMemHeight = 19 + N - 1;
	int LMemSize = LMemStride * LMemHeight;

	// kernel declaration
	char item[8192];
	snprintf(item, sizeof(item),
		OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
		"void %s(uint p0_width, uint p0_height, __global uchar * p0_buf, uint p0_stride, uint p0_offset, uint p1_width, uint p1_height, __global uchar * p1_buf, uint p1_stride, uint p1_offset)\n"
		"{\n"
		"  __local uchar lbuf[%d];\n" // LMemSize
		"  int lx = get_local_id(0);\n"
		"  int ly = get_local_id(1);\n"
		"  int gx = get_global_id(0);\n"
		"  int gy = get_global_id(1);\n"
		"  bool outputValid = ((gx < %d) && (gy < %d)) ? true : false;\n" // (width+3)/4, height
		"  p0_buf += p0_offset + (gy * p0_stride) + (gx << 2);\n"
		"  int gstride = p1_stride;\n"
		"  float fx =  mad((float)(gx - lx), %.12ef, %.12ef);\n" // xscale * 4, xoffset
		"  float fy =  mad((float)(gy - ly), %.12ef, %.12ef);\n" // yscale, yoffset
		"  gx = (uint)fx; fx -= (float)gx;\n"
		"  gy = (uint)fy; fy -= (float)gy;\n"
		"  gx = gx - 2 + 4;\n"
		"  gy = gy - 2;\n"
		"  uint lxalign = gx & 3;\n"
		"  __global uchar * gbuf = p1_buf + p1_offset + (gx & ~3) + gy * gstride;\n"
		"  gx = lx; gy = ly;\n"
		)
		, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME, LMemSize, (width + 3) / 4, height, xscale * 4.0f, xoffset, yscale, yoffset);
	node->opencl_code = item;
	// load input image into local
	if (HafGpu_Load_Local(work_group_width, work_group_height, LMemStride, LMemHeight, 4, 0, node->opencl_code) < 0) {
		return -1;
	}
	// perform filtering
	if (N == 5) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  __local uchar * lbuf_ptr = lbuf + ly * %d;\n" // LMemStride
			"  float flx = mad((float)(lx << 2), %.12ef, fx + (float)lxalign);\n" // xscale
			"  uint2 L0, isum; float fsum; uint ilx;\n"
			"  ilx = (uint)flx; L0 = vload2(0, (__local uint *)&lbuf_ptr[ilx & ~3]); L0.s0 = amd_bytealign(L0.s1, L0.s0, ilx); L0.s1 = amd_bytealign(L0.s1, L0.s1, ilx);\n"
			"  fsum = amd_unpack0(L0.s0); fsum = mad(amd_unpack1(L0.s0), 4.0f, fsum); fsum = mad(amd_unpack2(L0.s0), 6.0f, fsum); fsum = mad(amd_unpack3(L0.s0), 4.0f, fsum); fsum += amd_unpack0(L0.s1);\n"
			"  isum.s0 = (uint)fsum;\n"
			"  ilx = (uint)(flx + %.12ef); L0 = vload2(0, (__local uint *)&lbuf_ptr[ilx & ~3]); L0.s0 = amd_bytealign(L0.s1, L0.s0, ilx); L0.s1 = amd_bytealign(L0.s1, L0.s1, ilx);\n" // xscale
			"  fsum = amd_unpack0(L0.s0); fsum = mad(amd_unpack1(L0.s0), 4.0f, fsum); fsum = mad(amd_unpack2(L0.s0), 6.0f, fsum); fsum = mad(amd_unpack3(L0.s0), 4.0f, fsum); fsum += amd_unpack0(L0.s1);\n"
			"  isum.s0 |= (((uint)fsum) << 16);\n"
			"  ilx = (uint)(flx + %.12ef); L0 = vload2(0, (__local uint *)&lbuf_ptr[ilx & ~3]); L0.s0 = amd_bytealign(L0.s1, L0.s0, ilx); L0.s1 = amd_bytealign(L0.s1, L0.s1, ilx);\n" // xscale * 2
			"  fsum = amd_unpack0(L0.s0); fsum = mad(amd_unpack1(L0.s0), 4.0f, fsum); fsum = mad(amd_unpack2(L0.s0), 6.0f, fsum); fsum = mad(amd_unpack3(L0.s0), 4.0f, fsum); fsum += amd_unpack0(L0.s1);\n"
			"  isum.s1 = (uint)fsum;\n"
			"  ilx = (uint)(flx + %.12ef); L0 = vload2(0, (__local uint *)&lbuf_ptr[ilx & ~3]); L0.s0 = amd_bytealign(L0.s1, L0.s0, ilx); L0.s1 = amd_bytealign(L0.s1, L0.s1, ilx);\n" // xscale * 3
			"  fsum = amd_unpack0(L0.s0); fsum = mad(amd_unpack1(L0.s0), 4.0f, fsum); fsum = mad(amd_unpack2(L0.s0), 6.0f, fsum); fsum = mad(amd_unpack3(L0.s0), 4.0f, fsum); fsum += amd_unpack0(L0.s1);\n"
			"  isum.s1 |= (((uint)fsum) << 16);\n"
			"  ((__local uint2 *)lbuf_ptr)[lx] = isum;\n"
			"  if (ly < 7) {\n"
			"    lbuf_ptr += %d;\n" // LMemStride * 16
			"    ilx = (uint)flx; L0 = vload2(0, (__local uint *)&lbuf_ptr[ilx & ~3]); L0.s0 = amd_bytealign(L0.s1, L0.s0, ilx); L0.s1 = amd_bytealign(L0.s1, L0.s1, ilx);\n"
			"    fsum = amd_unpack0(L0.s0); fsum = mad(amd_unpack1(L0.s0), 4.0f, fsum); fsum = mad(amd_unpack2(L0.s0), 6.0f, fsum); fsum = mad(amd_unpack3(L0.s0), 4.0f, fsum); fsum += amd_unpack0(L0.s1);\n"
			"    isum.s0 = (uint)fsum;\n"
			"    ilx = (uint)(flx + %.12ef); L0 = vload2(0, (__local uint *)&lbuf_ptr[ilx & ~3]); L0.s0 = amd_bytealign(L0.s1, L0.s0, ilx); L0.s1 = amd_bytealign(L0.s1, L0.s1, ilx);\n" // xscale
			"    fsum = amd_unpack0(L0.s0); fsum = mad(amd_unpack1(L0.s0), 4.0f, fsum); fsum = mad(amd_unpack2(L0.s0), 6.0f, fsum); fsum = mad(amd_unpack3(L0.s0), 4.0f, fsum); fsum += amd_unpack0(L0.s1);\n"
			"    isum.s0 |= (((uint)fsum) << 16);\n"
			"    ilx = (uint)(flx + %.12ef); L0 = vload2(0, (__local uint *)&lbuf_ptr[ilx & ~3]); L0.s0 = amd_bytealign(L0.s1, L0.s0, ilx); L0.s1 = amd_bytealign(L0.s1, L0.s1, ilx);\n" // xscale * 2
			"    fsum = amd_unpack0(L0.s0); fsum = mad(amd_unpack1(L0.s0), 4.0f, fsum); fsum = mad(amd_unpack2(L0.s0), 6.0f, fsum); fsum = mad(amd_unpack3(L0.s0), 4.0f, fsum); fsum += amd_unpack0(L0.s1);\n"
			"    isum.s1 = (uint)fsum;\n"
			"    ilx = (uint)(flx + %.12ef); L0 = vload2(0, (__local uint *)&lbuf_ptr[ilx & ~3]); L0.s0 = amd_bytealign(L0.s1, L0.s0, ilx); L0.s1 = amd_bytealign(L0.s1, L0.s1, ilx);\n" // xscale * 3
			"    fsum = amd_unpack0(L0.s0); fsum = mad(amd_unpack1(L0.s0), 4.0f, fsum); fsum = mad(amd_unpack2(L0.s0), 6.0f, fsum); fsum = mad(amd_unpack3(L0.s0), 4.0f, fsum); fsum += amd_unpack0(L0.s1);\n"
			"    isum.s1 |= (((uint)fsum) << 16);\n"
			"    ((__local uint2 *)lbuf_ptr)[lx] = isum;\n"
			"    lbuf_ptr -= %d;\n" // LMemStride * 16
			"  }\n"
			"  barrier(CLK_LOCAL_MEM_FENCE);\n"
			"  float fly = fy + (float)ly * %.12ef; float4 sum;\n" // yscale
			"  lbuf_ptr = lbuf + (uint)fly * %d + (lx << 3);\n" // LMemStride
			"  L0 = vload2(0, (__local uint *) lbuf_ptr);\n"
			"  sum.s0 = (float)(L0.s0 & 0xffff); sum.s1 = (float)(L0.s0 >> 16); sum.s2 = (float)(L0.s1 & 0xffff); sum.s3 = (float)(L0.s1 >> 16);\n"
			"  L0 = vload2(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride
			"  sum.s0 = mad((float)(L0.s0 & 0xffff), 4.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 4.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 4.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 4.0f, sum.s3);\n"
			"  L0 = vload2(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride * 2
			"  sum.s0 = mad((float)(L0.s0 & 0xffff), 6.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 6.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 6.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 6.0f, sum.s3);\n"
			"  L0 = vload2(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride * 3
			"  sum.s0 = mad((float)(L0.s0 & 0xffff), 4.0f, sum.s0); sum.s1 = mad((float)(L0.s0 >> 16), 4.0f, sum.s1); sum.s2 = mad((float)(L0.s1 & 0xffff), 4.0f, sum.s2); sum.s3 = mad((float)(L0.s1 >> 16), 4.0f, sum.s3);\n"
			"  L0 = vload2(0, (__local uint *)&lbuf_ptr[%d]);\n" // LMemStride * 4
			"  sum.s0 += (float)(L0.s0 & 0xffff); sum.s1 += (float)(L0.s0 >> 16); sum.s2 += (float)(L0.s1 & 0xffff); sum.s3 += (float)(L0.s1 >> 16);\n"
			"  sum = sum * (float4)0.00390625f;\n"
			"  if (outputValid) {;\n"
			"    *(__global uint *)p0_buf = amd_pack(sum);\n"
			"  }\n"
			"}\n"
			)
			, LMemStride, xscale, xscale, xscale * 2.0f, xscale * 3.0f, 
			  LMemStride * 16, xscale, xscale * 2.0f, xscale * 3.0f, LMemStride * 16, 
			  yscale, LMemStride, LMemStride, LMemStride * 2, LMemStride * 3, LMemStride * 4);
		node->opencl_code += item;
	}

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_param_discard_mask = 0;
	node->opencl_work_dim = 2;
	node->opencl_global_work[0] = (((width + 3) >> 2) + work_group_width - 1) & ~(work_group_width - 1);
	node->opencl_global_work[1] = (height + work_group_height - 1) & ~(work_group_height - 1);
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;

	return status;
}

#endif