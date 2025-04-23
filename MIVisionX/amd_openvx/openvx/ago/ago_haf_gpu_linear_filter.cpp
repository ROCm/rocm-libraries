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
// Generate OpenCL code for LinearFilter_U8_U8, LinearFilter_S16_U8, and LinearFilter_F32_U8
//
int HafGpu_LinearFilter_ANY_U8(AgoNode * node, vx_df_image dst_image_format, AgoData * src_filter, bool roundingMode)
{
	int status = VX_SUCCESS;
	// get destination type
	const char * dstRegType = "U8";
	bool dstIsS16 = false;
	bool dstIsF32 = false;
	float roundingBias = roundingMode ? 0.0f : -0.49999f;
	if (dst_image_format == VX_DF_IMAGE_S16) {
		dstRegType = "S16";
		dstIsS16 = true;
		roundingBias = roundingMode ? 0.5f : 0.0f;
	}
	else if (dst_image_format == VX_DF_IMAGE_F32_AMD) {
		dstRegType = "F32";
		dstIsF32 = true;
		roundingBias = 0.0f;
	}
	else if (dst_image_format != VX_DF_IMAGE_U8) {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_U8 doesn't support non-U8/S16/F32 destinations for kernel %s\n", node->akernel->name);
		return -1;
	}
	// get filter details
	bool filterCoefAreConstants = src_filter->ref.read_only;
	vx_uint32 filterWidth = 0, filterHeight = 0;
	float * filterCoef = nullptr;
	if (src_filter->ref.type == VX_TYPE_CONVOLUTION) {
		filterWidth = (vx_uint32)src_filter->u.conv.columns;
		filterHeight = (vx_uint32)src_filter->u.conv.rows;
		filterCoef = (float *)src_filter->reserved;
	}
	else if (src_filter->ref.type == VX_TYPE_MATRIX) {
		filterWidth = (vx_uint32)src_filter->u.mat.columns;
		filterHeight = (vx_uint32)src_filter->u.mat.rows;
		filterCoef = (float *)src_filter->buffer;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_U8 doesn't expects vx_matrix or vx_convolution object for kernel %s\n", node->akernel->name);
		return -1;
	}
	bool clampNotNeeded = false;
	bool filterCoefAreIntegers = false;
	if (filterCoefAreConstants) {
		float sumP = 0.0f, sumN = 0.0f;
		filterCoefAreIntegers = true;
		for (vx_uint32 i = 0; i < filterWidth * filterHeight; i++) {
			if (floorf(filterCoef[i]) != filterCoef[i])
				filterCoefAreIntegers = false;
			if (filterCoef[i] < 0.0f) sumN += filterCoef[i];
			else sumP += filterCoef[i];
		}
		if (sumN*255.0f > -32767.0f && sumP*255.0f < 32766.0f)
			clampNotNeeded = true;
	}

	char item[1024];
	std::string code;
	if (filterHeight == 1 && filterWidth > 1) {
		// generate code for Mx1 filter
		vx_uint32 Mdiv2 = filterWidth >> 1; if (Mdiv2 == 0) { 
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_U8 doesn't support %dx%d filter\n", filterWidth, filterHeight);
			return -1; 
		}
		// function declaration
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_%dx1;\n"
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_%dx1 coef) {\n"
				), filterWidth, filterWidth, node->opencl_name, dstRegType, filterWidth);
		}
		code = item;

		// configuration
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;
		vx_uint32 LMemWidth = AGO_OPENCL_WORKGROUP_SIZE_0 * 8;
		vx_uint32 LMemSideAlign = (Mdiv2 < 8) ? 3 : 7;
		vx_uint32 LMemSide = ((Mdiv2 + LMemSideAlign) & ~LMemSideAlign);
		vx_uint32 LMemStride = LMemWidth + 2 * LMemSide;

		node->opencl_param_discard_mask = filterCoefAreConstants ? (1 << 2) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 1);
		node->opencl_local_buffer_size_in_bytes = LMemHeight * LMemStride;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemStride, LMemHeight, LMemSide, 0, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  F32x8 sum; uint2 pix; float fval;\n"
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			), LMemStride);
		code += item;
		int numQW = (LMemSide / 4) + 1;
		for (int qw = 0; qw < numQW; qw++) {
			bool loaded_pix = false;
			for (int x = 0; x < 8; x++) {
				int bytepos = qw * 8 + x;
				int xpos = bytepos - LMemSide;
				if (xpos >= -(int)Mdiv2 && xpos <= (7 + (int)Mdiv2)) {
					bool loaded_fval = false;
					for (int ix = 0; ix < 8; ix++) {
						int ixpos = xpos - ix;
						if (ixpos == -(int)Mdiv2) {
							if (filterCoefAreConstants) {
								if (filterCoef[0] == 0.0f) {
									snprintf(item, sizeof(item), "  sum.s%d = 0.0f;\n", ix);
								}
								else {
									if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
									if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
									snprintf(item, sizeof(item), "  sum.s%d =     fval* %.12ef;\n", ix, filterCoef[0]);
								}
							}
							else {
								if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
								if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
								snprintf(item, sizeof(item), "  sum.s%d =     fval* coef.f[ 0];\n", ix);
							}
							code += item;
						}
						else if ((ixpos > -(int)Mdiv2) && (ixpos <= (int)Mdiv2)) {
							if (filterCoefAreConstants) {
								if (filterCoef[ixpos + Mdiv2] != 0.0f) {
									if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
									if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
									snprintf(item, sizeof(item), "  sum.s%d = mad(fval, %.12ef, sum.s%d);\n", ix, filterCoef[ixpos + Mdiv2], ix);
									code += item;
								}
							}
							else {
								if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
								if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
								snprintf(item, sizeof(item), "  sum.s%d = mad(fval, coef.f[%2d], sum.s%d);\n", ix, ixpos + Mdiv2, ix);
								code += item;
							}
						}
					}
				}
			}
		}
	}
	else if (filterWidth == 1) {
		// generate code for Mx1 filter
		vx_uint32 Ndiv2 = filterHeight >> 1;
		// function declaration
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_1x%d;\n"
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_1x%d coef) {\n"
				), filterHeight, filterHeight, node->opencl_name, dstRegType, filterHeight);
		}
		code = item;

		// configuration
		vx_uint32 LMemWidth = AGO_OPENCL_WORKGROUP_SIZE_0 * 8;
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;

		node->opencl_param_discard_mask = filterCoefAreConstants ? (1 << 2) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 1);
		node->opencl_local_buffer_size_in_bytes = (LMemHeight + 2 * Ndiv2) * LMemWidth;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemWidth, LMemHeight + Ndiv2 * 2, 0, Ndiv2, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  F32x8 sum; uint2 pix; float fval;\n"
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			), LMemWidth);
		code += item;

		bool first_item = true;
		for (int y = 0; y < (int)filterHeight; y++) {
			if (!filterCoefAreConstants || filterCoef[y] != 0.0f) {
				snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", y * LMemWidth / 8); code += item;
				if (filterCoefAreConstants) {
					snprintf(item, sizeof(item), "  fval = %.12ef;\n", filterCoef[y]); code += item;
				}
				else {
					snprintf(item, sizeof(item), "  fval = coef.f[%d];\n", y); code += item;
				}
				if (first_item) {
					first_item = false;
					for (int x = 0; x < 8; x++) {
						snprintf(item, sizeof(item), "  sum.s%d = amd_unpack%d(pix.s%d) * fval;\n", x, x & 3, x >> 2); code += item;
					}
				}
				else {
					for (int x = 0; x < 8; x++) {
						snprintf(item, sizeof(item), "  sum.s%d = mad(amd_unpack%d(pix.s%d), fval, sum.s%d);\n", x, x & 3, x >> 2, x); code += item;
					}
				}
			}
		}
	}
	else {
		// generate code for MxN filter
		vx_uint32 Ndiv2 = filterHeight >> 1;
		vx_uint32 Mdiv2 = filterWidth >> 1;

		// function declaration
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_%dx%d;\n"
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_%dx%d coef) {\n"
				), filterWidth*filterHeight, filterWidth, filterHeight, node->opencl_name, dstRegType, filterWidth, filterHeight);
		}
		code = item;

		// configuration
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;
		vx_uint32 LMemWidth = AGO_OPENCL_WORKGROUP_SIZE_0 * 8;
		vx_uint32 LMemSideLR = ((Mdiv2 + 3) & ~3);
		vx_uint32 LMemStride = LMemWidth + 2 * LMemSideLR;
		vx_uint32 LMemSideTB = Ndiv2;
		vx_uint32 LMemSize = (LMemHeight + 2 * LMemSideTB) * LMemStride;
		node->opencl_param_discard_mask = filterCoefAreConstants ? (1 << 2) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 1);
		node->opencl_local_buffer_size_in_bytes = LMemSize;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemStride, LMemHeight + 2 * LMemSideTB, LMemSideLR, LMemSideTB, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item), 
			OPENCL_FORMAT(
			"  F32x8 sum = (F32x8)0.0f; uint2 pix; float fval;\n"
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			), LMemStride);
		code += item;
		int numQW = (LMemSideLR / 4) + 1;
		for (int y = 0; y < (int)filterHeight; y++) {
			snprintf(item, sizeof(item), "  // filterRow = %d\n", y); code += item;
			for (int qw = 0; qw < numQW; qw++) {
				bool loaded_pix = false;
				for (int x = 0; x < 8; x++) {
					int bytepos = qw * 8 + x;
					int xpos = bytepos - LMemSideLR;
					if (xpos >= -(int)Mdiv2 && xpos <= (7 + (int)Mdiv2)) {
						bool loaded_fval = false;
						for (int ix = 0; ix < 8; ix++) {
							int ixpos = xpos - ix;
							if ((ixpos >= -(int)Mdiv2) && (ixpos <= (int)Mdiv2)) {
								int coefPos = y * filterWidth + ixpos + Mdiv2;
								if (filterCoefAreConstants) {
									if (filterCoef[coefPos] != 0.0f) {
										if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw + y*LMemStride / 8); code += item; }
										if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
										if (filterCoef[coefPos] == 1.0f)       snprintf(item, sizeof(item), "  sum.s%d += fval;\n", ix);
										else if (filterCoef[coefPos] == -1.0f) snprintf(item, sizeof(item), "  sum.s%d -= fval;\n", ix);
										else                                   snprintf(item, sizeof(item), "  sum.s%d  = mad(fval, %.12ef, sum.s%d);\n", ix, filterCoef[coefPos], ix);
										code += item;
									}
								}
								else {
									if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw + y*LMemStride / 8); code += item; }
									if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
									snprintf(item, sizeof(item), "  sum.s%d = mad(fval, coef.f[%2d], sum.s%d);\n", ix, coefPos, ix);
									code += item;
								}
							}
						}
					}
				}
			}
		}
	}
	if (!filterCoefAreIntegers && roundingBias != 0.0f) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  sum.s0 = sum.s0 + %.12ef;\n"
			"  sum.s1 = sum.s1 + %.12ef;\n"
			"  sum.s2 = sum.s2 + %.12ef;\n"
			"  sum.s3 = sum.s3 + %.12ef;\n"
			"  sum.s4 = sum.s4 + %.12ef;\n"
			"  sum.s5 = sum.s5 + %.12ef;\n"
			"  sum.s6 = sum.s6 + %.12ef;\n"
			"  sum.s7 = sum.s7 + %.12ef;\n"
			), roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias);
		code += item;
	}
	if (dstIsS16) {
		if (clampNotNeeded) {
			code +=
				OPENCL_FORMAT(
				"  S16x8 rv;\n"
				"  rv.s0  = ((int)sum.s0) & 0xffff;\n"
				"  rv.s0 |= ((int)sum.s1) << 16;\n"
				"  rv.s1  = ((int)sum.s2) & 0xffff;\n"
				"  rv.s1 |= ((int)sum.s3) << 16;\n"
				"  rv.s2  = ((int)sum.s4) & 0xffff;\n"
				"  rv.s2 |= ((int)sum.s5) << 16;\n"
				"  rv.s3  = ((int)sum.s6) & 0xffff;\n"
				"  rv.s3 |= ((int)sum.s7) << 16;\n"
				"  *r = rv;\n"
				"}\n");
		}
		else {
			code +=
				OPENCL_FORMAT(
				"  S16x8 rv;\n"
				"  rv.s0  = ((int)clamp(sum.s0, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s0 |= ((int)clamp(sum.s1, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s1  = ((int)clamp(sum.s2, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s1 |= ((int)clamp(sum.s3, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s2  = ((int)clamp(sum.s4, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s2 |= ((int)clamp(sum.s5, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s3  = ((int)clamp(sum.s6, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s3 |= ((int)clamp(sum.s7, -32768.0f, 32767.0f)) << 16;\n"
				"  *r = rv;\n"
				"}\n");
		}
	}
	else if (dstIsF32) {
		code +=
			"  *r = sum;\n"
			"}\n";
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
// Generate OpenCL code for LinearFilter_U8_S16, LinearFilter_S16_S16, and LinearFilter_F32_S16
//
int HafGpu_LinearFilter_ANY_S16(AgoNode * node, vx_df_image dst_image_format, AgoData * src_filter, bool roundingMode)
{
	int status = VX_SUCCESS;
	// get destination type
	const char * dstRegType = "U8";
	bool dstIsS16 = false;
	bool dstIsF32 = false;
	float roundingBias = roundingMode ? 0.0f : -0.49999f;
	if (dst_image_format == VX_DF_IMAGE_S16) {
		dstRegType = "S16";
		dstIsS16 = true;
		roundingBias = roundingMode ? 0.5f : 0.0f;
	}
	else if (dst_image_format == VX_DF_IMAGE_F32_AMD) {
		dstRegType = "F32";
		dstIsF32 = true;
		roundingBias = 0.0f;
	}
	else if (dst_image_format != VX_DF_IMAGE_U8) {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_S16 doesn't support non-U8/S16/F32 destinations for kernel %s\n", node->akernel->name);
		return -1;
	}
	// get filter size
	bool filterCoefAreConstants = src_filter->ref.read_only;
	vx_uint32 filterWidth = 0, filterHeight = 0;
	float * filterCoef = nullptr;
	if (src_filter->ref.type == VX_TYPE_CONVOLUTION) {
		filterWidth = (vx_uint32)src_filter->u.conv.columns;
		filterHeight = (vx_uint32)src_filter->u.conv.rows;
		filterCoef = (float *)src_filter->reserved;
	}
	else if (src_filter->ref.type == VX_TYPE_MATRIX) {
		filterWidth = (vx_uint32)src_filter->u.mat.columns;
		filterHeight = (vx_uint32)src_filter->u.mat.rows;
		filterCoef = (float *)src_filter->buffer;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_S16 doesn't expects vx_matrix or vx_convolution object for kernel %s\n", node->akernel->name);
		return -1;
	}
	bool filterCoefAreIntegers = false;
	if (filterCoefAreConstants) {
		filterCoefAreIntegers = true;
		for (vx_uint32 i = 0; i < filterWidth * filterHeight; i++) {
			if (floorf(filterCoef[i]) != filterCoef[i])
				filterCoefAreIntegers = false;
		}
	}

	if (filterHeight == 1 && filterWidth > 1) {
		// generate code for Mx1 filter
		vx_uint32 Mdiv2 = filterWidth >> 1; if (Mdiv2 == 0) { 
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_S16 doesn't support %dx%d filter\n", filterWidth, filterHeight);
			return -1; 
		}
		vx_uint32 BytesPerPixel = (vx_uint32)sizeof(vx_int16);
		vx_uint32 LMemSidePixelAlign = 4;
		vx_uint32 BytesPerWorkItem = 8 * BytesPerPixel;
		vx_uint32 BytesPerPixelShift = leftmostbit(BytesPerPixel);
		vx_uint32 BytesPerWorkItemShift = leftmostbit(BytesPerWorkItem);
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;
		vx_uint32 LMemWidth = (AGO_OPENCL_WORKGROUP_SIZE_0 * 8) * BytesPerPixel;
		vx_uint32 LMemSide = ((Mdiv2 + (LMemSidePixelAlign - 1)) & ~(LMemSidePixelAlign - 1)) * BytesPerPixel;
		vx_uint32 LMemStride = LMemWidth + 2 * LMemSide;
		char item[1024];
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_%dx1;\n"
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_%dx1 coef) {\n"
				), filterWidth, filterWidth, node->opencl_name, dstRegType, filterWidth);
		}
		std::string code = item;

		node->opencl_param_discard_mask = filterCoefAreConstants ? (1 << 2) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 1);
		node->opencl_local_buffer_size_in_bytes = LMemHeight * LMemStride;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemStride, LMemHeight, LMemSide, 0, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  F32x8 sum; short4 pix; float fval;\n"
			"  __local short4 * lbufptr = (__local short4 *) (lbuf + ly * %d + (lx << 4));\n" // LMemStride
			), LMemStride);
		code += item;
		int numQF = 2 * (((2 * LMemSide) / BytesPerWorkItem) + 1);
		for (int qf = 0; qf < numQF; qf++) {
			bool loaded_pix = false;
			for (int x = 0; x < 4; x++) {
				int pixpos = qf * 4 + x;
				int xpos = pixpos - (LMemSide / BytesPerPixel);
				bool loaded_fval = false;
				for (int ix = 0; ix < 8; ix++) {
					int ixpos = xpos - ix;
					if (ixpos == -(int)Mdiv2) {
						if (filterCoefAreConstants) {
							if (filterCoef[0] == 0.0f) {
								if (dstIsS16) snprintf(item, sizeof(item), "  sum.s%d = 0.5f;\n", ix);
								else          snprintf(item, sizeof(item), "  sum.s%d = 0.0f;\n", ix);
							}
							else {
								if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qf);  code += item; }
								if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = (float)pix.s%d;\n", x);  code += item; }
								if (dstIsS16) snprintf(item, sizeof(item), "  sum.s%d = mad(fval, %.12ef, 0.5f);\n", ix, filterCoef[0]);
								else          snprintf(item, sizeof(item), "  sum.s%d =     fval* %.12ef;\n", ix, filterCoef[0]);
							}
						}
						else {
							if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qf);  code += item; }
							if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = (float)pix.s%d;\n", x);  code += item; }
							if (dstIsS16) snprintf(item, sizeof(item), "  sum.s%d = mad(fval, coef.f[ 0], 0.5f);\n", ix);
							else          snprintf(item, sizeof(item), "  sum.s%d =     fval* coef.f[ 0];\n", ix);
						}
						code += item;
					}
					else if ((ixpos > -(int)Mdiv2) && (ixpos <= (int)Mdiv2)) {
						if (filterCoefAreConstants) {
							if (filterCoef[ixpos + Mdiv2] != 0.0f) {
								if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qf);  code += item; }
								if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = (float)pix.s%d;\n", x);  code += item; }
								snprintf(item, sizeof(item), "  sum.s%d = mad(fval, %.12ef, sum.s%d);\n", ix, filterCoef[ixpos + Mdiv2], ix);
								code += item;
							}
						}
						else {
							if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qf);  code += item; }
							if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = (float)pix.s%d;\n", x);  code += item; }
							snprintf(item, sizeof(item), "  sum.s%d = mad(fval, coef.f[%2d], sum.s%d);\n", ix, ixpos + Mdiv2, ix);
							code += item;
						}
					}
				}
			}
		}
		if (!filterCoefAreIntegers && roundingBias != 0.0f) {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"  sum.s0 = sum.s0 + %.12ef;\n"
				"  sum.s1 = sum.s1 + %.12ef;\n"
				"  sum.s2 = sum.s2 + %.12ef;\n"
				"  sum.s3 = sum.s3 + %.12ef;\n"
				"  sum.s4 = sum.s4 + %.12ef;\n"
				"  sum.s5 = sum.s5 + %.12ef;\n"
				"  sum.s6 = sum.s6 + %.12ef;\n"
				"  sum.s7 = sum.s7 + %.12ef;\n"
				), roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias);
			code += item;
		}
		if (dstIsS16) {
			code +=
				OPENCL_FORMAT(
				"  S16x8 rv;\n"
				"  rv.s0  = ((int)clamp(sum.s0, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s0 |= ((int)clamp(sum.s1, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s1  = ((int)clamp(sum.s2, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s1 |= ((int)clamp(sum.s3, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s2  = ((int)clamp(sum.s4, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s2 |= ((int)clamp(sum.s5, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s3  = ((int)clamp(sum.s6, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s3 |= ((int)clamp(sum.s7, -32768.0f, 32767.0f)) << 16;\n"
				"  *r = rv;\n"
				"}\n");
		}
		else if (dstIsF32) {
			code +=
				"  *r = sum;\n"
				"}\n";
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
	}
	else if (filterWidth == 1) {
		// generate code for Mx1 filter
		vx_uint32 Ndiv2 = filterHeight >> 1;
		// function declaration
		char item[1024];
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_1x%d;\n"
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_1x%d coef) {\n"
				), filterHeight, filterHeight, node->opencl_name, dstRegType, filterHeight);
		}
		std::string code = item;

		// configuration
		vx_uint32 LMemWidth = AGO_OPENCL_WORKGROUP_SIZE_0 * 8 * 2;
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;

		node->opencl_param_discard_mask = filterCoefAreConstants ? (1 << 2) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 1);
		node->opencl_local_buffer_size_in_bytes = (LMemHeight + 2 * Ndiv2) * LMemWidth;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemWidth, LMemHeight + Ndiv2 * 2, 0, Ndiv2, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  F32x8 sum; short8 pix; float fval;\n"
			"  __local short8 * lbufptr = (__local short8 *) (lbuf + ly * %d + (lx << 4));\n" // LMemStride
			), LMemWidth);
		code += item;
		bool first_item = true;
		for (int y = 0; y < (int)filterHeight; y++) {
			if (!filterCoefAreConstants || filterCoef[y] != 0.0f) {
				snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", y * LMemWidth / 16); code += item;
				if (filterCoefAreConstants) {
					snprintf(item, sizeof(item), "  fval = %.12ef;\n", filterCoef[y]); code += item;
				}
				else {
					snprintf(item, sizeof(item), "  fval = coef.f[%d];\n", y); code += item;
				}
				if (first_item) {
					first_item = false;
					for (int x = 0; x < 8; x++) {
						if (dstIsS16) {
							snprintf(item, sizeof(item), "  sum.s%d = mad((float)pix.s%d, fval, 0.5f);\n", x, x); code += item;
						}
						else {
							snprintf(item, sizeof(item), "  sum.s%d = (float)pix.s%d * fval;\n", x, x); code += item;
						}
					}
				}
				else {
					for (int x = 0; x < 8; x++) {
						snprintf(item, sizeof(item), "  sum.s%d = mad((float)pix.s%d, fval, sum.s%d);\n", x, x, x); code += item;
					}
				}
			}
		}
		if (!filterCoefAreIntegers && roundingBias != 0.0f) {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"  sum.s0 = sum.s0 + %.12ef;\n"
				"  sum.s1 = sum.s1 + %.12ef;\n"
				"  sum.s2 = sum.s2 + %.12ef;\n"
				"  sum.s3 = sum.s3 + %.12ef;\n"
				"  sum.s4 = sum.s4 + %.12ef;\n"
				"  sum.s5 = sum.s5 + %.12ef;\n"
				"  sum.s6 = sum.s6 + %.12ef;\n"
				"  sum.s7 = sum.s7 + %.12ef;\n"
				), roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias);
			code += item;
		}
		if (dstIsS16) {
			code +=
				OPENCL_FORMAT(
				"  S16x8 rv;\n"
				"  rv.s0  = ((int)clamp(sum.s0, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s0 |= ((int)clamp(sum.s1, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s1  = ((int)clamp(sum.s2, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s1 |= ((int)clamp(sum.s3, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s2  = ((int)clamp(sum.s4, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s2 |= ((int)clamp(sum.s5, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s3  = ((int)clamp(sum.s6, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s3 |= ((int)clamp(sum.s7, -32768.0f, 32767.0f)) << 16;\n"
				"  *r = rv;\n"
				"}\n");
		}
		else if (dstIsF32) {
			code +=
				"  *r = sum;\n"
				"}\n";
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
	}
	else {
		status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	}
	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for LinearFilter_U8_F32, LinearFilter_S16_F32, and LinearFilter_F32_F32
//
int HafGpu_LinearFilter_ANY_F32(AgoNode * node, vx_df_image dst_image_format, AgoData * src_filter, bool roundingMode)
{
	int status = VX_SUCCESS;
	// get destination type
	const char * dstRegType = "U8";
	bool dstIsS16 = false;
	bool dstIsF32 = false;
	float roundingBias = roundingMode ? 0.0f : -0.49999f;
	if (dst_image_format == VX_DF_IMAGE_S16) {
		dstRegType = "S16";
		dstIsS16 = true;
		roundingBias = roundingMode ? 0.5f : 0.0f;
	}
	else if (dst_image_format == VX_DF_IMAGE_F32_AMD) {
		dstRegType = "F32";
		dstIsF32 = true;
		roundingBias = 0.0f;
	}
	else if (dst_image_format != VX_DF_IMAGE_U8) {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_F32 doesn't support non-U8/S16/F32 destinations for kernel %s\n", node->akernel->name);
		return -1;
	}
	// get filter size
	bool filterCoefAreConstants = src_filter->ref.read_only;
	vx_uint32 filterWidth = 0, filterHeight = 0;
	float * filterCoef = nullptr;
	if (src_filter->ref.type == VX_TYPE_CONVOLUTION) {
		filterWidth = (vx_uint32)src_filter->u.conv.columns;
		filterHeight = (vx_uint32)src_filter->u.conv.rows;
		filterCoef = (float *)src_filter->reserved;
	}
	else if (src_filter->ref.type == VX_TYPE_MATRIX) {
		filterWidth = (vx_uint32)src_filter->u.mat.columns;
		filterHeight = (vx_uint32)src_filter->u.mat.rows;
		filterCoef = (float *)src_filter->buffer;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_F32 doesn't expects vx_matrix or vx_convolution object for kernel %s\n", node->akernel->name);
		return -1;
	}
	if (filterHeight == 1 && filterWidth > 1) {
		// generate code for Mx1 filter
		vx_uint32 Mdiv2 = filterWidth >> 1; if (Mdiv2 == 0) { 
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_F32 doesn't support %dx%d filter\n", filterWidth, filterHeight);
			return -1; 
		}
		vx_uint32 BytesPerPixel = (vx_uint32)sizeof(vx_float32);
		vx_uint32 LMemSidePixelAlign = 4;
		vx_uint32 BytesPerWorkItem = 8 * BytesPerPixel;
		vx_uint32 BytesPerPixelShift = leftmostbit(BytesPerPixel);
		vx_uint32 BytesPerWorkItemShift = leftmostbit(BytesPerWorkItem);
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;
		vx_uint32 LMemWidth = (AGO_OPENCL_WORKGROUP_SIZE_0 * 8) * BytesPerPixel;
		vx_uint32 LMemSide = ((Mdiv2 + (LMemSidePixelAlign - 1)) & ~(LMemSidePixelAlign - 1)) * BytesPerPixel;
		vx_uint32 LMemStride = LMemWidth + 2 * LMemSide;
		char item[1024];
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_%dx1;\n"
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_%dx1 coef) {\n"
				), filterWidth, filterWidth, node->opencl_name, dstRegType, filterWidth);
		}
		std::string code = item;

		node->opencl_param_discard_mask = filterCoefAreConstants ? (1 << 2) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 1);
		node->opencl_local_buffer_size_in_bytes = LMemHeight * LMemStride;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemStride, LMemHeight, LMemSide, 0, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  F32x8 sum; float4 pix;\n"
			"  __local float4 * lbufptr = (__local float4 *) (lbuf + ly * %d + (lx << 5));\n" // LMemStride
			), LMemStride);
		code += item;
		int numQF = 2 * (((2 * LMemSide) / BytesPerWorkItem) + 1);
		for (int qf = 0; qf < numQF; qf++) {
			bool loaded_pix = false;
			for (int x = 0; x < 4; x++) {
				int pixpos = qf * 4 + x;
				int xpos = pixpos - (LMemSide / BytesPerPixel);
				for (int ix = 0; ix < 8; ix++) {
					int ixpos = xpos - ix;
					if (ixpos == -(int)Mdiv2) {
						if (filterCoefAreConstants) {
							if (filterCoef[0] == 0.0f) {
								if (dstIsS16) snprintf(item, sizeof(item), "  sum.s%d = 0.5f;\n", ix);
								else          snprintf(item, sizeof(item), "  sum.s%d = 0.0f;\n", ix);
							}
							else {
								if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qf); code += item; }
								if (dstIsS16) snprintf(item, sizeof(item), "  sum.s%d = mad(pix.s%d, %.12ef, 0.5f);\n", ix, x, filterCoef[0]);
								else          snprintf(item, sizeof(item), "  sum.s%d =     pix.s%d* %.12ef;\n", ix, x, filterCoef[0]);
							}
						}
						else {
							if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qf); code += item; }
							if (dstIsS16) snprintf(item, sizeof(item), "  sum.s%d = mad(pix.s%d, coef.f[ 0], 0.5f);\n", ix, x);
							else          snprintf(item, sizeof(item), "  sum.s%d =     pix.s%d* coef.f[ 0];\n", ix, x);
						}
						code += item;
					}
					else if ((ixpos > -(int)Mdiv2) && (ixpos <= (int)Mdiv2)) {
						if (filterCoefAreConstants) {
							if (filterCoef[ixpos + Mdiv2] != 0.0f) {
								if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qf); code += item; }
								snprintf(item, sizeof(item), "  sum.s%d = mad(pix.s%d, %.12ef, sum.s%d);\n", ix, x, filterCoef[ixpos + Mdiv2], ix);
								code += item;
							}
						}
						else {
							if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qf); code += item; }
							snprintf(item, sizeof(item), "  sum.s%d = mad(pix.s%d, coef.f[%2d], sum.s%d);\n", ix, x, ixpos + Mdiv2, ix);
							code += item;
						}
					}
				}
			}
		}
		if (roundingBias != 0.0f) {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"  sum.s0 = sum.s0 + %.12ef;\n"
				"  sum.s1 = sum.s1 + %.12ef;\n"
				"  sum.s2 = sum.s2 + %.12ef;\n"
				"  sum.s3 = sum.s3 + %.12ef;\n"
				"  sum.s4 = sum.s4 + %.12ef;\n"
				"  sum.s5 = sum.s5 + %.12ef;\n"
				"  sum.s6 = sum.s6 + %.12ef;\n"
				"  sum.s7 = sum.s7 + %.12ef;\n"
				), roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias);
			code += item;
		}
		if (dstIsS16) {
			code +=
				OPENCL_FORMAT(
				"  S16x8 rv;\n"
				"  rv.s0  = ((int)clamp(sum.s0, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s0 |= ((int)clamp(sum.s1, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s1  = ((int)clamp(sum.s2, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s1 |= ((int)clamp(sum.s3, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s2  = ((int)clamp(sum.s4, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s2 |= ((int)clamp(sum.s5, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s3  = ((int)clamp(sum.s6, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s3 |= ((int)clamp(sum.s7, -32768.0f, 32767.0f)) << 16;\n"
				"  *r = rv;\n"
				"}\n");
		}
		else if (dstIsF32) {
			code +=
				"  *r = sum;\n"
				"}\n";
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
	}
	else if (filterWidth == 1) {
		// generate code for Mx1 filter
		vx_uint32 Ndiv2 = filterHeight >> 1;
		// function declaration
		char item[1024];
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_1x%d;\n"
				"void %s(%sx8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_1x%d coef) {\n"
				), filterHeight, filterHeight, node->opencl_name, dstRegType, filterHeight);
		}
		std::string code = item;

		// configuration
		vx_uint32 LMemWidth = AGO_OPENCL_WORKGROUP_SIZE_0 * 8 * 4;
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;

		node->opencl_param_discard_mask = filterCoefAreConstants ? (1 << 2) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 1);
		node->opencl_local_buffer_size_in_bytes = (LMemHeight + 2 * Ndiv2) * LMemWidth;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemWidth, LMemHeight + Ndiv2 * 2, 0, Ndiv2, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  F32x8 sum; float8 pix; float fval;\n"
			"  __local float8 * lbufptr = (__local float8 *) (lbuf + ly * %d + (lx << 5));\n" // LMemStride
			), LMemWidth);
		code += item;
		bool first_item = true;
		for (int y = 0; y < (int)filterHeight; y++) {
			if (!filterCoefAreConstants || filterCoef[y] != 0.0f) {
				snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", y * LMemWidth / 32); code += item;
				if (filterCoefAreConstants) {
					snprintf(item, sizeof(item), "  fval = %.12ef;\n", filterCoef[y]); code += item;
				}
				else {
					snprintf(item, sizeof(item), "  fval = coef.f[%d];\n", y); code += item;
				}
				if (first_item) {
					first_item = false;
					for (int x = 0; x < 8; x++) {
						if (dstIsS16) {
							snprintf(item, sizeof(item), "  sum.s%d = mad(pix.s%d, fval, 0.5f);\n", x, x); code += item;
						}
						else {
							snprintf(item, sizeof(item), "  sum.s%d = pix.s%d * fval;\n", x, x); code += item;
						}
					}
				}
				else {
					for (int x = 0; x < 8; x++) {
						snprintf(item, sizeof(item), "  sum.s%d = mad(pix.s%d, fval, sum.s%d);\n", x, x, x); code += item;
					}
				}
			}
		}
		if (roundingBias != 0.0f) {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"  sum.s0 = sum.s0 + %.12ef;\n"
				"  sum.s1 = sum.s1 + %.12ef;\n"
				"  sum.s2 = sum.s2 + %.12ef;\n"
				"  sum.s3 = sum.s3 + %.12ef;\n"
				"  sum.s4 = sum.s4 + %.12ef;\n"
				"  sum.s5 = sum.s5 + %.12ef;\n"
				"  sum.s6 = sum.s6 + %.12ef;\n"
				"  sum.s7 = sum.s7 + %.12ef;\n"
				), roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias);
			code += item;
		}
		if (dstIsS16) {
			code +=
				OPENCL_FORMAT(
				"  S16x8 rv;\n"
				"  rv.s0  = ((int)clamp(sum.s0, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s0 |= ((int)clamp(sum.s1, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s1  = ((int)clamp(sum.s2, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s1 |= ((int)clamp(sum.s3, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s2  = ((int)clamp(sum.s4, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s2 |= ((int)clamp(sum.s5, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s3  = ((int)clamp(sum.s6, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s3 |= ((int)clamp(sum.s7, -32768.0f, 32767.0f)) << 16;\n"
				"  *r = rv;\n"
				"}\n");
		}
		else if (dstIsF32) {
			code +=
				"  *r = sum;\n"
				"}\n";
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
	}
	else {
		status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	}
	return status;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generate OpenCL code for LinearFilter_U8x2_U8, LinearFilter_S16x2_U8, and LinearFilter_F32x2_U8
//
int HafGpu_LinearFilter_ANYx2_U8(AgoNode * node, vx_df_image dst_image_format, AgoData * src_filter, AgoData * src_filter2, bool roundingMode)
{
	int status = VX_SUCCESS;
	// get destination type
	const char * dstRegType = "U8";
	bool dstIsS16 = false;
	bool dstIsF32 = false;
	float roundingBias = roundingMode ? 0.0f : -0.49999f;
	if (dst_image_format == VX_DF_IMAGE_S16) {
		dstRegType = "S16";
		dstIsS16 = true;
		roundingBias = roundingMode ? 0.5f : 0.0f;
	}
	else if (dst_image_format == VX_DF_IMAGE_F32_AMD) {
		dstRegType = "F32";
		dstIsF32 = true;
		roundingBias = 0.0f;
	}
	else if (dst_image_format != VX_DF_IMAGE_U8) {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANYx2_U8 doesn't support non-U8/S16/F32 destinations for kernel %s\n", node->akernel->name);
		return -1;
	}
	// get filter size
	bool filterCoefAreConstants = src_filter->ref.read_only;
	vx_uint32 filterWidth = 0, filterHeight = 0, filter2Width = 0, filter2Height = 0;
	float * filterCoef = nullptr, *filter2Coef = nullptr;
	if (src_filter->ref.type == VX_TYPE_CONVOLUTION) {
		filterWidth = (vx_uint32)src_filter->u.conv.columns;
		filterHeight = (vx_uint32)src_filter->u.conv.rows;
		filterCoef = (float *)src_filter->reserved;
		filter2Width = (vx_uint32)src_filter2->u.conv.columns;
		filter2Height = (vx_uint32)src_filter2->u.conv.rows;
		filter2Coef = (float *)src_filter2->reserved;
	}
	else if (src_filter->ref.type == VX_TYPE_MATRIX) {
		filterWidth = (vx_uint32)src_filter->u.mat.columns;
		filterHeight = (vx_uint32)src_filter->u.mat.rows;
		filterCoef = (float *)src_filter->buffer;
		filter2Width = (vx_uint32)src_filter2->u.mat.columns;
		filter2Height = (vx_uint32)src_filter2->u.mat.rows;
		filter2Coef = (float *)src_filter2->buffer;
	}
	else {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANYx2_U8 doesn't expects vx_matrix or vx_convolution object for kernel %s\n", node->akernel->name);
		return -1;
	}
	if (filterWidth != filter2Width || filterHeight != filter2Height || src_filter->ref.read_only != src_filter2->ref.read_only || src_filter->ref.type != src_filter2->ref.type) {
		agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANYx2_U8 requires both filters to have same attributes\n");
		return -1;
	}
	bool clampNotNeeded = false;
	bool filterCoefAreIntegers = false;
	if (filterCoefAreConstants) {
		float sumP = 0.0f, sumN = 0.0f;
		float sumP2 = 0.0f, sumN2 = 0.0f;
		filterCoefAreIntegers = true;
		for (vx_uint32 i = 0; i < filterWidth * filterHeight; i++) {
			if (floorf(filterCoef[i]) != filter2Coef[i] || floorf(filter2Coef[i]) != filter2Coef[i])
				filterCoefAreIntegers = false;
			if (filterCoef[i] < 0.0f) sumN += filterCoef[i];
			else sumP += filterCoef[i];
			if (filter2Coef[i] < 0.0f) sumN2 += filter2Coef[i];
			else sumP2 += filter2Coef[i];
		}
		if ((sumN*255.0f > -32767.0f && sumP*255.0f < 32766.0f) && (sumN2*255.0f > -32767.0f && sumP2*255.0f < 32766.0f))
			clampNotNeeded = true;
	}

	std::string code;
	char item[1024];
	if (filterHeight == 1 && filterWidth > 1) {
		// generate code for Mx1 filter
		vx_uint32 Mdiv2 = filterWidth >> 1; if (Mdiv2 == 0) { 
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANY_U8 doesn't support %dx%d filter\n", filterWidth, filterHeight);
			return -1; 
		}
		// function declaration
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r1, %sx8 * r2, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_%dx1;\n"
				"void %s(%sx8 * r1, %sx8 * r2, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_%dx1 coef1, COEF_%dx1 coef2) {\n"
				), filterWidth, filterWidth, node->opencl_name, dstRegType, dstRegType, filterWidth, filterWidth);
		}
		code = item;

		// configuration
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;
		vx_uint32 LMemWidth = AGO_OPENCL_WORKGROUP_SIZE_0 * 8;
		vx_uint32 LMemSideAlign = (Mdiv2 < 8) ? 3 : 7;
		vx_uint32 LMemSide = ((Mdiv2 + LMemSideAlign) & ~LMemSideAlign);
		vx_uint32 LMemStride = LMemWidth + 2 * LMemSide;

		node->opencl_param_discard_mask = filterCoefAreConstants ? (3 << 3) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 2);
		node->opencl_local_buffer_size_in_bytes = LMemHeight * LMemStride;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemStride, LMemHeight, LMemSide, 0, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  F32x8 sum1, sum2; uint2 pix; float fval;\n"
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			), LMemStride);
		code += item;
		int numQW = (LMemSide / 4) + 1;
		for (int qw = 0; qw < numQW; qw++) {
			bool loaded_pix = false;
			for (int x = 0; x < 8; x++) {
				int bytepos = qw * 8 + x;
				int xpos = bytepos - LMemSide;
				if (xpos >= -(int)Mdiv2 && xpos <= (7 + (int)Mdiv2)) {
					bool loaded_fval = false;
					for (int ix = 0; ix < 8; ix++) {
						int ixpos = xpos - ix;
						if (ixpos == -(int)Mdiv2) {
							if (filterCoefAreConstants) {
								if (filterCoef[0] == 0.0f) {
									snprintf(item, sizeof(item), "  sum1.s%d = 0.0f;\n", ix);
								}
								else {
									if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
									if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
									if (filterCoef[0] == 1.0f) {
										snprintf(item, sizeof(item), "  sum1.s%d =     fval;\n", ix);
									}
									else {
										snprintf(item, sizeof(item), "  sum1.s%d =     fval* %.12ef;\n", ix, filterCoef[0]);
									}
								}
								code += item;
								if (filter2Coef[0] == 0.0f) {
									snprintf(item, sizeof(item), "  sum2.s%d = 0.0f;\n", ix);
								}
								else {
									if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
									if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
									if (filter2Coef[0] == 1.0f) {
										snprintf(item, sizeof(item), "  sum2.s%d =     fval;\n", ix);
									}
									else {
										snprintf(item, sizeof(item), "  sum2.s%d =     fval* %.12ef;\n", ix, filter2Coef[0]);
									}
								}
								code += item;
							}
							else {
								if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
								if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
								snprintf(item, sizeof(item), 
									"  sum1.s%d =     fval* coef1.f[ 0];\n"
									"  sum2.s%d =     fval* coef2.f[ 0];\n"
									, ix, ix);
								code += item;
							}
						}
						else if ((ixpos > -(int)Mdiv2) && (ixpos <= (int)Mdiv2)) {
							if (filterCoefAreConstants) {
								if (filterCoef[ixpos + Mdiv2] != 0.0f) {
									if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
									if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
									snprintf(item, sizeof(item), "  sum1.s%d = mad(fval, %.12ef, sum1.s%d);\n", ix, filterCoef[ixpos + Mdiv2], ix);
									code += item;
								}
								if (filter2Coef[ixpos + Mdiv2] != 0.0f) {
									if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
									if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
									snprintf(item, sizeof(item), "  sum2.s%d = mad(fval, %.12ef, sum2.s%d);\n", ix, filter2Coef[ixpos + Mdiv2], ix);
									code += item;
								}
							}
							else {
								if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw); code += item; }
								if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
								snprintf(item, sizeof(item), "  sum1.s%d = mad(fval, coef1.f[%2d], sum1.s%d);\n", ix, ixpos + Mdiv2, ix);
								code += item;
								snprintf(item, sizeof(item), "  sum2.s%d = mad(fval, coef2.f[%2d], sum2.s%d);\n", ix, ixpos + Mdiv2, ix);
								code += item;
							}
						}
					}
				}
			}
		}
	}
	else {
		// generate code for MxN filter
		vx_uint32 Ndiv2 = filterHeight >> 1;
		vx_uint32 Mdiv2 = filterWidth >> 1;
		if (Mdiv2 == 0 || Ndiv2 == 0) { 
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: HafGpu_LinearFilter_ANYx2_U8 doesn't support %dx%d filter\n", filterWidth, filterHeight);
			return -1; 
		}

		// function declaration
		if (filterCoefAreConstants) {
			snprintf(item, sizeof(item),
				"void %s(%sx8 * r1, %sx8 * r2, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {\n"
				, node->opencl_name, dstRegType, dstRegType);
		}
		else {
			snprintf(item, sizeof(item),
				OPENCL_FORMAT(
				"typedef struct { float f[%d]; } COEF_%dx%d;\n"
				"void %s(%sx8 * r1, %sx8 * r2, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride, COEF_%dx%d coef1, COEF_%dx%d coef2) {\n"
				), filterWidth*filterHeight, filterWidth, filterHeight, node->opencl_name, dstRegType, dstRegType, filterWidth, filterHeight, filterWidth, filterHeight);
		}
		code = item;

		// configuration
		vx_uint32 LMemHeight = AGO_OPENCL_WORKGROUP_SIZE_1;
		vx_uint32 LMemWidth = AGO_OPENCL_WORKGROUP_SIZE_0 * 8;
		vx_uint32 LMemSideLR = ((Mdiv2 + 3) & ~3);
		vx_uint32 LMemStride = LMemWidth + 2 * LMemSideLR;
		vx_uint32 LMemSideTB = Ndiv2;
		vx_uint32 LMemSize = (LMemHeight + 2 * LMemSideTB) * LMemStride;
		node->opencl_param_discard_mask = filterCoefAreConstants ? (3 << 3) : 0;
		node->opencl_local_buffer_usage_mask = (1 << 2);
		node->opencl_local_buffer_size_in_bytes = LMemSize;

		// generate local memory load
		code +=
			OPENCL_FORMAT(
			"  int lx = get_local_id(0);\n"
			"  int ly = get_local_id(1);\n"
			"  int gx = x >> 3;\n"
			"  int gy = y;\n"
			"  int gstride = stride;\n"
			"  __global uchar * gbuf = p;\n");
		if (HafGpu_Load_Local(AGO_OPENCL_WORKGROUP_SIZE_0, AGO_OPENCL_WORKGROUP_SIZE_1, LMemStride, LMemHeight + 2 * LMemSideTB, LMemSideLR, LMemSideTB, code) < 0) {
			return -1;
		}

		// generate computation
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  F32x8 sum1 = (F32x8)0.0f, sum2 = (F32x8)0.0f; uint2 pix; float fval;\n"
			"  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * %d + (lx << 3));\n" // LMemStride
			), LMemStride);
		code += item;
		int numQW = (LMemSideLR / 4) + 1;
		for (int y = 0; y < (int)filterHeight; y++) {
			snprintf(item, sizeof(item), "  // filterRow = %d\n", y); code += item;
			for (int qw = 0; qw < numQW; qw++) {
				bool loaded_pix = false;
				for (int x = 0; x < 8; x++) {
					int bytepos = qw * 8 + x;
					int xpos = bytepos - LMemSideLR;
					if (xpos >= -(int)Mdiv2 && xpos <= (7 + (int)Mdiv2)) {
						bool loaded_fval = false;
						for (int ix = 0; ix < 8; ix++) {
							int ixpos = xpos - ix;
							if ((ixpos >= -(int)Mdiv2) && (ixpos <= (int)Mdiv2)) {
								int coefPos = y * filterWidth + ixpos + Mdiv2;
								if (filterCoefAreConstants) {
									if (filterCoef[coefPos] != 0.0f) {
										if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw + y*LMemStride / 8); code += item; }
										if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
										if (filterCoef[coefPos] == 1.0f)       snprintf(item, sizeof(item), "  sum1.s%d += fval;\n", ix);
										else if (filterCoef[coefPos] == -1.0f) snprintf(item, sizeof(item), "  sum1.s%d -= fval;\n", ix);
										else                                   snprintf(item, sizeof(item), "  sum1.s%d  = mad(fval, %.12ef, sum1.s%d);\n", ix, filterCoef[coefPos], ix);
										code += item;
									}
									if (filter2Coef[coefPos] != 0.0f) {
										if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw + y*LMemStride / 8); code += item; }
										if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
										if (filter2Coef[coefPos] == 1.0f)       snprintf(item, sizeof(item), "  sum2.s%d += fval;\n", ix);
										else if (filter2Coef[coefPos] == -1.0f) snprintf(item, sizeof(item), "  sum2.s%d -= fval;\n", ix);
										else                                    snprintf(item, sizeof(item), "  sum2.s%d  = mad(fval, %.12ef, sum2.s%d);\n", ix, filter2Coef[coefPos], ix);
										code += item;
									}
								}
								else {
									if (!loaded_pix) { loaded_pix = true; snprintf(item, sizeof(item), "  pix = lbufptr[%d];\n", qw + y*LMemStride / 8); code += item; }
									if (!loaded_fval) { loaded_fval = true; snprintf(item, sizeof(item), "  fval = amd_unpack%d(pix.s%d);\n", x & 3, x >> 2); code += item; }
									snprintf(item, sizeof(item), "  sum1.s%d = mad(fval, coef1.f[%2d], sum1.s%d);\n", ix, coefPos, ix);
									code += item;
									snprintf(item, sizeof(item), "  sum2.s%d = mad(fval, coef2.f[%2d], sum2.s%d);\n", ix, coefPos, ix);
									code += item;
								}
							}
						}
					}
				}
			}
		}
	}
	if (!filterCoefAreIntegers && roundingBias != 0.0f) {
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
			"  sum1.s0 = sum1.s0 + %.12ef;\n"
			"  sum1.s1 = sum1.s1 + %.12ef;\n"
			"  sum1.s2 = sum1.s2 + %.12ef;\n"
			"  sum1.s3 = sum1.s3 + %.12ef;\n"
			"  sum1.s4 = sum1.s4 + %.12ef;\n"
			"  sum1.s5 = sum1.s5 + %.12ef;\n"
			"  sum1.s6 = sum1.s6 + %.12ef;\n"
			"  sum1.s7 = sum1.s7 + %.12ef;\n"
			"  sum2.s0 = sum2.s0 + %.12ef;\n"
			"  sum2.s1 = sum2.s1 + %.12ef;\n"
			"  sum2.s2 = sum2.s2 + %.12ef;\n"
			"  sum2.s3 = sum2.s3 + %.12ef;\n"
			"  sum2.s4 = sum2.s4 + %.12ef;\n"
			"  sum2.s5 = sum2.s5 + %.12ef;\n"
			"  sum2.s6 = sum2.s6 + %.12ef;\n"
			"  sum2.s7 = sum2.s7 + %.12ef;\n")
			, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias
			, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias, roundingBias);
		code += item;
	}
	if (dstIsS16) {
		if (clampNotNeeded) {
			code +=
				OPENCL_FORMAT(
				"  S16x8 rv;\n"
				"  rv.s0  = ((int)sum1.s0) & 0xffff;\n"
				"  rv.s0 |= ((int)sum1.s1) << 16;\n"
				"  rv.s1  = ((int)sum1.s2) & 0xffff;\n"
				"  rv.s1 |= ((int)sum1.s3) << 16;\n"
				"  rv.s2  = ((int)sum1.s4) & 0xffff;\n"
				"  rv.s2 |= ((int)sum1.s5) << 16;\n"
				"  rv.s3  = ((int)sum1.s6) & 0xffff;\n"
				"  rv.s3 |= ((int)sum1.s7) << 16;\n"
				"  *r1 = rv;\n"
				"  rv.s0  = ((int)sum2.s0) & 0xffff;\n"
				"  rv.s0 |= ((int)sum2.s1) << 16;\n"
				"  rv.s1  = ((int)sum2.s2) & 0xffff;\n"
				"  rv.s1 |= ((int)sum2.s3) << 16;\n"
				"  rv.s2  = ((int)sum2.s4) & 0xffff;\n"
				"  rv.s2 |= ((int)sum2.s5) << 16;\n"
				"  rv.s3  = ((int)sum2.s6) & 0xffff;\n"
				"  rv.s3 |= ((int)sum2.s7) << 16;\n"
				"  *r2 = rv;\n"
				"}\n");
		}
		else {
			code +=
				OPENCL_FORMAT(
				"  S16x8 rv;\n"
				"  rv.s0  = ((int)clamp(sum1.s0, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s0 |= ((int)clamp(sum1.s1, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s1  = ((int)clamp(sum1.s2, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s1 |= ((int)clamp(sum1.s3, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s2  = ((int)clamp(sum1.s4, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s2 |= ((int)clamp(sum1.s5, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s3  = ((int)clamp(sum1.s6, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s3 |= ((int)clamp(sum1.s7, -32768.0f, 32767.0f)) << 16;\n"
				"  *r1 = rv;\n"
				"  rv.s0  = ((int)clamp(sum2.s0, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s0 |= ((int)clamp(sum2.s1, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s1  = ((int)clamp(sum2.s2, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s1 |= ((int)clamp(sum2.s3, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s2  = ((int)clamp(sum2.s4, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s2 |= ((int)clamp(sum2.s5, -32768.0f, 32767.0f)) << 16;\n"
				"  rv.s3  = ((int)clamp(sum2.s6, -32768.0f, 32767.0f)) & 0xffff;\n"
				"  rv.s3 |= ((int)clamp(sum2.s7, -32768.0f, 32767.0f)) << 16;\n"
				"  *r2 = rv;\n"
				"}\n");
		}
	}
	else if (dstIsF32) {
		code +=
			"  *r1 = sum1;\n"
			"  *r2 = sum2;\n"
			"}\n";
	}
	else {
		code +=
			OPENCL_FORMAT(
			"  U8x8 rv;\n"
			"  rv.s0 = amd_pack(sum1.s0123);\n"
			"  rv.s1 = amd_pack(sum1.s4567);\n"
			"  *r1 = rv;\n"
			"  rv.s0 = amd_pack(sum2.s0123);\n"
			"  rv.s1 = amd_pack(sum2.s4567);\n"
			"  *r2 = rv;\n"
			"}\n");
	}
	node->opencl_code += code;
	node->opencl_type = NODE_OPENCL_TYPE_MEM2REG;

	return status;
}

#endif
