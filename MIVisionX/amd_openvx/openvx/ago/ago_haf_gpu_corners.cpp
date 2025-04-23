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
// Generate OpenCL code for following fast corner detector kernels:
//   VX_KERNEL_AMD_FAST_CORNERS_XY_U8_NOSUPRESSION, VX_KERNEL_AMD_FAST_CORNERS_XY_U8_SUPRESSION,
//
int HafGpu_FastCorners_XY_U8(AgoNode * node)
{
	std::string code;
	char item[8192];
	int status = VX_SUCCESS;
	bool useNonMax = (node->akernel->id == VX_KERNEL_AMD_FAST_CORNERS_XY_U8_SUPRESSION);

	// configuration
	AgoData * cornerList = node->paramList[0];
	AgoData * numCorners = node->paramList[1];
	AgoData * inputImg = node->paramList[2];
	AgoData * inputThr = node->paramList[3];
	int work_group_width = 16;
	int work_group_height = 16;

	// use completely separate kernel
	node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
	node->opencl_work_dim = 2;
	node->opencl_global_work[2] = 0;
	node->opencl_local_work[0] = work_group_width;
	node->opencl_local_work[1] = work_group_height;
	node->opencl_local_work[2] = 0;
	node->opencl_param_discard_mask = 0;
	node->opencl_param_atomic_mask = (1 << 0);
	node->opencl_local_buffer_usage_mask = 0;
	node->opencl_local_buffer_size_in_bytes = 0;
	node->gpu_scalar_array_output_sync.enable = false;
	if (numCorners) {
		// discard the scalar argument and inform the framework that it needs to be synched with array output numitems
		node->opencl_param_discard_mask = (1 << 1);
		node->gpu_scalar_array_output_sync.enable = true;
		node->gpu_scalar_array_output_sync.paramIndexArray = 0;
		node->gpu_scalar_array_output_sync.paramIndexScalar = 1;
	}

	if (useNonMax)
	{
		// FAST with non-max supression

		// OpenCL work items
		node->opencl_global_work[0] = (size_t) ceil((inputImg->u.img.width - 4)/14)*16;
		node->opencl_global_work[1] = (size_t) ceil((inputImg->u.img.height - 4)/14)*16;

		// Pragma, data structure declarations and helper functions
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
				"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
				"#define MASK_EARLY_EXIT 4369\n\n"								//((1<<0) | (1<<4) | (1<<8) | (1<<12))
				"typedef struct {\n"
				"\t int x;\n"
				"\t int y;\n"
				"\t float strength;\n"
				"\t float scale;\n"
				"\t float orientation;\n"
				"\t int tracking_status;\n"
				"\t float error;\n"
				"} KeyPt;\n\n"
				"inline int getScore(int * boundary)	{\n"
				"\t int strength, tmp = 0;\n"
				"\t for (int i = 0; i < 16; i += 2)	{\n"
				"\t\t int s = min(boundary[(i + 1) & 15], boundary[(i + 2) & 15]);\n"
				"\t\t s = min(s, boundary[(i + 3) & 15]);\n"
				"\t\t s = min(s, boundary[(i + 4) & 15]);\n"
				"\t\t s = min(s, boundary[(i + 5) & 15]);\n"
				"\t\t s = min(s, boundary[(i + 6) & 15]);\n"
				"\t\t s = min(s, boundary[(i + 7) & 15]);\n"
				"\t\t s = min(s, boundary[(i + 8) & 15]);\n"
				"\t\t tmp = max(tmp, min(s, boundary[i & 15]));\n"
				"\t\t tmp = max(tmp, min(s, boundary[(i + 9) & 15]));\n"
				"\t }\n"
				"\t strength = -tmp;\n"
				"\t for (int i = 0; i < 16; i += 2)	{\n"
				"\t\t int s = max(boundary[(i + 1) & 15], boundary[(i + 2) & 15]);\n"
				"\t\t s = max(s, boundary[(i + 3) & 15]);\n"
				"\t\t s = max(s, boundary[(i + 4) & 15]);\n"
				"\t\t s = max(s, boundary[(i + 5) & 15]);\n"
				"\t\t s = max(s, boundary[(i + 6) & 15]);\n"
				"\t\t s = max(s, boundary[(i + 7) & 15]);\n"
				"\t\t s = max(s, boundary[(i + 8) & 15]);\n"
				"\t\t strength = min(strength, max(s, boundary[i & 15]));\n"
				"\t\t strength = min(strength, max(s, boundary[(i + 9) & 15]));\n"
				"\t }\n"
				"\t return(-strength-1);\n } \n"
			)
			);
		code = item;

		// function declaration
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
				"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
				"void %s(__global char * corner_buf, uint corner_buf_offset, uint corner_capacity, uint img_width, uint img_height, __global uchar * img_buf, uint img_stride, uint img_offset, float strength_thresh)\n"
				"{\n"
			)
			, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME);
		code += item;

		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
				"\t int lidx = (int) get_local_id(0);\n"
				"\t int lidy = (int)get_local_id(1);\n"
				"\t int gidx = (int)get_group_id(0);\n"
				"\t int gidy = (int)get_group_id(1);\n"
				"\t int xoffset = gidx * 14 + lidx + 2;\n"
				"\t int yoffset = gidy * 14 + lidy + 2;\n"
				"\t __global const uchar * pTempImg = img_buf + img_offset + mad24(yoffset, (int)img_stride, xoffset);\n"
				"\t __local int pLocalStrengthShare[16][16];\n"
				"\t bool doCompute = true;\n"
				"\t if((xoffset > (int)img_width - 3) || (yoffset > (int)img_height - 3) || (xoffset < 3) || (yoffset < 3))	{\n"
				"\t\t doCompute = false;\n"
				"\t\t pLocalStrengthShare[lidy][lidx] = 0;\n \t}\n"
				"\t int local_strength;\n"
				"\t if(doCompute)	{\n"
				"\t\t int boundary[16];\n"
				"\t\t int pos_mask, neg_mask, offs;\n"
				"\t\t int centerPixel_neg = pTempImg[0];\n"
				"\t\t for(int i = 0; i < 16; i++)\n"
				"\t\t\t boundary[i] = centerPixel_neg;\n"
				"\t\t int centerPixel_pos = centerPixel_neg + (int)strength_thresh;\n"
				"\t\t centerPixel_neg -= (int) strength_thresh;\n"
				"\t\t int candp = pTempImg[3];\n"
				"\t\t int candn = pTempImg[-3];\n"
				"\t\t neg_mask = (candp < centerPixel_neg) | ((candn < centerPixel_neg) << 8);\n"
				"\t\t pos_mask = (candp > centerPixel_pos) | ((candn > centerPixel_pos) << 8);\n"
				"\t\t boundary[0] -= candp;\n"
				"\t\t boundary[8] -= candn;\n"
				"\t\t offs = -img_stride*3;\n"
				"\t\t candp = pTempImg[offs];\n"
				"\t\t candn = pTempImg[-offs];\n"
				"\t\t neg_mask |= (((candp < centerPixel_neg) << 4) | ((candn < centerPixel_neg) << 12));\n"
				"\t\t pos_mask |= (((candp > centerPixel_pos) << 4) | ((candn > centerPixel_pos) << 12));\n"
				"\t\t boundary[4] -= candp;\n"
				"\t\t boundary[12] -= candn;\n"
				"\t\t if(((pos_mask | neg_mask) & MASK_EARLY_EXIT) == 0)	{\n"
				"\t\t\t pLocalStrengthShare[lidy][lidx] = 0;\n"
				"\t\t\t doCompute = false;\n \t\t }\n"
				"\t\t else  {\n"
				"\t\t\t offs = -img_stride*3 + 1;\n"
				"\t\t\t candp = pTempImg[offs];\n"
				"\t\t\t candn = pTempImg[-offs];\n"
				"\t\t\t neg_mask |= (((candp < centerPixel_neg) << 3) | ((candn < centerPixel_neg) << 11));\n"
				"\t\t\t pos_mask |= (((candp > centerPixel_pos) << 3) | ((candn > centerPixel_pos) << 11));\n"
				"\t\t\t boundary[3] -= candp;\n"
				"\t\t\t boundary[11] -= candn;\n"
				"\t\t\t offs = -img_stride*3 - 1;\n"
				"\t\t\t candp = pTempImg[offs];\n"
				"\t\t\t candn = pTempImg[-offs];\n"
				"\t\t\t neg_mask |= (((candp < centerPixel_neg) << 5) | ((candn < centerPixel_neg) << 13));\n"
				"\t\t\t pos_mask |= (((candp > centerPixel_pos) << 5) | ((candn > centerPixel_pos) << 13));\n"
				"\t\t\t boundary[5] -= candp;\n"
				"\t\t\t boundary[13] -= candn;\n"
				"\t\t\t offs = -(img_stride<<1) + 2;\n"
				"\t\t\t candp = pTempImg[offs];\n"
				"\t\t\t candn = pTempImg[-offs];\n"
				"\t\t\t neg_mask |= (((candp < centerPixel_neg) << 2) | ((candn < centerPixel_neg) << 10));\n"
				"\t\t\t pos_mask |= (((candp > centerPixel_pos) << 2) | ((candn > centerPixel_pos) << 10));\n"
				"\t\t\t boundary[2] -= candp;\n"
				"\t\t\t boundary[10] -= candn;\n"
				"\t\t\t offs = -(img_stride<<1) - 2;\n"
				"\t\t\t candp = pTempImg[offs];\n"
				"\t\t\t candn = pTempImg[-offs];\n"
				"\t\t\t neg_mask |= (((candp < centerPixel_neg) << 6) | ((candn < centerPixel_neg) << 14));\n"
				"\t\t\t pos_mask |= (((candp > centerPixel_pos) << 6) | ((candn > centerPixel_pos) << 14));\n"
				"\t\t\t boundary[6] -= candp;\n"
				"\t\t\t boundary[14] -= candn;\n"
				"\t\t\t offs = -img_stride + 3;\n"
				"\t\t\t candp = pTempImg[offs];\n"
				"\t\t\t candn = pTempImg[-offs];\n"
				"\t\t\t neg_mask |= (((candp < centerPixel_neg) << 1) | ((candn < centerPixel_neg) << 9));\n"
				"\t\t\t pos_mask |= (((candp > centerPixel_pos) << 1) | ((candn > centerPixel_pos) << 9));\n"
				"\t\t\t boundary[1] -= candp;\n"
				"\t\t\t boundary[9] -= candn;\n"
				"\t\t\t offs = -img_stride - 3;\n"
				"\t\t\t candp = pTempImg[offs];\n"
				"\t\t\t candn = pTempImg[-offs];\n"
				"\t\t\t neg_mask |= (((candp < centerPixel_neg) << 7) | ((candn < centerPixel_neg) << 15));\n"
				"\t\t\t pos_mask |= (((candp > centerPixel_pos) << 7) | ((candn > centerPixel_pos) << 15));\n"
				"\t\t\t boundary[7] -= candp;\n"
				"\t\t\t boundary[15] -= candn;\n"
				"\t\t\t pos_mask |= (pos_mask << 16);\n"
				"\t\t\t neg_mask |= (neg_mask << 16);\n"
				"\t\t\t int cornerMask = 511;\n"
				"\t\t\t int isCorner = 0;\n"
				"\t\t\t for (int i = 0; i < 16; i++)	{\n"
				"\t\t\t\t isCorner += ((pos_mask & cornerMask) == cornerMask);\n"
				"\t\t\t\t isCorner += ((neg_mask & cornerMask) == cornerMask);\n"
				"\t\t\t\t pos_mask >>= 1;\n"
				"\t\t\t\t neg_mask >>= 1;\n\t\t\t }\n"
				"\t\t\t if(isCorner == 0)	{\n"
				"\t\t\t\t pLocalStrengthShare[lidy][lidx] = 0;\n"
				"\t\t\t\t doCompute = false;\n\t\t\t }\n"
				"\t\t\t else	{\n"
				"\t\t\t\t local_strength = getScore(boundary);\n"
				"\t\t\t\t pLocalStrengthShare[lidy][lidx] = local_strength;\n\t\t\t }\n"
				"\t\t }\n\t }\n"
				"\t barrier(CLK_LOCAL_MEM_FENCE);\n\n"
				"\t bool writeCorner = doCompute && (local_strength >= pLocalStrengthShare[lidy-1][lidx-1]) && (local_strength >= pLocalStrengthShare[lidy-1][lidx]) && (local_strength >= pLocalStrengthShare[lidy-1][lidx+1])\n"
				"\t\t\t\t\t\t && (local_strength >= pLocalStrengthShare[lidy][lidx-1]) && (local_strength > pLocalStrengthShare[lidy][lidx+1])\n"
				"\t\t\t\t\t\t && (local_strength > pLocalStrengthShare[lidy+1][lidx-1]) && (local_strength > pLocalStrengthShare[lidy+1][lidx]) && (local_strength >= pLocalStrengthShare[lidy+1][lidx+1])\n"
				"\t\t\t\t\t\t && (lidx > 0) && (lidy > 0) && (lidx < 15) && (lidy < 15);\n"
				"\t __global int * numKeypoints = (__global int *) corner_buf;\n"
				"\t __global KeyPt * keypt_list = (__global KeyPt *)(corner_buf + corner_buf_offset);\n"
				"\t if(writeCorner)	{\n"
				"\t\t\t int old_idx = atomic_inc(numKeypoints);\n"
				"\t\t if(old_idx < corner_capacity)	{\n"
				"\t\t\t keypt_list[old_idx].x = xoffset;\n"
				"\t\t\t keypt_list[old_idx].y = yoffset;\n"
				"\t\t\t keypt_list[old_idx].strength = (float) local_strength;\n"
				"\t\t\t keypt_list[old_idx].scale = 0;\n"
				"\t\t\t keypt_list[old_idx].orientation = 0;\n"
				"\t\t\t keypt_list[old_idx].tracking_status = 1;\n"
				"\t\t\t keypt_list[old_idx].error = 0;\n \t\t} \n \t}\n"
			)
			);
		code += item;
		code += "}\n";
	}
	else
	{
		// FAST without non-max supression

		// OpenCL work items
		node->opencl_global_work[0] = (inputImg->u.img.width  - 6 + work_group_width  - 1) & ~(work_group_width  - 1);
		node->opencl_global_work[1] = (inputImg->u.img.height - 6 + work_group_height - 1) & ~(work_group_height - 1);

		// Pragma and data structure declarations
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
				"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
				"#define MASK_EARLY_EXIT 4369\n\n"								//((1<<0) | (1<<4) | (1<<8) | (1<<12))
				"typedef struct {\n"
				"\t int x;\n"
				"\t int y;\n"
				"\t float strength;\n"
				"\t float scale;\n"
				"\t float orientation;\n"
				"\t int tracking_status;\n"
				"\t float error;\n"
				"} KeyPt;\n"
			)
			);
		code = item;

		// function declaration
		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
				"__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\n"
				"void %s(__global char * corner_buf, uint corner_buf_offset, uint corner_capacity, uint img_width, uint img_height, __global uchar * img_buf, uint img_stride, uint img_offset, float strength_thresh)\n"
				"{\n"
			)
			, work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME);
		code += item;

		snprintf(item, sizeof(item),
			OPENCL_FORMAT(
				"\t int idx = (int) get_global_id(0) + 3;\n"
				"\t int idy = (int) get_global_id(1) + 3;\n"
				"\t int stride = (int) img_stride;\n"
				"\t if((idx > (int)img_width - 3) || (idy > (int)img_height - 3))  return;\n"
				"\t __global const uchar * pTempImg = img_buf + img_offset + mad24(idy, stride, idx);\n"
				"\t int centerPixel_neg = pTempImg[0];\n"
				"\t int centerPixel_pos = centerPixel_neg + (int)strength_thresh;\n"
				"\t centerPixel_neg -= (int)strength_thresh;\n"
				"\t int candp, candn, pos_mask, neg_mask;\n"
				"\t candp = pTempImg[3];\n"
				"\t candn = pTempImg[-3];\n"
				"\t neg_mask = (candp < centerPixel_neg) | ((candn < centerPixel_neg) << 8);\n"				// Position 0 and 8
				"\t pos_mask = (candp > centerPixel_pos) | ((candn > centerPixel_pos) << 8);\n"
				"\t int offs = -stride*3;\n"
				"\t candp = pTempImg[offs];\n"
				"\t candn = pTempImg[-offs];\n"
				"\t neg_mask |= (((candp < centerPixel_neg) << 4) | ((candn < centerPixel_neg) << 12));\n"		// Position 4,12
				"\t pos_mask |= (((candp > centerPixel_pos) << 4) | ((candn > centerPixel_pos) << 12));\n"
				"\t if(((pos_mask | neg_mask) & MASK_EARLY_EXIT) == 0)   return;\n"							// Early exit condition
				"\t offs = -stride*3 + 1;\n"
				"\t candp = pTempImg[offs];\n"
				"\t candn = pTempImg[-offs];\n"
				"\t neg_mask |= (((candp < centerPixel_neg) << 3) | ((candn < centerPixel_neg) << 11));\n"		// Position 3,11
				"\t pos_mask |= (((candp > centerPixel_pos) << 3) | ((candn > centerPixel_pos) << 11));\n"
				"\t offs = -stride*3 - 1;\n"
				"\t candp = pTempImg[offs];\n"
				"\t candn = pTempImg[-offs];\n"
				"\t neg_mask |= (((candp < centerPixel_neg) << 5) | ((candn < centerPixel_neg) << 13));\n"		// Position 5,13
				"\t pos_mask |= (((candp > centerPixel_pos) << 5) | ((candn > centerPixel_pos) << 13));\n"
				"\t offs = -(stride << 1) + 2;\n"
				"\t candp = pTempImg[offs];\n"
				"\t candn = pTempImg[-offs];\n"
				"\t neg_mask |= (((candp < centerPixel_neg) << 2) | ((candn < centerPixel_neg) << 10));\n"		// Position 2,10
				"\t pos_mask |= (((candp > centerPixel_pos) << 2) | ((candn > centerPixel_pos) << 10));\n"
				"\t offs = -(stride << 1) - 2;\n"
				"\t candp = pTempImg[offs];\n"
				"\t candn = pTempImg[-offs];\n"
				"\t neg_mask |= (((candp < centerPixel_neg) << 6) | ((candn < centerPixel_neg) << 14));\n"		// Position 6,14
				"\t pos_mask |= (((candp > centerPixel_pos) << 6) | ((candn > centerPixel_pos) << 14));\n"
				"\t offs = -stride + 3;\n"
				"\t candp = pTempImg[offs];\n"
				"\t candn = pTempImg[-offs];\n"
				"\t neg_mask |= (((candp < centerPixel_neg) << 1) | ((candn < centerPixel_neg) << 9));\n"		// Position 1,9
				"\t pos_mask |= (((candp > centerPixel_pos) << 1) | ((candn > centerPixel_pos) << 9));\n"
				"\t offs = -stride - 3;\n"
				"\t candp = pTempImg[offs];\n"
				"\t candn = pTempImg[-offs];\n"
				"\t neg_mask |= (((candp < centerPixel_neg) << 7) | ((candn < centerPixel_neg) << 15));\n"		// Position 7,15
				"\t pos_mask |= (((candp > centerPixel_pos) << 7) | ((candn > centerPixel_pos) << 15));\n"
				"\t pos_mask |= (pos_mask << 16);		neg_mask |= (neg_mask << 16);\n"
				"\t int cornerMask = 511, isCorner = 0;\n"
				"\t for(int i = 0; i < 16; i++)	{\n"
				"\t\t isCorner += ((pos_mask & cornerMask) == cornerMask);\n"
				"\t\t isCorner += ((neg_mask & cornerMask) == cornerMask);\n"
				"\t\t pos_mask >>= 1;\n"
				"\t\t neg_mask >>= 1;\n \t} \n"
				"\t __global int * numKeypoints = (__global int *) corner_buf;\n"
				"\t __global KeyPt * keypt_list = (__global KeyPt *)(corner_buf + corner_buf_offset);\n"
				"\t if(isCorner)	{\n"
				"\t\t\t int old_idx = atomic_inc(numKeypoints);\n"
				"\t\t if(old_idx < corner_capacity)	{\n"
				"\t\t\t keypt_list[old_idx].x = idx;\n"
				"\t\t\t keypt_list[old_idx].y = idy;\n"
				"\t\t\t keypt_list[old_idx].strength = strength_thresh;\n"
				"\t\t\t keypt_list[old_idx].scale = 0;\n"
				"\t\t\t keypt_list[old_idx].orientation = 0;\n"
				"\t\t\t keypt_list[old_idx].tracking_status = 1;\n"
				"\t\t\t keypt_list[old_idx].error = 0;\n \t\t} \n \t}\n"
			)
			);

		code += item;
		code += "}\n";
	}
	
	node->opencl_code = code;
	return status;
}

#endif
