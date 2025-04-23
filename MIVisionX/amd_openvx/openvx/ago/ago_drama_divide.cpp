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


#include "ago_internal.h"

#define SANITY_CHECK_DATA_TYPE(data,data_type)          if(!data || data->ref.type != data_type) return -1
#define SANITY_CHECK_DATA_TYPE_OPTIONAL(data,data_type) if( data && data->ref.type != data_type) return -1

int agoDramaDivideAppend(AgoNodeList * nodeList, AgoNode * anode, vx_enum new_kernel_id, vx_reference * paramList, vx_uint32 paramCount)
{
	if (new_kernel_id == VX_KERNEL_AMD_INVALID) {
		// TBD: error handling
		agoAddLogEntry(&anode->akernel->ref, VX_FAILURE, "ERROR: agoDramaDivideAppend(*,0x%08x[%s],INVALID) not implemented\n", anode->akernel->id, anode->akernel->name);
		return -1;
	}
	// create a new AgoNode and add it to the nodeList
	AgoNode * childnode = agoCreateNode((AgoGraph *)anode->ref.scope, new_kernel_id);
	for (vx_uint32 i = 0; i < paramCount; i++) {
		childnode->paramList[i] = (AgoData *)paramList[i];
	}
	anode->drama_divide_invoked = true;
	// transfer attributes from anode to childnode
	agoImportNodeConfig(childnode, anode);

	// set childnode for future reference
	anode->newchildnode = childnode;
	
	// verify the node
	return agoVerifyNode(childnode);
}

vx_status VX_CALLBACK agoDramaDivideAddNodeCallback(vx_node node, vx_enum kernel_id, vx_reference * paramList, vx_uint32 paramCount)
{
    return agoDramaDivideAppend(&((AgoGraph *)node->ref.scope)->nodeList, node, kernel_id, paramList, paramCount);
}

int agoDramaDivideAppend(AgoNodeList * nodeList, AgoNode * anode, vx_enum new_kernel_id)
{
    return agoDramaDivideAppend(nodeList, anode, new_kernel_id, (vx_reference *)anode->paramList, anode->paramCount);
}

int agoDramaDivideColorConvertNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// get params
	AgoData * srcParam = anode->paramList[0];
	AgoData * dstParam = anode->paramList[1];
	vx_df_image itype = srcParam->u.img.format;
	vx_df_image otype = dstParam->u.img.format;
	// divide the node
	if (otype == VX_DF_IMAGE_RGB) {
		if (itype == VX_DF_IMAGE_RGBX) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGB_RGBX);
		}
		else if (itype == VX_DF_IMAGE_UYVY) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGB_UYVY);
		}
		else if (itype == VX_DF_IMAGE_YUYV) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGB_YUYV);
		}
		else if (itype == VX_DF_IMAGE_NV12) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam->children[0];
			anode->paramList[2] = srcParam->children[1];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV12);
		}
		else if (itype == VX_DF_IMAGE_NV21) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam->children[0];
			anode->paramList[2] = srcParam->children[1];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGB_NV21);
		}
		else if (itype == VX_DF_IMAGE_IYUV) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam->children[0];
			anode->paramList[2] = srcParam->children[1];
			anode->paramList[3] = srcParam->children[2];
			anode->paramCount = 4;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGB_IYUV);
		}
	}
	else if (otype == VX_DF_IMAGE_RGBX) {
		if (itype == VX_DF_IMAGE_RGB) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGBX_RGB);
		}
		else if (itype == VX_DF_IMAGE_UYVY) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGBX_UYVY);
		}
		else if (itype == VX_DF_IMAGE_YUYV) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGBX_YUYV);
		}
		else if (itype == VX_DF_IMAGE_NV12) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam->children[0];
			anode->paramList[2] = srcParam->children[1];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV12);
		}
		else if (itype == VX_DF_IMAGE_NV21) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam->children[0];
			anode->paramList[2] = srcParam->children[1];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGBX_NV21);
		}
		else if (itype == VX_DF_IMAGE_IYUV) {
			anode->paramList[0] = dstParam;
			anode->paramList[1] = srcParam->children[0];
			anode->paramList[2] = srcParam->children[1];
			anode->paramList[3] = srcParam->children[2];
			anode->paramCount = 4;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_RGBX_IYUV);
		}
	}
	else if (otype == VX_DF_IMAGE_NV12) {
		if (itype == VX_DF_IMAGE_UYVY) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = dstParam->children[1];
			anode->paramList[2] = srcParam;
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_NV12_UYVY);
		}
		else if (itype == VX_DF_IMAGE_YUYV) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = dstParam->children[1];
			anode->paramList[2] = srcParam;
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_NV12_YUYV);
		}
		else if (itype == VX_DF_IMAGE_IYUV) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam->children[0];
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = srcParam->children[1];
			anode->paramList[2] = srcParam->children[2];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_UV12_IUV);
		}
		else if (itype == VX_DF_IMAGE_RGB) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_Y_RGB)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGB);
		}
		else if (itype == VX_DF_IMAGE_RGBX) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_Y_RGBX)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGBX);
		}
	}
	else if (otype == VX_DF_IMAGE_IYUV) {
		if (itype == VX_DF_IMAGE_UYVY) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = dstParam->children[1];
			anode->paramList[2] = dstParam->children[2];
			anode->paramList[3] = srcParam;
			anode->paramCount = 4;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_UYVY);
		}
		else if (itype == VX_DF_IMAGE_YUYV) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = dstParam->children[1];
			anode->paramList[2] = dstParam->children[2];
			anode->paramList[3] = srcParam;
			anode->paramCount = 4;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_IYUV_YUYV);
		}
		else if (itype == VX_DF_IMAGE_NV12) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam->children[0];
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = dstParam->children[2];
			anode->paramList[2] = srcParam->children[1];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_IUV_UV12);
		}
		else if (itype == VX_DF_IMAGE_NV21) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam->children[0];
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
			anode->paramList[0] = dstParam->children[2];
			anode->paramList[1] = dstParam->children[1];
			anode->paramList[2] = srcParam->children[1];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_IUV_UV12);
		}
		else if (itype == VX_DF_IMAGE_RGB) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_Y_RGB)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_IU_RGB)) return -1;
			anode->paramList[0] = dstParam->children[2];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_IV_RGB);
		}
		else if (itype == VX_DF_IMAGE_RGBX) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_Y_RGBX)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_IU_RGBX)) return -1;
			anode->paramList[0] = dstParam->children[2];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_IV_RGBX);
		}
	}
	else if (otype == VX_DF_IMAGE_YUV4) {
		if (itype == VX_DF_IMAGE_IYUV) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam->children[0];
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = srcParam->children[1];
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_SCALE_UP_2x2_U8_U8)) return -1;
			anode->paramList[0] = dstParam->children[2];
			anode->paramList[1] = srcParam->children[2];
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_SCALE_UP_2x2_U8_U8);
		}
		else if (itype == VX_DF_IMAGE_NV12) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam->children[0];
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = dstParam->children[2];
			anode->paramList[2] = srcParam->children[1];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_UV_UV12);
		}
		else if (itype == VX_DF_IMAGE_NV21) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam->children[0];
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
			anode->paramList[0] = dstParam->children[2];
			anode->paramList[1] = dstParam->children[1];
			anode->paramList[2] = srcParam->children[1];
			anode->paramCount = 3;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_FORMAT_CONVERT_UV_UV12);
		}
		else if (itype == VX_DF_IMAGE_RGB) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_Y_RGB)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_U_RGB)) return -1;
			anode->paramList[0] = dstParam->children[2];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_V_RGB);
		}
		else if (itype == VX_DF_IMAGE_RGBX) {
			anode->paramList[0] = dstParam->children[0];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_Y_RGBX)) return -1;
			anode->paramList[0] = dstParam->children[1];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_U_RGBX)) return -1;
			anode->paramList[0] = dstParam->children[2];
			anode->paramList[1] = srcParam;
			anode->paramCount = 2;
			return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_COLOR_CONVERT_V_RGBX);
		}
	}
	return -1;
}

int agoDramaDivideChannelExtractNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// get params
	AgoData * srcParam = anode->paramList[0];
	AgoData * channelParam = anode->paramList[1];
	AgoData * dstParam = anode->paramList[2];
	vx_df_image itype = srcParam->u.img.format;
	vx_enum channel_e = channelParam->u.scalar.u.e;
	// divide the node
	if (itype == VX_DF_IMAGE_RGB) {
		anode->paramList[0] = dstParam;
		anode->paramList[1] = srcParam;
		anode->paramCount = 2;
		if (channel_e == VX_CHANNEL_R) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS0);
		else if (channel_e == VX_CHANNEL_G) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS1);
		else if (channel_e == VX_CHANNEL_B) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS2);
	}
	else if (itype == VX_DF_IMAGE_RGBX) {
		anode->paramList[0] = dstParam;
		anode->paramList[1] = srcParam;
		anode->paramCount = 2;
		if (channel_e == VX_CHANNEL_R) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS0);
		else if (channel_e == VX_CHANNEL_G) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS1);
		else if (channel_e == VX_CHANNEL_B) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS2);
		else if (channel_e == VX_CHANNEL_A) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS3);
	}
	else if (itype == VX_DF_IMAGE_NV12) {
		anode->paramList[0] = dstParam;
		anode->paramList[1] = srcParam->children[(channel_e != VX_CHANNEL_Y) ? 1 : 0];
		anode->paramCount = 2;
		if (channel_e == VX_CHANNEL_Y) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8);
		else if (channel_e == VX_CHANNEL_U) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U16_POS0);
		else if (channel_e == VX_CHANNEL_V) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U16_POS1);
	}
	else if (itype == VX_DF_IMAGE_NV21) {
		anode->paramList[0] = dstParam;
		anode->paramList[1] = srcParam->children[(channel_e != VX_CHANNEL_Y) ? 1 : 0];
		anode->paramCount = 2;
		if (channel_e == VX_CHANNEL_Y) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8);
		else if (channel_e == VX_CHANNEL_U) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U16_POS1);
		else if (channel_e == VX_CHANNEL_V) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U16_POS0);
	}
	else if (itype == VX_DF_IMAGE_UYVY) {
		anode->paramList[0] = dstParam;
		anode->paramList[1] = srcParam;
		anode->paramCount = 2;
		if (channel_e == VX_CHANNEL_Y) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U16_POS1);
		else if (channel_e == VX_CHANNEL_U) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS0);
		else if (channel_e == VX_CHANNEL_V) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS2);
	}
	else if (itype == VX_DF_IMAGE_YUYV) {
		anode->paramList[0] = dstParam;
		anode->paramList[1] = srcParam;
		anode->paramCount = 2;
		if (channel_e == VX_CHANNEL_Y) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U16_POS0);
		else if (channel_e == VX_CHANNEL_U) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS1);
		else if (channel_e == VX_CHANNEL_V) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS3);
	}
	else if (itype == VX_DF_IMAGE_IYUV || itype == VX_DF_IMAGE_YUV4) {
		anode->paramList[0] = dstParam;
		anode->paramList[1] = srcParam->children[channel_e - VX_CHANNEL_Y];
		anode->paramCount = 2;
		if (channel_e == VX_CHANNEL_Y) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8);
		else if (channel_e == VX_CHANNEL_U) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8);
		else if (channel_e == VX_CHANNEL_V) return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8);
	}
	return -1;
}

int agoDramaDivideChannelCombineNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 5) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[2], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[3], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[4], VX_TYPE_IMAGE);
	int inputMask = 3 | (anode->paramList[2] ? 4 : 0) | (anode->paramList[3] ? 8 : 0);
	// perform the divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	vx_uint32 paramCount = anode->paramCount;
	vx_df_image otype = paramList[4]->u.img.format;
	if (otype == VX_DF_IMAGE_RGB) {
		if (inputMask != 7) return -1;
		anode->paramList[0] = paramList[4];
		anode->paramList[1] = paramList[0];
		anode->paramList[2] = paramList[1];
		anode->paramList[3] = paramList[2];
		anode->paramCount = 4;
		return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COMBINE_U24_U8U8U8_RGB);
	}
	else if (otype == VX_DF_IMAGE_RGBX) {
		if (inputMask != 15) return -1;
		anode->paramList[0] = paramList[4];
		anode->paramList[1] = paramList[0];
		anode->paramList[2] = paramList[1];
		anode->paramList[3] = paramList[2];
		anode->paramList[4] = paramList[3];
		anode->paramCount = 5;
		return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8U8_RGBX);
	}
	else if (otype == VX_DF_IMAGE_UYVY) {
		if (inputMask != 7) return -1;
		anode->paramList[0] = paramList[4];
		anode->paramList[1] = paramList[0];
		anode->paramList[2] = paramList[1];
		anode->paramList[3] = paramList[2];
		anode->paramCount = 4;
		return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_UYVY);
	}
	else if (otype == VX_DF_IMAGE_YUYV) {
		if (inputMask != 7) return -1;
		anode->paramList[0] = paramList[4];
		anode->paramList[1] = paramList[0];
		anode->paramList[2] = paramList[1];
		anode->paramList[3] = paramList[2];
		anode->paramCount = 4;
		return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COMBINE_U32_U8U8U8_YUYV);
	}
	else if (otype == VX_DF_IMAGE_NV12) {
		if (inputMask != 7) return -1;
		anode->paramList[0] = paramList[4]->children[0];
		anode->paramList[1] = paramList[0]->children ? paramList[0]->children[0] : paramList[0];
		anode->paramCount = 2;
		if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
		anode->paramList[0] = paramList[4]->children[1];
		anode->paramList[1] = paramList[1];
		anode->paramList[2] = paramList[2];
		anode->paramCount = 3;
		return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COMBINE_U16_U8U8);
	}
	else if (otype == VX_DF_IMAGE_NV21) {
		if (inputMask != 7) return -1;
		anode->paramList[0] = paramList[4]->children[0];
		anode->paramList[1] = paramList[0]->children ? paramList[0]->children[0] : paramList[0];
		anode->paramCount = 2;
		if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
		anode->paramList[0] = paramList[4]->children[1];
		anode->paramList[1] = paramList[2];
		anode->paramList[2] = paramList[1];
		anode->paramCount = 3;
		return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COMBINE_U16_U8U8);
	}
	else if ((otype == VX_DF_IMAGE_IYUV) || (otype == VX_DF_IMAGE_YUV4)) {
		if (inputMask != 7) return -1;
		anode->paramList[0] = paramList[4]->children[0];
		anode->paramList[1] = paramList[0]->children ? paramList[0]->children[0] : paramList[0];
		anode->paramCount = 2;
		if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
		anode->paramList[0] = paramList[4]->children[1];
		anode->paramList[1] = paramList[1]->children ? paramList[1]->children[0] : paramList[1];
		anode->paramCount = 2;
		if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8)) return -1;
		anode->paramList[0] = paramList[4]->children[2];
		anode->paramList[1] = paramList[2]->children ? paramList[2]->children[0] : paramList[2];
		anode->paramCount = 2;
		return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8);
	}
	return -1;
}

int agoDramaDivideSobel3x3Node(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	if (paramList[1]) {
		anode->paramList[0] = paramList[1];
		anode->paramList[1] = paramList[0];
		anode->paramCount = 2;
		if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GX)) return -1;
	}
	if (paramList[2]) {
		anode->paramList[0] = paramList[2];
		anode->paramList[1] = paramList[0];
		anode->paramCount = 2;
		if (agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GY)) return -1;
	}
	return VX_SUCCESS;
}

int agoDramaDivideMagnitudeNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MAGNITUDE_S16_S16S16);
}

int agoDramaDividePhaseNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_PHASE_U8_S16S16);
}

int agoDramaDivideScaleImageNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	if (anode->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || anode->paramList[1]->u.img.format != VX_DF_IMAGE_U8) return -1;
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	// check for special no-scale case
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if ((paramList[0]->u.img.width == paramList[1]->u.img.width) && (paramList[0]->u.img.height == paramList[1]->u.img.height)) {
		// just perform copy
		anode->paramList[0] = paramList[1];
		anode->paramList[1] = paramList[0];
		anode->paramCount = 2;
		new_kernel_id = VX_KERNEL_AMD_CHANNEL_COPY_U8_U8;
	}
	else {
		vx_enum interpolation = paramList[2]->u.scalar.u.e;
		// identify scale kernel
		anode->paramList[0] = paramList[1];
		anode->paramList[1] = paramList[0];
		anode->paramCount = 2;
		if (anode->attr_border_mode.mode == VX_BORDER_MODE_UNDEFINED) {
			if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_NEAREST;
			else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR;
			else if (interpolation == VX_INTERPOLATION_TYPE_AREA) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_AREA;
		}
		else if (anode->attr_border_mode.mode == VX_BORDER_MODE_REPLICATE) {
			if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_NEAREST; // TBD remove -- this should be an error
			else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR_REPLICATE;
			else if (interpolation == VX_INTERPOLATION_TYPE_AREA) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_AREA; // TBD remove -- this should be an error
		}
		else if (anode->attr_border_mode.mode == VX_BORDER_MODE_CONSTANT) {
			if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_NEAREST; // TBD remove -- this should be an error
			else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) {
				new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR_CONSTANT;
				// create scalar object for border mode
				AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
				char desc[64]; snprintf(desc, sizeof(desc), "scalar-virtual:UINT8,%d", anode->attr_border_mode.constant_value.U8);
				AgoData * dataBorder = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
				if (!dataBorder) return -1;
				agoGenerateVirtualDataName(agraph, "scalar", dataBorder->name);
				agoAddData(&agraph->dataList, dataBorder);
				// make it 3rd argument
				anode->paramList[anode->paramCount++] = dataBorder;
			}
			else if (interpolation == VX_INTERPOLATION_TYPE_AREA) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_AREA; // TBD remove -- this should be an error
		}
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideTableLookupNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_LUT);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[1]->u.lut.type ==  VX_TYPE_UINT8)
		new_kernel_id = VX_KERNEL_AMD_LUT_U8_U8;
	else if(paramList[1]->u.lut.type ==  VX_TYPE_INT16)
		new_kernel_id = VX_KERNEL_AMD_LUT_S16_S16;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideHistogramNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_DISTRIBUTION);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_HISTOGRAM_DATA_U8);
}

int agoDramaDivideEqualizeHistogramNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	// create virtual histogram and look-up table objects
	AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
	AgoData * hist = agoCreateDataFromDescription(anode->ref.context, agraph, "distribution-virtual:256,0,256", false);
	AgoData * lut = agoCreateDataFromDescription(anode->ref.context, agraph, "lut-virtual:UINT8,256", false);
	if (!hist || !lut) return -1;
	agoGenerateVirtualDataName(agraph, "histogram", hist->name);
	agoGenerateVirtualDataName(agraph, "lut", lut->name);
	agoAddData(&agraph->dataList, hist);
	agoAddData(&agraph->dataList, lut);
	// histogram
	anode->paramList[0] = hist;
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	int status = agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_HISTOGRAM_DATA_U8);
	// equalization
	anode->paramList[0] = lut;
	anode->paramList[1] = hist;
	anode->paramCount = 2;
	status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_EQUALIZE_DATA_DATA);
	// table lookup
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = lut;
	anode->paramCount = 3;
	status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_LUT_U8_U8);
	return status;
}

int agoDramaDivideAbsdiffNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[2]->u.img.format == VX_DF_IMAGE_U8) new_kernel_id = VX_KERNEL_AMD_ABS_DIFF_U8_U8U8;
	else if (paramList[2]->u.img.format == VX_DF_IMAGE_S16) new_kernel_id = VX_KERNEL_AMD_ABS_DIFF_S16_S16S16_SAT;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideMeanStddevNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	// create virtual AGO_TYPE_MEANSTDDEV_DATA
	AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
	AgoData * data = agoCreateDataFromDescription(anode->ref.context, agraph, "ago-meanstddev-data-virtual:", false);
	if (!data) return -1;
	agoGenerateVirtualDataName(agraph, "meanstddev", data->name);
	agoAddData(&agraph->dataList, data);
	// compute sum and sum-of-squares
	anode->paramList[0] = data;
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	int status = 0;
	if(paramList[0]->u.img.format == VX_DF_IMAGE_U8)
		status = agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MEAN_STD_DEV_DATA_U8);
	else if(paramList[0]->u.img.format == VX_DF_IMAGE_U1)
		status = agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MEAN_STD_DEV_DATA_U1);
	// compute mean and average
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[2];
	anode->paramList[2] = data;
	anode->paramCount = 3;
	status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MEAN_STD_DEV_MERGE_DATA_DATA);
	return status;
}

int agoDramaDivideThresholdNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_THRESHOLD);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[1]->u.thr.thresh_type == VX_THRESHOLD_TYPE_BINARY && paramList[1]->u.thr.input_format == VX_DF_IMAGE_U8 && paramList[1]->u.thr.output_format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_THRESHOLD_U8_U8_BINARY;
	else if (paramList[1]->u.thr.thresh_type == VX_THRESHOLD_TYPE_BINARY && paramList[1]->u.thr.input_format == VX_DF_IMAGE_S16 && paramList[1]->u.thr.output_format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_THRESHOLD_U8_S16_BINARY;
	else if (paramList[1]->u.thr.thresh_type == VX_THRESHOLD_TYPE_BINARY && paramList[1]->u.thr.input_format == VX_DF_IMAGE_U8 && paramList[1]->u.thr.output_format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_THRESHOLD_U1_U8_BINARY;
	else if (paramList[1]->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE && paramList[1]->u.thr.input_format == VX_DF_IMAGE_U8 && paramList[1]->u.thr.output_format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_THRESHOLD_U8_U8_RANGE;
	else if (paramList[1]->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE && paramList[1]->u.thr.input_format == VX_DF_IMAGE_S16 && paramList[1]->u.thr.output_format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_THRESHOLD_U8_S16_RANGE;
	else if (paramList[1]->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE && paramList[1]->u.thr.input_format == VX_DF_IMAGE_U8 && paramList[1]->u.thr.output_format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_THRESHOLD_U1_U8_RANGE;
	/*if (paramList[1]->u.thr.thresh_type == VX_THRESHOLD_TYPE_BINARY) new_kernel_id = VX_KERNEL_AMD_THRESHOLD_U8_U8_BINARY;
	else if (paramList[1]->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) new_kernel_id = VX_KERNEL_AMD_THRESHOLD_U8_U8_RANGE;
	*/return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideIntegralImageNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_INTEGRAL_IMAGE_U32_U8);
}

int agoDramaDivideDilate3x3Node(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_DILATE_U8_U8_3x3;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_DILATE_U1_U8_3x3;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_DILATE_U8_U1_3x3;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_DILATE_U1_U1_3x3;
	
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideErode3x3Node(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;

	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_ERODE_U8_U8_3x3;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_ERODE_U1_U8_3x3;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_ERODE_U8_U1_3x3;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_ERODE_U1_U1_3x3;
	
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideMedian3x3Node(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MEDIAN_U8_U8_3x3);
}

int agoDramaDivideBox3x3Node(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_BOX_U8_U8_3x3);
}

int agoDramaDivideGaussian3x3Node(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_GAUSSIAN_U8_U8_3x3);
}

int agoDramaDivideCustomConvolutionNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_CONVOLUTION);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_df_image dst_image_format = paramList[2]->u.img.format;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if ((paramList[1]->u.conv.rows & 1) && (paramList[1]->u.conv.columns & 1)) new_kernel_id = (dst_image_format == VX_DF_IMAGE_U8) ? VX_KERNEL_AMD_CONVOLVE_U8_U8 : VX_KERNEL_AMD_CONVOLVE_S16_U8;
	else {
		agoAddLogEntry(&paramList[1]->ref, VX_FAILURE, "ERROR: agoDramaDivideCustomConvolutionNode: convolution size " VX_FMT_SIZE "x" VX_FMT_SIZE " not supported\n", paramList[1]->u.conv.rows, paramList[1]->u.conv.columns);
		return -1;
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideGaussianPyramidNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_PYRAMID);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	AgoData * nextInput = paramList[0]->children ? paramList[0]->children[0] : paramList[0];
	int status = 0;
	for (vx_uint32 level = 0; level < paramList[1]->numChildren; level++) {
		anode->paramList[0] = paramList[1]->children[level];
		anode->paramList[1] = nextInput;
		anode->paramCount = 2;
		vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
		if (level == 0) new_kernel_id = VX_KERNEL_AMD_CHANNEL_COPY_U8_U8;
		else if (paramList[1]->u.pyr.scale == VX_SCALE_PYRAMID_HALF) new_kernel_id = VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_5x5;
		else if (paramList[1]->u.pyr.scale == VX_SCALE_PYRAMID_ORB) new_kernel_id = VX_KERNEL_AMD_SCALE_GAUSSIAN_ORB_U8_U8_5x5;
		status |= agoDramaDivideAppend(nodeList, anode, new_kernel_id);
		nextInput = paramList[1]->children[level];
	}
	return status;
}

int agoDramaDivideAccumulateNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_ACCUMULATE_S16_S16U8_SAT);
}

int agoDramaDivideAccumulateWeightedNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_ACCUMULATE_WEIGHTED_U8_U8U8);
}

int agoDramaDivideAccumulateSquareNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_ACCUMULATE_SQUARED_S16_S16U8_SAT);
}

int agoDramaDivideMinmaxlocNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount < 3 || anode->paramCount > 7) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[3], VX_TYPE_ARRAY);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[4], VX_TYPE_ARRAY);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[5], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[6], VX_TYPE_SCALAR);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	// create virtual AGO_TYPE_MINMAXLOC_DATA
	AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
	AgoData * data = agoCreateDataFromDescription(anode->ref.context, agraph, "ago-minmaxloc-data-virtual:", false);
	AgoData * data_final = agoCreateDataFromDescription(anode->ref.context, agraph, "ago-minmaxloc-data-virtual:", false);
	if (!data || !data_final) return -1;
	agoGenerateVirtualDataName(agraph, "minmaxloc", data->name);
	agoGenerateVirtualDataName(agraph, "minmaxloc-final", data_final->name);
	agoAddData(&agraph->dataList, data);
	agoAddData(&agraph->dataList, data_final);
	// perform divide
	int status = 0;
	if (paramList[0]->u.img.format == VX_DF_IMAGE_U8) {
		anode->paramList[0] = data;
		anode->paramList[1] = paramList[0];
		anode->paramCount = 2;
		status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_DATA_U8);
		anode->paramList[0] = paramList[1];
		anode->paramList[1] = paramList[2];
		anode->paramList[2] = data_final;
		anode->paramList[3] = data;
		anode->paramCount = 4;
		status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_MERGE_DATA_DATA);
		if (paramList[3] && paramList[4]) {
			anode->paramList[0] = paramList[3];
			anode->paramList[1] = paramList[4];
			anode->paramList[2] = paramList[5];
			anode->paramList[3] = paramList[6];
			anode->paramList[4] = paramList[0];
			anode->paramList[5] = data_final;
			anode->paramCount = 6;
			status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MINMAX_COUNT_MINMAX);
		}
		else if(paramList[3]) {
			if (paramList[5] && !paramList[6]) {
				anode->paramList[0] = paramList[3];
				anode->paramList[1] = paramList[5];
				anode->paramList[2] = paramList[0];
				anode->paramList[3] = data_final;
				anode->paramCount = 4;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MIN_COUNT_MIN);
			}
			else {
				anode->paramList[0] = paramList[3];
				anode->paramList[1] = paramList[5];
				anode->paramList[2] = paramList[6];
				anode->paramList[3] = paramList[0];
				anode->paramList[4] = data_final;
				anode->paramCount = 5;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MIN_COUNT_MINMAX);
			}
		}
		else if (paramList[4]) {
			if (!paramList[5] && paramList[6]) {
				anode->paramList[0] = paramList[4];
				anode->paramList[1] = paramList[6];
				anode->paramList[2] = paramList[0];
				anode->paramList[3] = data_final;
				anode->paramCount = 4;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MAX_COUNT_MAX);
			}
			else {
				anode->paramList[0] = paramList[4];
				anode->paramList[1] = paramList[5];
				anode->paramList[2] = paramList[6];
				anode->paramList[3] = paramList[0];
				anode->paramList[4] = data_final;
				anode->paramCount = 5;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_MAX_COUNT_MINMAX);
			}
		}
		else {
			if (paramList[5] && paramList[6]) {
				anode->paramList[0] = paramList[5];
				anode->paramList[1] = paramList[6];
				anode->paramList[2] = paramList[0];
				anode->paramList[3] = data_final;
				anode->paramCount = 4;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_NONE_COUNT_MINMAX);
			}
			else if (paramList[5]) {
				anode->paramList[0] = paramList[5];
				anode->paramList[1] = paramList[0];
				anode->paramList[2] = data_final;
				anode->paramCount = 3;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_NONE_COUNT_MIN);
			}
			else if (paramList[6]) {
				anode->paramList[0] = paramList[6];
				anode->paramList[1] = paramList[0];
				anode->paramList[2] = data_final;
				anode->paramCount = 3;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8DATA_LOC_NONE_COUNT_MAX);
			}
		}
	}
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_S16) {
		anode->paramList[0] = data;
		anode->paramList[1] = paramList[0];
		anode->paramCount = 2;
		status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_DATA_S16);
		anode->paramList[0] = paramList[1];
		anode->paramList[1] = paramList[2];
		anode->paramList[2] = data_final;
		anode->paramList[3] = data;
		anode->paramCount = 4;
		status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_MERGE_DATA_DATA);
		if (paramList[3] && paramList[4]) {
			anode->paramList[0] = paramList[3];
			anode->paramList[1] = paramList[4];
			anode->paramList[2] = paramList[5];
			anode->paramList[3] = paramList[6];
			anode->paramList[4] = paramList[0];
			anode->paramList[5] = data_final;
			anode->paramCount = 6;
			status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MINMAX_COUNT_MINMAX);
		}
		else if (paramList[3]) {
			if (paramList[5] && !paramList[6]) {
				anode->paramList[0] = paramList[3];
				anode->paramList[1] = paramList[5];
				anode->paramList[2] = paramList[0];
				anode->paramList[3] = data_final;
				anode->paramCount = 4;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MIN_COUNT_MIN);
			}
			else {
				anode->paramList[0] = paramList[3];
				anode->paramList[1] = paramList[5];
				anode->paramList[2] = paramList[6];
				anode->paramList[3] = paramList[0];
				anode->paramList[4] = data_final;
				anode->paramCount = 5;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MIN_COUNT_MINMAX);
			}
		}
		else if (paramList[4]) {
			if (!paramList[5] && paramList[6]) {
				anode->paramList[0] = paramList[4];
				anode->paramList[1] = paramList[6];
				anode->paramList[2] = paramList[0];
				anode->paramList[3] = data_final;
				anode->paramCount = 4;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MAX_COUNT_MAX);
			}
			else {
				anode->paramList[0] = paramList[4];
				anode->paramList[1] = paramList[5];
				anode->paramList[2] = paramList[6];
				anode->paramList[3] = paramList[0];
				anode->paramList[4] = data_final;
				anode->paramCount = 5;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_MAX_COUNT_MINMAX);
			}
		}
		else {
			if (paramList[5] && paramList[6]) {
				anode->paramList[0] = paramList[5];
				anode->paramList[1] = paramList[6];
				anode->paramList[2] = paramList[0];
				anode->paramList[3] = data_final;
				anode->paramCount = 4;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_NONE_COUNT_MINMAX);
			}
			else if (paramList[5]) {
				anode->paramList[0] = paramList[5];
				anode->paramList[1] = paramList[0];
				anode->paramList[2] = data_final;
				anode->paramCount = 3;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_NONE_COUNT_MIN);
			}
			else if (paramList[6]) {
				anode->paramList[0] = paramList[6];
				anode->paramList[1] = paramList[0];
				anode->paramList[2] = data_final;
				anode->paramCount = 3;
				status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_MIN_MAX_LOC_DATA_S16DATA_LOC_NONE_COUNT_MAX);
			}
		}
	}
	else status = -1;
	return status;
}

int agoDramaDivideConvertDepthNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 4) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_SCALAR);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[3];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[1]->u.img.format == VX_DF_IMAGE_S16 || paramList[0]->u.img.format == VX_DF_IMAGE_U8) {
		new_kernel_id = VX_KERNEL_AMD_COLOR_DEPTH_S16_U8;
	}
	else if (paramList[1]->u.img.format == VX_DF_IMAGE_U8 || paramList[0]->u.img.format == VX_DF_IMAGE_S16) {
		if (paramList[2]->u.scalar.u.e == VX_CONVERT_POLICY_WRAP) new_kernel_id = VX_KERNEL_AMD_COLOR_DEPTH_U8_S16_WRAP;
		else if (paramList[2]->u.scalar.u.e == VX_CONVERT_POLICY_SATURATE) new_kernel_id = VX_KERNEL_AMD_COLOR_DEPTH_U8_S16_SAT;
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideAndNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_AND_U8_U8U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_AND_U1_U8U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_AND_U8_U8U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_AND_U1_U8U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_AND_U8_U1U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_AND_U1_U1U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_AND_U8_U1U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_AND_U1_U1U1;

	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideOrNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_OR_U8_U8U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_OR_U1_U8U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_OR_U8_U8U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_OR_U1_U8U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_OR_U8_U1U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_OR_U1_U1U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_OR_U8_U1U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_OR_U1_U1U1;

	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideXorNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_XOR_U8_U8U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_XOR_U1_U8U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_XOR_U8_U8U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_XOR_U1_U8U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_XOR_U8_U1U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_XOR_U1_U1U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_XOR_U8_U1U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1 && paramList[2]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_XOR_U1_U1U1;

	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideNotNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 2) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_NOT_U8_U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U8 && paramList[1]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_NOT_U1_U8;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U8) 
		new_kernel_id = VX_KERNEL_AMD_NOT_U8_U1;
	else if (paramList[0]->u.img.format == VX_DF_IMAGE_U1 && paramList[1]->u.img.format == VX_DF_IMAGE_U1) 
		new_kernel_id = VX_KERNEL_AMD_NOT_U1_U1;
	
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideMultiplyNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 6) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[4], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[5], VX_TYPE_IMAGE);
	// get and re-order parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	vx_uint32 paramCount = anode->paramCount;
	vx_df_image otype = paramList[5]->u.img.format;
	vx_df_image itypeA = paramList[0]->u.img.format;
	vx_df_image itypeB = paramList[1]->u.img.format;
	vx_enum overflow_policy = paramList[3]->u.scalar.u.e;
	vx_enum rounding_policy = paramList[4]->u.scalar.u.e;
	anode->paramList[0] = paramList[5];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramList[3] = paramList[2];
	anode->paramCount = 4;
	// divide
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_U8) && (otype == VX_DF_IMAGE_U8)) {
		if (rounding_policy == VX_ROUND_POLICY_TO_ZERO)
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_U8_U8U8_SAT_TRUNC : VX_KERNEL_AMD_MUL_U8_U8U8_WRAP_TRUNC;
		else
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_U8_U8U8_SAT_ROUND : VX_KERNEL_AMD_MUL_U8_U8U8_WRAP_ROUND;
	}
	else if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_U8) && (otype == VX_DF_IMAGE_S16)) {
		if (rounding_policy == VX_ROUND_POLICY_TO_ZERO)
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_S16_U8U8_SAT_TRUNC : VX_KERNEL_AMD_MUL_S16_U8U8_WRAP_TRUNC;
		else
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_S16_U8U8_SAT_ROUND : VX_KERNEL_AMD_MUL_S16_U8U8_WRAP_ROUND;
	}
	else if ((itypeA == VX_DF_IMAGE_S16) && (itypeB == VX_DF_IMAGE_U8) && (otype == VX_DF_IMAGE_S16)) {
		if (rounding_policy == VX_ROUND_POLICY_TO_ZERO)
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_S16_S16U8_SAT_TRUNC : VX_KERNEL_AMD_MUL_S16_S16U8_WRAP_TRUNC;
		else
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_S16_S16U8_SAT_ROUND : VX_KERNEL_AMD_MUL_S16_S16U8_WRAP_ROUND;
	}
	else if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_S16) && (otype == VX_DF_IMAGE_S16)) {
		if (rounding_policy == VX_ROUND_POLICY_TO_ZERO)
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_S16_S16U8_SAT_TRUNC : VX_KERNEL_AMD_MUL_S16_S16U8_WRAP_TRUNC;
		else
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_S16_S16U8_SAT_ROUND : VX_KERNEL_AMD_MUL_S16_S16U8_WRAP_ROUND;
		// switch A & B parameters
		anode->paramList[1] = paramList[1];
		anode->paramList[2] = paramList[0];
	}
	else if ((itypeA == VX_DF_IMAGE_S16) && (itypeB == VX_DF_IMAGE_S16) && (otype == VX_DF_IMAGE_S16)) {
		if (rounding_policy == VX_ROUND_POLICY_TO_ZERO)
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_S16_S16S16_SAT_TRUNC : VX_KERNEL_AMD_MUL_S16_S16S16_WRAP_TRUNC;
		else
			new_kernel_id = (overflow_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_MUL_S16_S16S16_SAT_ROUND : VX_KERNEL_AMD_MUL_S16_S16S16_WRAP_ROUND;
	}
	else if ((itypeA == VX_DF_IMAGE_RGB) && (itypeB == VX_DF_IMAGE_U8) && (otype == VX_DF_IMAGE_RGB)) {
		if (rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN && overflow_policy == VX_CONVERT_POLICY_SATURATE)
			new_kernel_id = VX_KERNEL_AMD_MUL_U24_U24U8_SAT_ROUND;
	}
	else if ((itypeA == VX_DF_IMAGE_RGBX) && (itypeB == VX_DF_IMAGE_U8) && (otype == VX_DF_IMAGE_RGBX)) {
		if (rounding_policy == VX_ROUND_POLICY_TO_NEAREST_EVEN && overflow_policy == VX_CONVERT_POLICY_SATURATE)
			new_kernel_id = VX_KERNEL_AMD_MUL_U32_U32U8_SAT_ROUND;
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideAddNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 4) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_IMAGE);
	// get and re-order parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	vx_uint32 paramCount = anode->paramCount;
	vx_df_image otype = paramList[3]->u.img.format;
	vx_df_image itypeA = paramList[0]->u.img.format;
	vx_df_image itypeB = paramList[1]->u.img.format;
	vx_enum convert_policy = paramList[2]->u.scalar.u.e;
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	// divide
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_U8) && (otype == VX_DF_IMAGE_U8)) {
		new_kernel_id = (convert_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_ADD_U8_U8U8_SAT : VX_KERNEL_AMD_ADD_U8_U8U8_WRAP;
	}
	else if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_U8) && (otype == VX_DF_IMAGE_S16)) {
		new_kernel_id = VX_KERNEL_AMD_ADD_S16_U8U8;
	}
	else if ((itypeA == VX_DF_IMAGE_S16) && (itypeB == VX_DF_IMAGE_U8) && (otype == VX_DF_IMAGE_S16)) {
		new_kernel_id = (convert_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_ADD_S16_S16U8_SAT : VX_KERNEL_AMD_ADD_S16_S16U8_WRAP;
	}
	else if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_S16) && (otype == VX_DF_IMAGE_S16)) {
		new_kernel_id = (convert_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_ADD_S16_S16U8_SAT : VX_KERNEL_AMD_ADD_S16_S16U8_WRAP;
		// switch A & B parameters
		anode->paramList[1] = paramList[1];
		anode->paramList[2] = paramList[0];
	}
	else if ((itypeA == VX_DF_IMAGE_S16) && (itypeB == VX_DF_IMAGE_S16) && (otype == VX_DF_IMAGE_S16)) {
		new_kernel_id = (convert_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_ADD_S16_S16S16_SAT : VX_KERNEL_AMD_ADD_S16_S16S16_WRAP;
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideSubtractNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 4) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_IMAGE);
	// get and re-order parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	vx_uint32 paramCount = anode->paramCount;
	vx_df_image otype = paramList[3]->u.img.format;
	vx_df_image itypeA = paramList[0]->u.img.format;
	vx_df_image itypeB = paramList[1]->u.img.format;
	vx_enum convert_policy = paramList[2]->u.scalar.u.e;
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	// divide
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (otype == VX_DF_IMAGE_U8) {
		if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_U8)) {
			new_kernel_id = (convert_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_SUB_U8_U8U8_SAT : VX_KERNEL_AMD_SUB_U8_U8U8_WRAP;
		}
	}
	else if (otype == VX_DF_IMAGE_S16) {
		if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_U8)) {
			new_kernel_id = VX_KERNEL_AMD_SUB_S16_U8U8;
		}
		else if ((itypeA == VX_DF_IMAGE_S16) && (itypeB == VX_DF_IMAGE_U8)) {
			new_kernel_id = (convert_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_SUB_S16_S16U8_SAT : VX_KERNEL_AMD_SUB_S16_S16U8_WRAP;
		}
		else if ((itypeA == VX_DF_IMAGE_U8) && (itypeB == VX_DF_IMAGE_S16)) {
			new_kernel_id = (convert_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_SUB_S16_U8S16_SAT : VX_KERNEL_AMD_SUB_S16_U8S16_WRAP;
		}
		else if ((itypeA == VX_DF_IMAGE_S16) && (itypeB == VX_DF_IMAGE_S16)) {
			new_kernel_id = (convert_policy == VX_CONVERT_POLICY_SATURATE) ? VX_KERNEL_AMD_SUB_S16_S16S16_SAT : VX_KERNEL_AMD_SUB_S16_S16S16_WRAP;
		}
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideHalfscaleGaussianNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 3) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (paramList[2]->u.scalar.u.i == 1) new_kernel_id = VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_NEAREST;
	else if (paramList[2]->u.scalar.u.i == 3) new_kernel_id = VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_3x3;
	else if (paramList[2]->u.scalar.u.i == 5) new_kernel_id = VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_5x5;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideRemapNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 4) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_REMAP);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum interpolation = paramList[2]->u.scalar.u.e;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (anode->paramList[0]->u.img.format == VX_DF_IMAGE_U8 && anode->paramList[1]->u.img.format == VX_DF_IMAGE_U8) {
		if (anode->attr_border_mode.mode == VX_BORDER_MODE_UNDEFINED) {
			if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_REMAP_U8_U8_NEAREST;
			else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) new_kernel_id = VX_KERNEL_AMD_REMAP_U8_U8_BILINEAR;
		}
		else if (anode->attr_border_mode.mode == VX_BORDER_MODE_CONSTANT) {
			if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_REMAP_U8_U8_NEAREST_CONSTANT;
			else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) new_kernel_id = VX_KERNEL_AMD_REMAP_U8_U8_BILINEAR_CONSTANT;
			if (new_kernel_id != VX_KERNEL_AMD_INVALID) {
				// create scalar object for border mode
				AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
				char desc[64]; snprintf(desc, sizeof(desc), "scalar-virtual:UINT8,%d", anode->attr_border_mode.constant_value.U8);
				AgoData * dataBorder = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
				if (!dataBorder) return -1;
				agoGenerateVirtualDataName(agraph, "scalar", dataBorder->name);
				agoAddData(&agraph->dataList, dataBorder);
				// make it 4th argument
				anode->paramList[anode->paramCount++] = dataBorder;
			}
		}
	}
	else if (anode->paramList[0]->u.img.format == VX_DF_IMAGE_RGB && anode->paramList[1]->u.img.format == VX_DF_IMAGE_RGB) {
		if (anode->attr_border_mode.mode == VX_BORDER_MODE_UNDEFINED && interpolation == VX_INTERPOLATION_TYPE_BILINEAR) 
			new_kernel_id = VX_KERNEL_AMD_REMAP_U24_U24_BILINEAR;
	}
	else if (anode->paramList[0]->u.img.format == VX_DF_IMAGE_RGB && anode->paramList[1]->u.img.format == VX_DF_IMAGE_RGBX) {
		if (anode->attr_border_mode.mode == VX_BORDER_MODE_UNDEFINED && interpolation == VX_INTERPOLATION_TYPE_BILINEAR)
			new_kernel_id = VX_KERNEL_AMD_REMAP_U24_U32_BILINEAR;
	}
	else if (anode->paramList[0]->u.img.format == VX_DF_IMAGE_RGBX && anode->paramList[1]->u.img.format == VX_DF_IMAGE_RGBX) {
		if (anode->attr_border_mode.mode == VX_BORDER_MODE_UNDEFINED && interpolation == VX_INTERPOLATION_TYPE_BILINEAR)
			new_kernel_id = VX_KERNEL_AMD_REMAP_U32_U32_BILINEAR;
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideWarpAffineNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 4) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_MATRIX);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum interpolation = paramList[2]->u.scalar.u.e;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (anode->attr_border_mode.mode == VX_BORDER_MODE_UNDEFINED) {
		if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_WARP_AFFINE_U8_U8_NEAREST;
		else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) new_kernel_id = VX_KERNEL_AMD_WARP_AFFINE_U8_U8_BILINEAR;
	}
	else if (anode->attr_border_mode.mode == VX_BORDER_MODE_CONSTANT) {
		if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_WARP_AFFINE_U8_U8_NEAREST_CONSTANT;
		else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) new_kernel_id = VX_KERNEL_AMD_WARP_AFFINE_U8_U8_BILINEAR_CONSTANT;
		if (new_kernel_id != VX_KERNEL_AMD_INVALID) {
			// create scalar object for border mode
			AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
			char desc[64]; snprintf(desc, sizeof(desc), "scalar-virtual:UINT8,%d", anode->attr_border_mode.constant_value.U8);
			AgoData * dataBorder = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
			if (!dataBorder) return -1;
			agoGenerateVirtualDataName(agraph, "scalar", dataBorder->name);
			agoAddData(&agraph->dataList, dataBorder);
			// make it 4th argument
			anode->paramList[anode->paramCount++] = dataBorder;
		}
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideWarpPerspectiveNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 4) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_MATRIX);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_IMAGE);
	// perform divide
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum interpolation = paramList[2]->u.scalar.u.e;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (anode->attr_border_mode.mode == VX_BORDER_MODE_UNDEFINED) {
		if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_WARP_PERSPECTIVE_U8_U8_NEAREST;
		else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) new_kernel_id = VX_KERNEL_AMD_WARP_PERSPECTIVE_U8_U8_BILINEAR;
	}
	else if (anode->attr_border_mode.mode == VX_BORDER_MODE_CONSTANT) {
		if (interpolation == VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR) new_kernel_id = VX_KERNEL_AMD_WARP_PERSPECTIVE_U8_U8_NEAREST_CONSTANT;
		else if (interpolation == VX_INTERPOLATION_TYPE_BILINEAR) new_kernel_id = VX_KERNEL_AMD_WARP_PERSPECTIVE_U8_U8_BILINEAR_CONSTANT;
		if (new_kernel_id != VX_KERNEL_AMD_INVALID) {
			// create scalar object for border mode
			AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
			char desc[64]; snprintf(desc, sizeof(desc), "scalar-virtual:UINT8,%d", anode->attr_border_mode.constant_value.U8);
			AgoData * dataBorder = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
			if (!dataBorder) return -1;
			agoGenerateVirtualDataName(agraph, "scalar", dataBorder->name);
			agoAddData(&agraph->dataList, dataBorder);
			// make it 4th argument
			anode->paramList[anode->paramCount++] = dataBorder;
		}
	}
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideCannyEdgeDetectorNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 5) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_THRESHOLD);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[4], VX_TYPE_IMAGE);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	vx_int32 gradient_size = paramList[2]->u.scalar.u.i;
	vx_enum norm_type = paramList[3]->u.scalar.u.e;
	// create virtual stack data for canny edges
	//   stack size: TBD (currently set the size of the image)
	vx_uint32 canny_stack_size = paramList[0]->u.img.width * paramList[0]->u.img.height;
	char desc[256]; snprintf(desc, sizeof(desc), "ago-canny-stack-virtual:%u", canny_stack_size);
	AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
	AgoData * data = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
	if (!data) return -1;
	agoGenerateVirtualDataName(agraph, "canny-stack", data->name);
	agoAddData(&agraph->dataList, data);
#if USE_AGO_CANNY_SOBEL_SUPP_THRESHOLD
	// compute sobel, nonmax-supression, and threshold
	anode->paramList[0] = paramList[4];
	anode->paramList[1] = data;
	anode->paramList[2] = paramList[0];
	anode->paramList[3] = paramList[1];
	anode->paramCount = 4;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (norm_type == VX_NORM_L1) {
		if (gradient_size == 3) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_3x3_L1NORM;
		else if (gradient_size == 5) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_5x5_L1NORM;
		else if (gradient_size == 7) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_7x7_L1NORM;
	}
	else if (norm_type == VX_NORM_L2) {
		if (gradient_size == 3) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_3x3_L2NORM;
		else if (gradient_size == 5) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_5x5_L2NORM;
		else if (gradient_size == 7) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_7x7_L2NORM;
	}
	int status = agoDramaDivideAppend(nodeList, anode, new_kernel_id);
#else
	// create virtual data for sobel output
	char descSobel[64]; snprintf(descSobel, sizeof(descSobel), "image-virtual:U016,%d,%d", paramList[0]->u.img.width, paramList[0]->u.img.height);
	AgoData * dataSobel = agoCreateDataFromDescription(anode->ref.context, agraph, descSobel, false);
	if (!dataSobel) return -1;
	agoGenerateVirtualDataName(agraph, "canny-sobel", dataSobel->name);
	agoAddData(&agraph->dataList, dataSobel);
	// compute sobel
	anode->paramList[0] = dataSobel;
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (norm_type == VX_NORM_L1) {
		if (gradient_size == 3) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L1NORM;
		else if (gradient_size == 5) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L1NORM;
		else if (gradient_size == 7) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L1NORM;
	}
	else if (norm_type == VX_NORM_L2) {
		if (gradient_size == 3) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_3x3_L2NORM;
		else if (gradient_size == 5) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_5x5_L2NORM;
		else if (gradient_size == 7) new_kernel_id = VX_KERNEL_AMD_CANNY_SOBEL_U16_U8_7x7_L2NORM;
	}
	int status = agoDramaDivideAppend(nodeList, anode, new_kernel_id);
	// compute nonmax-supression and threshold
	anode->paramList[0] = paramList[4];
	anode->paramList[1] = data;
	anode->paramList[2] = dataSobel;
	anode->paramList[3] = paramList[1];
	anode->paramList[4] = paramList[2];
	anode->paramCount = 5;
	if (gradient_size == 7) {
		status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_7x7);	
	}
	else
		status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_3x3);
#endif
	// run edge trace
	anode->paramList[0] = paramList[4];
	anode->paramList[1] = data;
	anode->paramCount = 2;
	status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_CANNY_EDGE_TRACE_U8_U8XY);
	return status;
}

int agoDramaDivideHarrisCornersNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 7 && anode->paramCount != 8) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[4], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[5], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[6], VX_TYPE_ARRAY);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[7], VX_TYPE_SCALAR);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	vx_int32 window_size = paramList[4]->u.scalar.u.i;
	vx_int32 block_size = paramList[5]->u.scalar.u.i;
	// create virtual images for HG3, HVC, and XYS
	AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
	char desc[64];
	snprintf(desc, sizeof(desc), "image-virtual:F332,%d,%d", paramList[0]->u.img.width, paramList[0]->u.img.height);
	AgoData * dataHG3 = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
	snprintf(desc, sizeof(desc), "image-virtual:F032,%d,%d", paramList[0]->u.img.width, paramList[0]->u.img.height);
	AgoData * dataHVC = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
	snprintf(desc, sizeof(desc), "array-virtual:KEYPOINT_XYS,%d", paramList[0]->u.img.width * paramList[0]->u.img.height); // TBD: this array can have smaller capacity
	AgoData * dataXYS = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
	snprintf(desc, sizeof(desc), "scalar-virtual:UINT32,%d", paramList[0]->u.img.width);
	AgoData * dataWidth = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
	snprintf(desc, sizeof(desc), "scalar-virtual:UINT32,%d", paramList[0]->u.img.height);
	AgoData * dataHeight = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false);
	if (!dataHG3 || !dataHVC || !dataXYS || !dataWidth || !dataHeight) return -1;
	agoGenerateVirtualDataName(agraph, "HG3", dataHG3->name);
	agoGenerateVirtualDataName(agraph, "HVC", dataHVC->name);
	agoGenerateVirtualDataName(agraph, "XYS", dataXYS->name);
	agoGenerateVirtualDataName(agraph, "Width", dataWidth->name);
	agoGenerateVirtualDataName(agraph, "Height", dataHeight->name);
	agoAddData(&agraph->dataList, dataHG3);
	agoAddData(&agraph->dataList, dataHVC);
	agoAddData(&agraph->dataList, dataXYS);
	agoAddData(&agraph->dataList, dataWidth);
	agoAddData(&agraph->dataList, dataHeight);
	// compute HG3
	anode->paramList[0] = dataHG3;
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	vx_enum new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (window_size == 3) new_kernel_id = VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_3x3;
	else if (window_size == 5) new_kernel_id = VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_5x5;
	else if (window_size == 7) new_kernel_id = VX_KERNEL_AMD_HARRIS_SOBEL_HG3_U8_7x7;
	else {
		agoAddLogEntry(&anode->ref, VX_FAILURE, "ERROR: agoDramaDivideHarrisCornersNode: unsupported windows size: %d\n", window_size);
		return -1;
	}
	int status = agoDramaDivideAppend(nodeList, anode, new_kernel_id);
	// compute HVC
	anode->paramList[0] = dataHVC;
	anode->paramList[1] = dataHG3;
	anode->paramList[2] = paramList[3];
	anode->paramList[3] = paramList[1];
	anode->paramList[4] = paramList[4];
	anode->paramCount = 5;
	new_kernel_id = VX_KERNEL_AMD_INVALID;
	if (block_size == 3) new_kernel_id = VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_3x3;
	else if (block_size == 5) new_kernel_id = VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_5x5;
	else if (block_size == 7) new_kernel_id = VX_KERNEL_AMD_HARRIS_SCORE_HVC_HG3_7x7;
	else {
		agoAddLogEntry(&anode->ref, VX_FAILURE, "ERROR: agoDramaDivideHarrisCornersNode: unsupported block size: %d\n", block_size);
		return -1;
	}
	status |= agoDramaDivideAppend(nodeList, anode, new_kernel_id);
	// non-max suppression
	anode->paramList[0] = dataXYS;
	anode->paramList[1] = dataHVC;
	anode->paramCount = 2;
	status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_NON_MAX_SUPP_XY_ANY_3x3);
	// sort and pick corners
	anode->paramList[0] = paramList[6];
	anode->paramList[1] = paramList[7];
	anode->paramList[2] = dataXYS;
	anode->paramList[3] = paramList[2];
	anode->paramList[4] = dataWidth;
	anode->paramList[5] = dataHeight;
	anode->paramCount = 6;
	status |= agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_HARRIS_MERGE_SORT_AND_PICK_XY_XYS);
	return status;
}

int agoDramaDivideFastCornersNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount < 4 || anode->paramCount > 5) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_ARRAY);
	SANITY_CHECK_DATA_TYPE_OPTIONAL(anode->paramList[4], VX_TYPE_SCALAR);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[4];
	anode->paramList[2] = paramList[0];
	anode->paramList[3] = paramList[1];
	anode->paramCount = 4;
	vx_enum new_kernel_id = VX_KERNEL_AMD_FAST_CORNERS_XY_U8_SUPRESSION;
	if (paramList[2]->u.scalar.u.i == 0) new_kernel_id = VX_KERNEL_AMD_FAST_CORNERS_XY_U8_NOSUPRESSION;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideOpticalFlowPyrLkNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	if (anode->paramCount != 10) return -1;
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_PYRAMID);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_PYRAMID);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_ARRAY);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_ARRAY);
	SANITY_CHECK_DATA_TYPE(anode->paramList[4], VX_TYPE_ARRAY);
	SANITY_CHECK_DATA_TYPE(anode->paramList[5], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[6], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[7], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[8], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[9], VX_TYPE_SCALAR);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
#if 0 // TBD -- enable this when low-level primitives are ready
	AgoGraph * agraph = (AgoGraph *)anode->ref.scope;
	vx_status status;
	char desc[256];
	// add VX_KERNEL_AMD_OPTICAL_FLOW_PREPARE_LK_XY_XY node
	snprintf(desc, sizeof(desc), "array-virtual:INT32,%d", paramList[1]->u.arr.capacity);
	AgoData * dataXYmap = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false); if (!dataXYmap) return -1; 
	dataXYmap->name = agoGenerateVirtualDataName(agraph, "XYmap"); agoAddData(&agraph->dataList, dataXYmap);
	snprintf(desc, sizeof(desc), "array-virtual:COORDINATES2D,%d", paramList[1]->u.arr.capacity);
	AgoData * dataXY0 = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false); if (!dataXY0) return -1;
	dataXY0->name = agoGenerateVirtualDataName(agraph, "XY"); agoAddData(&agraph->dataList, dataXY0);
	anode->paramList[0] = dataXY0;      // tmpXY
	anode->paramList[1] = dataXYmap;    // XYmap
	anode->paramList[2] = paramList[2]; // old_points
	anode->paramList[3] = paramList[3]; // new_points_estimates
	anode->paramList[4] = paramList[8]; // use_initial_estimate
	anode->paramCount = 5;
	status = agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_OPTICAL_FLOW_PREPARE_LK_XY_XY); if (status) return status;
	// add VX_KERNEL_AMD_OPTICAL_FLOW_IMAGE_LK_XY_XY node per each image in reverse order from the pyramids
	float scale = anode->paramList[0]->u.pyr.scale * anode->paramList[0]->u.pyr.levels;
	for (vx_int32 child = (vx_int32)anode->paramList[0]->u.pyr.levels - 1; child >= 0; child--) {
		AgoData * imgOld = paramList[0]->children[child]; if (!imgOld) return VX_ERROR_INVALID_REFERENCE;
		AgoData * imgNew = paramList[1]->children[child]; if (!imgNew) return VX_ERROR_INVALID_REFERENCE;
		snprintf(desc, sizeof(desc), "array-virtual:COORDINATES2D,%d", paramList[1]->u.arr.capacity);
		AgoData * dataXY1 = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false); if (!dataXY1) return -1;
		dataXY1->name = agoGenerateVirtualDataName(agraph, "XY"); agoAddData(&agraph->dataList, dataXY1);
		snprintf(desc, sizeof(desc), "scalar-virtual:FLOAT,%g", scale);
		AgoData * dataScale = agoCreateDataFromDescription(anode->ref.context, agraph, desc, false); if (!dataScale) return -1;
		dataScale->name = agoGenerateVirtualDataName(agraph, "scale"); agoAddData(&agraph->dataList, dataScale);
		anode->paramList[0] = dataXY1;      // new points
		anode->paramList[1] = dataXY0;      // old points
		anode->paramList[2] = imgOld;       // old image
		anode->paramList[3] = imgNew;       // new image
		anode->paramList[4] = paramList[5]; // termination
		anode->paramList[5] = paramList[6]; // epsilon
		anode->paramList[6] = paramList[7]; // num_iterations
		anode->paramList[7] = paramList[9]; // window_dimension 
		anode->paramList[8] = dataScale;    // scale factor
		anode->paramCount = 9;
		status = agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_OPTICAL_FLOW_IMAGE_LK_XY_XY); if (status) return status;
		// save dataXY1 for future reference and set scale factor to inverse of pyramid scale
		dataXY0 = dataXY1;
		scale = 1.0f / anode->paramList[0]->u.pyr.scale;
	}
	// add VX_KERNEL_AMD_OPTICAL_FLOW_FINAL_LK_XY_XY node
	anode->paramList[0] = paramList[4]; // new_points
	anode->paramList[1] = dataXY0;      // tmpXY
	anode->paramList[2] = dataXYmap;    // XYmap
	anode->paramCount = 3;
	status = agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_OPTICAL_FLOW_FINAL_LK_XY_XY); if (status) return status;
	return status;
#else
	anode->paramList[0] = paramList[4];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramList[3] = paramList[2];
	anode->paramList[4] = paramList[3];
	anode->paramList[5] = paramList[5];
	anode->paramList[6] = paramList[6];
	anode->paramList[7] = paramList[7];
	anode->paramList[8] = paramList[8];
	anode->paramList[9] = paramList[9];
	anode->paramCount = 10;
	return agoDramaDivideAppend(nodeList, anode, VX_KERNEL_AMD_OPTICAL_FLOW_PYR_LK_XY_XY);
#endif
}

int agoDramaDivideCopyNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], anode->paramList[0]->ref.type);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[1];
	anode->paramList[1] = paramList[0];
	anode->paramCount = 2;
	vx_enum new_kernel_id = VX_KERNEL_AMD_COPY_DATA_DATA;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideSelectNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], anode->paramList[1]->ref.type);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], anode->paramList[1]->ref.type);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramList[3] = paramList[2];
	anode->paramCount = 4;
	vx_enum new_kernel_id = VX_KERNEL_AMD_SELECT_DATA_DATA_DATA;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideWeightedAverageNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_IMAGE);
	// // save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramList[3] = paramList[2];
	anode->paramCount = 4;
	// TBD: use amd optimized kernel
	vx_enum new_kernel_id = VX_KERNEL_AMD_WEIGHTED_AVERAGE_U8_U8_U8;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideNonLinearFilterNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_SCALAR);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_MATRIX);
	SANITY_CHECK_DATA_TYPE(anode->paramList[3], VX_TYPE_IMAGE);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[3];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramList[3] = paramList[2];
	anode->paramCount = 4;
	vx_enum new_kernel_id = VX_KERNEL_AMD_NON_LINEAR_FILTER_DATA_DATA_DATA;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideLaplacianPyramidNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_PYRAMID);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_LAPLACIAN_PYRAMID_DATA_DATA_DATA;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideLaplacianReconstructNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// sanity checks
	SANITY_CHECK_DATA_TYPE(anode->paramList[0], VX_TYPE_PYRAMID);
	SANITY_CHECK_DATA_TYPE(anode->paramList[1], VX_TYPE_IMAGE);
	SANITY_CHECK_DATA_TYPE(anode->paramList[2], VX_TYPE_IMAGE);
	// save parameters
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	anode->paramList[0] = paramList[2];
	anode->paramList[1] = paramList[0];
	anode->paramList[2] = paramList[1];
	anode->paramCount = 3;
	vx_enum new_kernel_id = VX_KERNEL_AMD_LAPLACIAN_RECONSTRUCT_DATA_DATA_DATA;
	return agoDramaDivideAppend(nodeList, anode, new_kernel_id);
}

int agoDramaDivideNode(AgoNodeList * nodeList, AgoNode * anode)
{
	// save parameter list
	vx_uint32 paramCount = anode->paramCount;
	AgoData * paramList[AGO_MAX_PARAMS]; memcpy(paramList, anode->paramList, sizeof(paramList));
	// divide the node depending on the type
	int status = -1;
	switch (anode->akernel->id)
	{
		case VX_KERNEL_COLOR_CONVERT:
			status = agoDramaDivideColorConvertNode(nodeList, anode);
			break;
		case VX_KERNEL_CHANNEL_EXTRACT:
			status = agoDramaDivideChannelExtractNode(nodeList, anode);
			break;
		case VX_KERNEL_CHANNEL_COMBINE:
			status = agoDramaDivideChannelCombineNode(nodeList, anode);
			break;
		case VX_KERNEL_SOBEL_3x3:
			status = agoDramaDivideSobel3x3Node(nodeList, anode);
			break;
		case VX_KERNEL_MAGNITUDE:
			status = agoDramaDivideMagnitudeNode(nodeList, anode);
			break;
		case VX_KERNEL_PHASE:
			status = agoDramaDividePhaseNode(nodeList, anode);
			break;
		case VX_KERNEL_SCALE_IMAGE:
			status = agoDramaDivideScaleImageNode(nodeList, anode);
			break;
		case VX_KERNEL_TABLE_LOOKUP:
			status = agoDramaDivideTableLookupNode(nodeList, anode);
			break;
		case VX_KERNEL_HISTOGRAM:
			status = agoDramaDivideHistogramNode(nodeList, anode);
			break;
		case VX_KERNEL_EQUALIZE_HISTOGRAM:
			status = agoDramaDivideEqualizeHistogramNode(nodeList, anode);
			break;
		case VX_KERNEL_ABSDIFF:
			status = agoDramaDivideAbsdiffNode(nodeList, anode);
			break;
		case VX_KERNEL_MEAN_STDDEV:
			status = agoDramaDivideMeanStddevNode(nodeList, anode);
			break;
		case VX_KERNEL_THRESHOLD:
			status = agoDramaDivideThresholdNode(nodeList, anode);
			break;
		case VX_KERNEL_INTEGRAL_IMAGE:
			status = agoDramaDivideIntegralImageNode(nodeList, anode);
			break;
		case VX_KERNEL_DILATE_3x3:
			status = agoDramaDivideDilate3x3Node(nodeList, anode);
			break;
		case VX_KERNEL_ERODE_3x3:
			status = agoDramaDivideErode3x3Node(nodeList, anode);
			break;
		case VX_KERNEL_MEDIAN_3x3:
			status = agoDramaDivideMedian3x3Node(nodeList, anode);
			break;
		case VX_KERNEL_BOX_3x3:
			status = agoDramaDivideBox3x3Node(nodeList, anode);
			break;
		case VX_KERNEL_GAUSSIAN_3x3:
			status = agoDramaDivideGaussian3x3Node(nodeList, anode);
			break;
		case VX_KERNEL_CUSTOM_CONVOLUTION:
			status = agoDramaDivideCustomConvolutionNode(nodeList, anode);
			break;
		case VX_KERNEL_GAUSSIAN_PYRAMID:
			status = agoDramaDivideGaussianPyramidNode(nodeList, anode);
			break;
		case VX_KERNEL_ACCUMULATE:
			status = agoDramaDivideAccumulateNode(nodeList, anode);
			break;
		case VX_KERNEL_ACCUMULATE_WEIGHTED:
			status = agoDramaDivideAccumulateWeightedNode(nodeList, anode);
			break;
		case VX_KERNEL_ACCUMULATE_SQUARE:
			status = agoDramaDivideAccumulateSquareNode(nodeList, anode);
			break;
		case VX_KERNEL_MINMAXLOC:
			status = agoDramaDivideMinmaxlocNode(nodeList, anode);
			break;
		case VX_KERNEL_CONVERTDEPTH:
			status = agoDramaDivideConvertDepthNode(nodeList, anode);
			break;
		case VX_KERNEL_CANNY_EDGE_DETECTOR:
			status = agoDramaDivideCannyEdgeDetectorNode(nodeList, anode);
			break;
		case VX_KERNEL_AND:
			status = agoDramaDivideAndNode(nodeList, anode);
			break;
		case VX_KERNEL_OR:
			status = agoDramaDivideOrNode(nodeList, anode);
			break;
		case VX_KERNEL_XOR:
			status = agoDramaDivideXorNode(nodeList, anode);
			break;
		case VX_KERNEL_NOT:
			status = agoDramaDivideNotNode(nodeList, anode);
			break;
		case VX_KERNEL_MULTIPLY:
			status = agoDramaDivideMultiplyNode(nodeList, anode);
			break;
		case VX_KERNEL_ADD:
			status = agoDramaDivideAddNode(nodeList, anode);
			break;
		case VX_KERNEL_SUBTRACT:
			status = agoDramaDivideSubtractNode(nodeList, anode);
			break;
		case VX_KERNEL_WARP_AFFINE:
			status = agoDramaDivideWarpAffineNode(nodeList, anode);
			break;
		case VX_KERNEL_WARP_PERSPECTIVE:
			status = agoDramaDivideWarpPerspectiveNode(nodeList, anode);
			break;
		case VX_KERNEL_HARRIS_CORNERS:
			status = agoDramaDivideHarrisCornersNode(nodeList, anode);
			break;
		case VX_KERNEL_FAST_CORNERS:
			status = agoDramaDivideFastCornersNode(nodeList, anode);
			break;
		case VX_KERNEL_OPTICAL_FLOW_PYR_LK:
			status = agoDramaDivideOpticalFlowPyrLkNode(nodeList, anode);
			break;
		case VX_KERNEL_REMAP:
			status = agoDramaDivideRemapNode(nodeList, anode);
			break;
		case VX_KERNEL_HALFSCALE_GAUSSIAN:
			status = agoDramaDivideHalfscaleGaussianNode(nodeList, anode);
			break;
		case VX_KERNEL_COPY:
			status = agoDramaDivideCopyNode(nodeList, anode);
			break;
		case VX_KERNEL_SELECT:
			status = agoDramaDivideSelectNode(nodeList, anode);
			break;
		case VX_KERNEL_WEIGHTED_AVERAGE:
			status = agoDramaDivideWeightedAverageNode(nodeList, anode);
			break;
		case VX_KERNEL_NON_LINEAR_FILTER:
			status = agoDramaDivideNonLinearFilterNode(nodeList, anode);
			break;
		case VX_KERNEL_LAPLACIAN_PYRAMID:
			status = agoDramaDivideLaplacianPyramidNode(nodeList, anode);
			break;
		case VX_KERNEL_LAPLACIAN_RECONSTRUCT:
			status = agoDramaDivideLaplacianReconstructNode(nodeList, anode);
			break;
		default:
			break;
	}
	// revert parameter list
	anode->paramCount = paramCount;
	memcpy(anode->paramList, paramList, sizeof(anode->paramList));
	return status;
}

int agoOptimizeDramaDivide(AgoGraph * agraph)
{
	int astatus = 0;
	for (AgoNode * anode = agraph->nodeList.head, *aprev = 0; anode;) {
		// check if current node is a general VX node, that needs division
		if ((anode->akernel->flags & AGO_KERNEL_FLAG_GROUP_MASK) == AGO_KERNEL_FLAG_GROUP_OVX10) {
			// divide the current node
			if (!agoDramaDivideNode(&agraph->nodeList, anode)) {
				// remove and release the current node
				if (aprev) aprev->next = anode->next;
				else agraph->nodeList.head = anode->next;
				agraph->nodeList.count--;
				if (agraph->nodeList.tail == anode) {
					agraph->nodeList.tail = aprev;
				}
				AgoNode * next = anode->next;
				// move anode to trash
				anode->ref.internal_count = 0;
				anode->next = agraph->nodeList.trash;
				agraph->nodeList.trash = anode;
				// advance to next node
				anode = next;
			}
			else {
				// TBD: error handling
				agoAddLogEntry(&anode->akernel->ref, VX_FAILURE, "ERROR: agoOptimizeDramaDivide: failed for node %s (not implemented yet)\n", anode->akernel->name);
				astatus = -1;
				// advance to next node, since node divide failed
				aprev = anode;
				anode = anode->next;
			}
		}
		else if (anode->akernel->regen_callback_f) {
			// try regenerating the node
			anode->drama_divide_invoked = false;
			vx_bool replace_original = vx_true_e;
			vx_status status = anode->akernel->regen_callback_f(anode, agoDramaDivideAddNodeCallback, replace_original);
			if (status == VX_SUCCESS) {
				if (anode->drama_divide_invoked && replace_original) {
					// remove and release the current node
					if (aprev) aprev->next = anode->next;
					else agraph->nodeList.head = anode->next;
					agraph->nodeList.count--;
					if (agraph->nodeList.tail == anode) {
						agraph->nodeList.tail = aprev;
					}
					AgoNode * next = anode->next;
					// move anode to trash
					anode->ref.internal_count = 0;
					anode->next = agraph->nodeList.trash;
					agraph->nodeList.trash = anode;
					// advance to next node
					anode = next;
				}
				else {
					// advance to next node
					aprev = anode;
					anode = anode->next;
				}
			}
			else {
				// TBD: error handling
				agoAddLogEntry(&anode->akernel->ref, VX_FAILURE, "ERROR: agoOptimizeDramaDivide: failed for node %s\n", anode->akernel->name);
				astatus = -1;
				// advance to next node, since node divide failed
				aprev = anode;
				anode = anode->next;
			}
		}
		else {
			// advance to next node
			aprev = anode;
			anode = anode->next;
		}
	}
	return astatus;
}
