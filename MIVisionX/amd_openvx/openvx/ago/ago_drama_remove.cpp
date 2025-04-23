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

///////////////////////////////////////////////////////////////////////////////
// rule book for merge part of remove
#define CHILD(n)                   (0x10 | ((n) << 8))
#define SOLITARY                   AGO_MERGE_RULE_SOLITARY_FLAG // This should be 0x20
#define BYTE2U1                    (0x40)
#define WRITEONLY                  (0x80)
#define ARG_INDEX(arg_spec)        ((arg_spec) & 0x0f)
#define ARG_HAS_CHILD(arg_spec)    ((arg_spec) & 0x10)
#define ARG_GET_CHILD(arg_spec)    ((arg_spec) >> 8)
#define ARG_IS_SOLITARY(arg_spec)  ((arg_spec) & SOLITARY)
#define ARG_IS_BYTE2U1(arg_spec)   ((arg_spec) & BYTE2U1)
#define ARG_IS_WRITEONLY(arg_spec) ((arg_spec) & WRITEONLY)
static AgoNodeMergeRule s_merge_rule[] = {
		{ // RGB to YUV4
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_Y_RGB, { 2, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_U_RGB, { 3, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_V_RGB, { 4, 1 } },
			},
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_YUV4_RGB, { 2, 3, 4, 1 } },
			}
		},
		{ // RGB to NV12
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_Y_RGB, { 2, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGB, { 3, 1 } },
			},
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGB, { 2, 3, 1 } },
			}
		},
		{ // RGB to IYUV
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_Y_RGB, { 2, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_IU_RGB, { 3, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_IV_RGB, { 4, 1 } },
			},
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGB, { 2, 3, 4, 1 } },
			}
		},
		{ // RGB to IUV
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_IU_RGB, { 2, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_IV_RGB, { 3, 1 } },
			},
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGB, { 2, 3, 1 } },
			}
		},
		{ // RGBX to YUV4
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_Y_RGBX, { 2, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_U_RGBX, { 3, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_V_RGBX, { 4, 1 } },
			},
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_YUV4_RGBX, { 2, 3, 4, 1 } },
			}
		},
		{ // RGBX to NV12
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_Y_RGBX, { 2, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_UV12_RGBX, { 3, 1 } },
			},
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_NV12_RGBX, { 2, 3, 1 } },
			}
		},
		{ // RGBX to IYUV
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_Y_RGBX, { 2, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_IU_RGBX, { 3, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_IV_RGBX, { 4, 1 } },
			},
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_IYUV_RGBX, { 2, 3, 4, 1 } },
			}
		},
		{ // RGBX to IUV
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_IU_RGBX, { 2, 1 } },
				{ VX_KERNEL_AMD_COLOR_CONVERT_IV_RGBX, { 3, 1 } },
			},
			{
				{ VX_KERNEL_AMD_COLOR_CONVERT_IUV_RGBX, { 2, 3, 1 } },
			}
		},
		{ // combined channel extract of RGBX from RGBX
			{
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS0, { 2, 1 } },
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS1, { 3, 1 } },
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS2, { 4, 1 } },
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS3, { 5, 1 } },
			},
			{
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8U8U8U8_U32, { 2, 3, 4, 5, 1 } },
			}
		},
		{ // combined channel extract of RGB from RGBX
			{
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS0, { 2, 1 } },
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS1, { 3, 1 } },
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U32_POS2, { 4, 1 } },
			},
			{
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8U8U8_U32, { 2, 3, 4, 1 } },
			}
		},
		{ // combined channel extract of RGB from RGB
			{
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS0, { 2, 1 } },
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS1, { 3, 1 } },
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8_U24_POS2, { 4, 1 } },
			},
			{
				{ VX_KERNEL_AMD_CHANNEL_EXTRACT_U8U8U8_U24, { 2, 3, 4, 1 } },
			}
		},
		{ // SOBEL GX + GY = GXY
			{
				{ VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GX, { 2, 1 } },
				{ VX_KERNEL_AMD_SOBEL_S16_U8_3x3_GY, { 3, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SOBEL_S16S16_U8_3x3_GXY, { 2, 3, 1 } },
			}
		},
		{ // SOBEL + MAGNITUDE + PHASE = SOBEL_MAGNITUDE_PHASE
			{
				{ VX_KERNEL_AMD_SOBEL_S16S16_U8_3x3_GXY, { 2 | SOLITARY, 3 | SOLITARY, 1 } },
				{ VX_KERNEL_AMD_MAGNITUDE_S16_S16S16, { 4, 2 | SOLITARY, 3 | SOLITARY } },
				{ VX_KERNEL_AMD_PHASE_U8_S16S16, { 5, 2 | SOLITARY, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_SOBEL_MAGNITUDE_PHASE_S16U8_U8_3x3, { 4, 5, 1 } },
			}
		},
		{ // SOBEL + MAGNITUDE = SOBEL_MAGNITUDE
			{
				{ VX_KERNEL_AMD_SOBEL_S16S16_U8_3x3_GXY, { 2 | SOLITARY, 3 | SOLITARY, 1 } },
				{ VX_KERNEL_AMD_MAGNITUDE_S16_S16S16, { 4, 2 | SOLITARY, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_SOBEL_MAGNITUDE_S16_U8_3x3, { 4, 1 } },
			}
		},
		{ // SOBEL + PHASE = SOBEL_PHASE
			{
				{ VX_KERNEL_AMD_SOBEL_S16S16_U8_3x3_GXY, { 2 | SOLITARY, 3 | SOLITARY, 1 } },
				{ VX_KERNEL_AMD_PHASE_U8_S16S16, { 4, 2 | SOLITARY, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_SOBEL_PHASE_U8_U8_3x3, { 4, 1 } },
			}
		},
		{ // AND_U8_U8U8 + NOT_U8_U8 = NAND_U8_U8U8
			{
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 4, 2, 1 } },
			}
		},
		{ // OR_U8_U8U8 + NOT_U8_U8 = NOR_U8_U8U8
			{
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 4, 2, 1 } },
			}
		},
		{ // XOR_U8_U8U8 + NOT_U8_U8 = XNOR_U8_U8U8
			{
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 4, 2, 1 } },
			}
		},
		{ // NAND_U8_U8U8 + NOT_U8_U8 = AND_U8_U8U8
			{
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 4, 2, 1 } },
			}
		},
		{ // NOR_U8_U8U8 + NOT_U8_U8 = OR_U8_U8U8
			{
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 4, 2, 1 } },
			}
		},
		{ // XNOR_U8_U8U8 + NOT_U8_U8 = XOR_U8_U8U8
			{
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 4, 2, 1 } },
			}
		},
		{ // THRESHOLD + NOT = THRESHOLD_NOT (U8 U8 U8 BINARY)
			{
				{ VX_KERNEL_AMD_THRESHOLD_U8_U8_BINARY, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_THRESHOLD_NOT_U8_U8_BINARY, { 4, 2, 1 } },
			}
		},
		{ // THRESHOLD + NOT = THRESHOLD_NOT (U8 U8 U8 RANGE)
			{
				{ VX_KERNEL_AMD_THRESHOLD_U8_U8_RANGE, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_THRESHOLD_NOT_U8_U8_RANGE, { 4, 2, 1 } },
			}
		},
		{ // THRESHOLD + NOT = THRESHOLD_NOT (U8 U1 U8 BINARY)
			{
				{ VX_KERNEL_AMD_THRESHOLD_U1_U8_BINARY, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U1, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_THRESHOLD_NOT_U8_U8_BINARY, { 4, 2, 1 } },
			}
		},
		{ // THRESHOLD + NOT = THRESHOLD_NOT (U8 U1 U8 RANGE)
			{
				{ VX_KERNEL_AMD_THRESHOLD_U1_U8_RANGE, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U1, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_THRESHOLD_NOT_U8_U8_RANGE, { 4, 2, 1 } },
			}
		},
		{ // THRESHOLD + NOT = THRESHOLD_NOT (U1 U1 U8 BINARY)
			{
				{ VX_KERNEL_AMD_THRESHOLD_U1_U8_BINARY, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U1_U1, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_THRESHOLD_NOT_U1_U8_BINARY, { 4, 2, 1 } },
			}
		},
		{ // THRESHOLD + NOT = THRESHOLD_NOT (U1 U1 U8 RANGE)
			{
				{ VX_KERNEL_AMD_THRESHOLD_U1_U8_RANGE, { 3 | SOLITARY, 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U1_U1, { 4, 3 | SOLITARY } },
			},
			{
				{ VX_KERNEL_AMD_THRESHOLD_NOT_U1_U8_RANGE, { 4, 2, 1 } },
			}
		},
		{ // AND_U8_U8U8(same-inputs) = COPY
			{
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 2, 1, 1 } },
			},
			{
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 2, 1 } },
			}
		},
		{ // OR_U8_U8U8(same-inputs) = COPY
			{
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 2, 1, 1 } },
			},
			{
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 2, 1 } },
			}
		},
		{ // XOR_U8_U8U8(same-inputs) = ZERO
			{
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 2, 1, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
			}
		},
		{ // NAND_U8_U8U8(same-inputs) = NOT_U8_U8
			{
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 2, 1, 1 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
			}
		},
		{ // NOR_U8_U8U8(same-inputs) = NOT_U8_U8
			{
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 2, 1, 1 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
			}
		},
		{ // XNOR_U8_U8U8(same-inputs) = FF
			{
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 2, 1, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
			}
		},
		{ // 00-NOT to 00-FF
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
			}
		},
		{ // 00-AND to 00-00
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // 00-AND to 00-00
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // FF-AND to FF-COPY
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 2 } },
			}
		},
		{ // FF-AND to FF-COPY
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 1 } },
			}
		},
		{ // FF-OR to FF-FF
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // FF-OR to FF-FF
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // 00-OR to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 2 } },
			}
		},
		{ // 00-OR to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 1 } },
			}
		},
		{ // 00-XOR to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 2 } },
			}
		},
		{ // 00-XOR to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 1 } },
			}
		},
		{ // FF-XOR to FF-NOT
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 2 } },
			}
		},
		{ // FF-XOR to FF-NOT
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 1 } },
			}
		},
		{ // 00-NAND to 00-FF
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // 00-NAND to 00-FF
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // FF-NAND to FF-NOT
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 2 } },
			}
		},
		{ // FF-NAND to FF-NOT
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 1 } },
			}
		},
		{ // FF-NOR to FF-00
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // FF-NOR to FF-00
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // 00-NOR to 00-NOT
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 2 } },
			}
		},
		{ // 00-NOR to 00-NOT
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 1 } },
			}
		},
		{ // 00-XNOR to 00-NOT
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 2 } },
			}
		},
		{ // 00-XNOR to 00-NOT
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 1 } },
			}
		},
		{ // FF-XNOR to FF-COPY
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 2 } },
			}
		},
		{ // FF-XNOR to FF-COPY
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 1 } },
			}
		},
		{ // 00-ADD(wrap) to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_ADD_U8_U8U8_WRAP, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 2 } },
			}
		},
		{ // 00-ADD(wrap) to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_ADD_U8_U8U8_WRAP, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 1 } },
			}
		},
		{ // 00-ADD(sat) to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_ADD_U8_U8U8_SAT, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 2 } },
			}
		},
		{ // 00-ADD(sat) to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_ADD_U8_U8U8_SAT, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 1 } },
			}
		},
		{ // 00-SUB(wrap) to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_SUB_U8_U8U8_WRAP, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 2 } },
			}
		},
		{ // 00-SUB(sat) to 00-COPY
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_SUB_U8_U8U8_SAT, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 2 } },
			}
		},
		{ // 00-ACCUMULATE to 00
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_ACCUMULATE_S16_S16U8_SAT, { 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
			}
		},
		{ // 00-ACCUMULATE_SQUARED to 00
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
				{ VX_KERNEL_AMD_ACCUMULATE_SQUARED_S16_S16U8_SAT, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
			}
		},
		{ // NOT-NOT to NOT-COPY
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_NOT_U8_U8, { 3, 2 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, { 3, 1 } },
			}
		},
		{ // NOT-AND to NOT-00
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // NOT-AND to NOT-00
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_AND_U8_U8U8, { 3, 1, 2 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // NOT-OR to NOT-FF
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // NOT-OR to NOT-FF
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_OR_U8_U8U8, { 3, 1, 2 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // NOT-XOR to NOT-FF
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // NOT-XOR to NOT-FF
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_XOR_U8_U8U8, { 3, 1, 2 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // NOT-NAND to NOT-FF
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // NOT-NAND to NOT-FF
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_NAND_U8_U8U8, { 3, 1, 2 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 3 } },
			}
		},
		{ // NOT-NOR to NOT-00
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // NOT-NOR to NOT-00
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_NOR_U8_U8U8, { 3, 1, 2 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // NOT-XNOR to NOT-00
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 3, 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // NOT-XNOR to NOT-00
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_XNOR_U8_U8U8, { 3, 1, 2 } },
			},
			{
				{ VX_KERNEL_AMD_NOT_U8_U8, { 2, 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 3 } },
			}
		},
		{ // 00-DILATE to 00-00
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_ERODE_U8_U8_3x3, { 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
			}
		},
		{ // FF-DILATE to FF-FF
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_ERODE_U8_U8_3x3, { 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
			}
		},
		{ // 00-ERODE to 00-00
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_ERODE_U8_U8_3x3, { 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_00_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_00_U8, { 2 } },
			}
		},
		{ // FF-ERODE to FF-FF
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_ERODE_U8_U8_3x3, { 2, 1 } },
			},
			{
				{ VX_KERNEL_AMD_SET_FF_U8, { 1 } },
				{ VX_KERNEL_AMD_SET_FF_U8, { 2 } },
			}
		},
};
static vx_uint32 s_merge_rule_count = sizeof(s_merge_rule) / sizeof(s_merge_rule[0]);

///////////////////////////////////////////////////////////////////////////////
// rule book for VX_DF_IMAGE_U8 to VX_DF_IMAGE_U1_AMD conversion
typedef struct AgoImageU8toU1Rule_t {
	vx_enum    find_kernel_id;
	vx_int32   arg_index;
	vx_enum    replace_kernel_id;
} AgoImageU8toU1Rule;
static AgoImageU8toU1Rule s_U8toU1_rule[] = {
	// VX_KERNEL_AMD_NOT_* kernels
	{ VX_KERNEL_AMD_NOT_U8_U8, 1, VX_KERNEL_AMD_NOT_U8_U1 },
	{ VX_KERNEL_AMD_NOT_U1_U8, 1, VX_KERNEL_AMD_NOT_U1_U1 },
	{ VX_KERNEL_AMD_NOT_U8_U1, 0, VX_KERNEL_AMD_NOT_U1_U1 },
	// VX_KERNEL_AMD_AND_* kernels
	{ VX_KERNEL_AMD_AND_U8_U8U8, 2, VX_KERNEL_AMD_AND_U8_U8U1 },
	{ VX_KERNEL_AMD_AND_U8_U8U8, 1, VX_KERNEL_AMD_AND_U8_U1U8 },
	{ VX_KERNEL_AMD_AND_U8_U8U1, 1, VX_KERNEL_AMD_AND_U8_U1U1 },
	{ VX_KERNEL_AMD_AND_U8_U1U8, 2, VX_KERNEL_AMD_AND_U8_U1U1 },
	{ VX_KERNEL_AMD_AND_U8_U1U1, 0, VX_KERNEL_AMD_AND_U1_U1U1 },
	// VX_KERNEL_AMD_OR_* kernels
	{ VX_KERNEL_AMD_OR_U8_U8U8, 2, VX_KERNEL_AMD_OR_U8_U8U1 },
	{ VX_KERNEL_AMD_OR_U8_U8U8, 1, VX_KERNEL_AMD_OR_U8_U1U8 },
	{ VX_KERNEL_AMD_OR_U8_U8U1, 1, VX_KERNEL_AMD_OR_U8_U1U1 },
	{ VX_KERNEL_AMD_OR_U8_U1U8, 2, VX_KERNEL_AMD_OR_U8_U1U1 },
	{ VX_KERNEL_AMD_OR_U8_U1U1, 0, VX_KERNEL_AMD_OR_U1_U1U1 },
	// VX_KERNEL_AMD_XOR_* kernels
	{ VX_KERNEL_AMD_XOR_U8_U8U8, 2, VX_KERNEL_AMD_XOR_U8_U8U1 },
	{ VX_KERNEL_AMD_XOR_U8_U8U8, 1, VX_KERNEL_AMD_XOR_U8_U1U8 },
	{ VX_KERNEL_AMD_XOR_U8_U8U1, 1, VX_KERNEL_AMD_XOR_U8_U1U1 },
	{ VX_KERNEL_AMD_XOR_U8_U1U8, 2, VX_KERNEL_AMD_XOR_U8_U1U1 },
	{ VX_KERNEL_AMD_XOR_U8_U1U1, 0, VX_KERNEL_AMD_XOR_U1_U1U1 },
	// VX_KERNEL_AMD_NAND_* kernels
	{ VX_KERNEL_AMD_NAND_U8_U8U8, 2, VX_KERNEL_AMD_NAND_U8_U8U1 },
	{ VX_KERNEL_AMD_NAND_U8_U8U8, 1, VX_KERNEL_AMD_NAND_U8_U1U8 },
	{ VX_KERNEL_AMD_NAND_U8_U8U1, 1, VX_KERNEL_AMD_NAND_U8_U1U1 },
	{ VX_KERNEL_AMD_NAND_U8_U1U8, 2, VX_KERNEL_AMD_NAND_U8_U1U1 },
	{ VX_KERNEL_AMD_NAND_U8_U1U1, 0, VX_KERNEL_AMD_NAND_U1_U1U1 },
	// VX_KERNEL_AMD_NOR_* kernels
	{ VX_KERNEL_AMD_NOR_U8_U8U8, 2, VX_KERNEL_AMD_NOR_U8_U8U1 },
	{ VX_KERNEL_AMD_NOR_U8_U8U8, 1, VX_KERNEL_AMD_NOR_U8_U1U8 },
	{ VX_KERNEL_AMD_NOR_U8_U8U1, 1, VX_KERNEL_AMD_NOR_U8_U1U1 },
	{ VX_KERNEL_AMD_NOR_U8_U1U8, 2, VX_KERNEL_AMD_NOR_U8_U1U1 },
	{ VX_KERNEL_AMD_NOR_U8_U1U1, 0, VX_KERNEL_AMD_NOR_U1_U1U1 },
	// VX_KERNEL_AMD_XNOR_* kernels
	{ VX_KERNEL_AMD_XNOR_U8_U8U8, 2, VX_KERNEL_AMD_XNOR_U8_U8U1 },
	{ VX_KERNEL_AMD_XNOR_U8_U8U8, 1, VX_KERNEL_AMD_XNOR_U8_U1U8 },
	{ VX_KERNEL_AMD_XNOR_U8_U8U1, 1, VX_KERNEL_AMD_XNOR_U8_U1U1 },
	{ VX_KERNEL_AMD_XNOR_U8_U1U8, 2, VX_KERNEL_AMD_XNOR_U8_U1U1 },
	{ VX_KERNEL_AMD_XNOR_U8_U1U1, 0, VX_KERNEL_AMD_XNOR_U1_U1U1 },
	// VX_KERNEL_AMD_THRESHOLD_* kernels
	{ VX_KERNEL_AMD_THRESHOLD_U8_U8_BINARY,     0, VX_KERNEL_AMD_THRESHOLD_U1_U8_BINARY },
	{ VX_KERNEL_AMD_THRESHOLD_U8_U8_RANGE,      0, VX_KERNEL_AMD_THRESHOLD_U1_U8_RANGE },
	{ VX_KERNEL_AMD_THRESHOLD_NOT_U8_U8_BINARY, 0, VX_KERNEL_AMD_THRESHOLD_NOT_U1_U8_BINARY },
	{ VX_KERNEL_AMD_THRESHOLD_NOT_U8_U8_RANGE,  0, VX_KERNEL_AMD_THRESHOLD_NOT_U1_U8_RANGE },
	// VX_KERNEL_AMD_DILATE_* kernels
	{ VX_KERNEL_AMD_DILATE_U8_U8_3x3, 1, VX_KERNEL_AMD_DILATE_U8_U1_3x3 },
	{ VX_KERNEL_AMD_DILATE_U1_U8_3x3, 1, VX_KERNEL_AMD_DILATE_U1_U1_3x3 },
	{ VX_KERNEL_AMD_DILATE_U8_U1_3x3, 0, VX_KERNEL_AMD_DILATE_U1_U1_3x3 },
	// VX_KERNEL_AMD_ERODE_* kernels
	{ VX_KERNEL_AMD_ERODE_U8_U8_3x3, 1, VX_KERNEL_AMD_ERODE_U8_U1_3x3 },
	{ VX_KERNEL_AMD_ERODE_U1_U8_3x3, 1, VX_KERNEL_AMD_ERODE_U1_U1_3x3 },
	{ VX_KERNEL_AMD_ERODE_U8_U1_3x3, 0, VX_KERNEL_AMD_ERODE_U1_U1_3x3 },
	// VX_KERNEL_AMD_CHANNEL_COPY_* kernels
	{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U8, 1, VX_KERNEL_AMD_CHANNEL_COPY_U8_U1 },
	{ VX_KERNEL_AMD_CHANNEL_COPY_U1_U8, 1, VX_KERNEL_AMD_CHANNEL_COPY_U1_U1 },
	{ VX_KERNEL_AMD_CHANNEL_COPY_U8_U1, 0, VX_KERNEL_AMD_CHANNEL_COPY_U1_U1 },
};
static vx_uint32 s_U8toU1_rule_count = sizeof(s_U8toU1_rule) / sizeof(s_U8toU1_rule[0]);

int agoOptimizeDramaRemoveCopyNodes(AgoGraph * agraph)
{
	// find and remove COPY nodes with virtual buffers
	for (AgoNode * anode = agraph->nodeList.head; anode; anode = anode->next) {
		AgoKernel * akernel = anode->akernel;
		bool nodeCanBeRemoved = false;
		if (anode->akernel->id == VX_KERNEL_AMD_CHANNEL_COPY_U8_U8 || anode->akernel->id == VX_KERNEL_AMD_COPY_DATA_DATA)
		{
			// copy of a virtual data can be removed by just replacing the virtual data
			// TBD: need to handle possible optimizations with buffers in delay object
			AgoData * dstParam = anode->paramList[0];
			AgoData * srcParam = anode->paramList[1];
			if (anode->akernel->id == VX_KERNEL_AMD_COPY_DATA_DATA) {
				srcParam = anode->paramList[0];
				dstParam = anode->paramList[1];
			}
			bool replaceSrc = false;
			bool replaceDst = false;
			if (dstParam->isVirtual && !agoIsPartOfDelay(dstParam)) {
				replaceDst = true;
			}
			if (srcParam->isVirtual && !agoIsPartOfDelay(srcParam)) {
				replaceSrc = true;
			}
			if (replaceSrc && replaceDst) {
				// prioritize between src and dst
				if (dstParam->parent && srcParam->parent) {
					if (dstParam->parent->ref.type == VX_TYPE_PYRAMID && srcParam->parent->ref.type == VX_TYPE_PYRAMID) {
						// if both pyramids are used by a node, needs special handling
						if (dstParam->parent->inputUsageCount > 0 && srcParam->parent->inputUsageCount > 0) {
							// TBD: this needs to be optimized carefully
							replaceDst = false;
							replaceSrc = false;
						}
					}
					else if (dstParam->parent->ref.type == VX_TYPE_PYRAMID) {
						replaceDst = false;
					}
				}
				else if (dstParam->parent) {
					replaceDst = false;
				}
			}
			if (replaceDst) {
#if ENABLE_DEBUG_MESSAGES
				vx_char srcName[256], dstName[256];
				agoGetDataName(srcName, srcParam);
				agoGetDataName(dstName, dstParam);
				debug_printf("agoOptimizeDramaRemoveCopyNodes: replacing %s(dst) with %s(src)\n", dstName[0] ? dstName : "<?>", srcName[0] ? srcName : "<?>");
#endif
				nodeCanBeRemoved = true;
				// replace all occurances of dstParam with srcParam
				agoReplaceDataInGraph(agraph, dstParam, srcParam);
			}
			else if (replaceSrc) {
#if ENABLE_DEBUG_MESSAGES
				vx_char srcName[256], dstName[256];
				agoGetDataName(srcName, srcParam);
				agoGetDataName(dstName, dstParam);
				debug_printf("agoOptimizeDramaRemoveCopyNodes: replacing %s(src) with %s(dst)\n", srcName[0] ? srcName : "<?>", dstName[0] ? dstName : "<?>");
#endif
				nodeCanBeRemoved = true;
				// replace all occurances of srcParam with dstParam
				agoReplaceDataInGraph(agraph, srcParam, dstParam);
			}
		}
		if (nodeCanBeRemoved) {
			debug_printf("INFO: agoOptimizeDramaRemoveCopyNodes: removing node %s\n", anode->akernel->name);
			// remove the node
			if (agoRemoveNode(&agraph->nodeList, anode, true)) {
				agoAddLogEntry(&anode->akernel->ref, -1, "ERROR: agoOptimizeDramaRemoveCopyNodes: agoRemoveNode(*,%s) failed\n", anode->akernel->name);
				return -1;
			}
			// make only one change at a time
			return 1;
		}
	}
	// no changes happened to the graph
	return 0;
}

int agoOptimizeDramaRemoveNodesWithUnusedOutputs(AgoGraph * agraph)
{
	// find and remove nodes who's outputs are not used
	for (AgoNode * anode = agraph->nodeList.head; anode; anode = anode->next) {
		AgoKernel * akernel = anode->akernel;
		bool nodeCanBeRemoved = true;
		for (vx_uint32 arg = 0; arg < anode->paramCount; arg++) {
			if (anode->paramList[arg]) {
				vx_uint32 inputUsageCount = anode->paramList[arg]->inputUsageCount;
				for (AgoData * pdata = anode->paramList[arg]->parent; pdata; pdata = pdata->parent) {
					inputUsageCount += pdata->inputUsageCount;
				}
				if (anode->paramList[arg]->isVirtual && (akernel->argConfig[arg] & AGO_KERNEL_ARG_OUTPUT_FLAG) && (inputUsageCount > 0)) {
					// found a virtual output data that is being used elsewhere
					nodeCanBeRemoved = false;
					break;
				}
				else if (!anode->paramList[arg]->isVirtual && (akernel->argConfig[arg] & AGO_KERNEL_ARG_OUTPUT_FLAG)) {
					// found a physical output data that can be accessed by user
					nodeCanBeRemoved = false;
					break;
				}
			}
		}
		if (nodeCanBeRemoved) {
			debug_printf("INFO: agoOptimizeDramaRemoveNodesWithUnusedOutputs: removing node %s\n", anode->akernel->name);
			// remove the node
			if (agoRemoveNode(&agraph->nodeList, anode, true)) {
				agoAddLogEntry(&anode->akernel->ref, -1, "ERROR: agoOptimizeDramaRemoveNodesWithUnusedOutputs: agoRemoveNode(*,%s) failed\n", anode->akernel->name);
				return -1;
			}
			// make only one change at a time
			return 1;
		}
	}
	// no changes happened to the graph
	return 0;
}

int agoOptimizeDramaRemoveNodeMerge(AgoGraph * agraph)
{
	// apply node merge rules
	int ruleSet = 0;
	vx_uint32 rule_count = s_merge_rule_count;
	for (vx_uint32 iRule = 0; iRule <= rule_count; iRule++) {
		AgoNodeMergeRule * rule;
		if (iRule == rule_count) {
			if (ruleSet++ == 1)
				break;
			rule_count = (vx_uint32)agraph->ref.context->merge_rules.size();
			if (rule_count == 0)
				break;
			iRule = 0;
		}
		if (ruleSet == 0) {
			rule = &s_merge_rule[iRule];
		}
		else {
			rule = &agraph->ref.context->merge_rules[iRule];
		}
		// find match
		vx_uint32 numMatchNodes = 0;
		for (vx_uint32 iNode = 0; iNode < AGO_MERGE_RULE_MAX_FIND && rule->find[iNode].kernel_id; iNode++) {
			numMatchNodes++;
		}
		AgoData * mdata[AGO_MAX_PARAMS] = { 0 };
		AgoNode * stack[AGO_MERGE_RULE_MAX_FIND] { agraph->nodeList.head };
		vx_int32 stackTop = 0;
		for (;;) {
			bool foundMatch = false;
			if (stack[stackTop]->akernel->id == rule->find[stackTop].kernel_id) {
				foundMatch = true;
				memset(mdata, 0, sizeof(mdata));
				for (vx_int32 iNode = 0; iNode <= stackTop; iNode++) {
					for (vx_uint32 arg = 0; arg < AGO_MAX_PARAMS; arg++) {
						// get argument specificaiton from the rule of current node
						vx_uint32 arg_spec = rule->find[iNode].arg_spec[arg];
						if (arg_spec) {
							if (!(arg < stack[iNode]->paramCount && stack[iNode]->paramList[arg])) {
								// node doesn't have required argument
								foundMatch = false;
								break;
							}
						}
						else {
							if (arg < stack[iNode]->paramCount && stack[iNode]->paramList[arg]) {
								// node has argument that is missing in the rule
								foundMatch = false;
								break;
							}
							// this matches the rule
							continue;
						}
						AgoData * data = stack[iNode]->paramList[arg];

						// get argument info and sanity checks
						vx_int32 arg_index = ARG_INDEX(arg_spec);
						vx_int32 arg_child = ARG_HAS_CHILD(arg_spec) ? ARG_GET_CHILD(arg_spec) : -1;
						if (arg_child >= 0) {
							if (!data->parent || !(arg_child < (vx_int32)data->parent->numChildren) || !(data->parent->children[arg_child] == data)) {
								// node doesn't have required argument as a child
								foundMatch = false;
								break;
							}
							data = data->parent;
						}
						if (!mdata[arg_index]) {
							// save the data object for comparison with other parameter comparision
							mdata[arg_index] = data;
						}
						if (mdata[arg_index] != data) {
							// data doesn't match with previously saved parameter as dectated by the rule
							foundMatch = false;
							break;
						}
						if ((ARG_IS_SOLITARY(arg_spec) || ARG_IS_WRITEONLY(arg_spec)) && !data->isVirtual) {
							// data virtual properties doesn't match with the rule requirements
							foundMatch = false;
							break;
						}
						if (ARG_IS_WRITEONLY(arg_spec) && data->inputUsageCount > 0) {
							// data write-only properties doesn't match with the rule requirements
							foundMatch = false;
							break;
						}
					}
				}
			}
			// check if a match is found, proceed to next step in the search
			if (foundMatch) {
				if ((stackTop + 1) == numMatchNodes) {
					// check for virtual node removal criteria
					for (vx_int32 arg_index = 0; arg_index < AGO_MAX_PARAMS; arg_index++) {
						if (mdata[arg_index]) {
							// check if data in find rule spec is missing in the replace rule spec, or
							// solitary check is requested
							bool data_missing_in_replace = true;
							bool solitary_requested = false;
							for (vx_uint32 iNode = 0; iNode < AGO_MERGE_RULE_MAX_REPLACE && rule->replace[iNode].kernel_id; iNode++) {
								for (vx_uint32 arg = 0; arg < AGO_MAX_PARAMS; arg++) {
									vx_int32 arg_spec = rule->replace[iNode].arg_spec[arg];
									if (arg_spec) {
										if (arg_index == ARG_INDEX(arg_spec)) {
											data_missing_in_replace = false;
											if (ARG_IS_SOLITARY(arg_spec)) {
												solitary_requested = true;
											}
										}
									}
								}
							}
							if (data_missing_in_replace || solitary_requested) {
								// make sure that the data is virtual and no other nodes except nodes in the stack[] use this data
								if (!mdata[arg_index]->isVirtual)
									foundMatch = false;
								else {
									for (AgoNode * anode = agraph->nodeList.head; anode; anode = anode->next) {
										bool node_on_stack = false;
										for (vx_int32 i = 0; i <= stackTop; i++) {
											if (stack[i] == anode) {
												node_on_stack = true;
												break;
											}
										}
										if (!node_on_stack) {
											// check if data used by the node
											bool data_used_outside_rule = false;
											for (vx_uint32 i = 0; i < anode->paramCount; i++) {
												if (anode->paramList[i] == mdata[arg_index]) {
													data_used_outside_rule = true;
													break;
												}
											}
											if (data_used_outside_rule) {
												// the data can't be discarded by this rule
												foundMatch = false;
												break;
											}
										}
									}
								}
							}
						}
					}
					if (foundMatch) {
						// found a match to the complete rule
						stackTop++;
						break;
					}
				}
				if (foundMatch) {
					// skip to next-node in the rule and start searching
					stack[++stackTop] = agraph->nodeList.head;
				}
			}
			if(!foundMatch) {
				// skip to next node, since no match has been found at stackTop-node in the rule
				stack[stackTop] = stack[stackTop]->next;
				// when end-of-node-list is reached, go back one node in the rule and try next node
				while (!stack[stackTop]) {
					stackTop--;
					if (stackTop < 0) {
						break;
					}
					stack[stackTop] = stack[stackTop]->next;
				}
				if (stackTop < 0) {
					// reached end of search and no matched were found
					break;
				}
			}
		}

		if (stackTop == numMatchNodes) {
			// get affinity, border_mode, and callback attributes
			AgoTargetAffinityInfo_ attr_affinity = { 0 };
			vx_border_mode_t attr_border_mode = { 0 };
			vx_nodecomplete_f callback = NULL;
			for (vx_int32 iNode = 0; iNode < stackTop; iNode++) {
				if (stack[iNode]->callback) {
					callback = stack[iNode]->callback;
				}
				if (stack[iNode]->attr_affinity.device_type) {
					attr_affinity.device_type = stack[iNode]->attr_affinity.device_type;
				}
				if (stack[iNode]->attr_border_mode.mode) {
					// TBD: check whether to progate border mode
					// attr_border_mode = stack[iNode]->attr_border_mode;
				}
			}
			// add new nodes per rule's replace[] specification
			for (vx_uint32 iNode = 0; iNode < AGO_MERGE_RULE_MAX_REPLACE && rule->replace[iNode].kernel_id; iNode++) {
				// create a new AgoNode and add it to the nodeList
				AgoNode * childnode = agoCreateNode(agraph, rule->replace[iNode].kernel_id);
				for (vx_uint32 arg = 0; arg < AGO_MAX_PARAMS; arg++) {
					vx_int32 arg_spec = rule->replace[iNode].arg_spec[arg];
					if (arg_spec) {
						vx_int32 arg_index = ARG_INDEX(arg_spec);
						vx_int32 arg_child = ARG_HAS_CHILD(arg_spec) ? ARG_GET_CHILD(arg_spec) : -1;
						AgoData * data = mdata[arg_index];
						if (arg_child >= 0) {
							if (!(arg_child < (vx_int32)data->numChildren) || !data->children[arg_child]) {
								// TBD: error handling
								agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoOptimizeDramaRemoveNodeMerge: invalid child(%d) in arg:%d of replace-node:%d of rule:%d\n", arg_child, arg, iNode, iRule);
								return -1;
							}
							data = data->children[arg_child];
						}
						if (ARG_IS_BYTE2U1(arg_spec)) {
							// process the request to convert U8 image to U1 image
							if (data->ref.type == VX_TYPE_IMAGE && data->u.img.format == VX_DF_IMAGE_U8) {
								data->u.img.format = VX_DF_IMAGE_U1_AMD;
							}
						}
						childnode->paramList[arg] = data;
					}
				}
				// transfer configuration from rule to childnode
				childnode->attr_affinity = attr_affinity;
				//childnode->attr_border_mode = attr_border_mode;
				//childnode->callback = callback;
				debug_printf("INFO: agoOptimizeDramaRemoveNodeMerge: added node %s\n", childnode->akernel->name);
				// verify the node
				if (agoVerifyNode(childnode)) {
					return -1;
				}
			}
			// remove the nodes that matched with rule's find[]
			for (vx_int32 iNode = 0; iNode < stackTop; iNode++) {
				debug_printf("INFO: agoOptimizeDramaRemoveNodeMerge: removing node %s\n", stack[iNode]->akernel->name);
				if (agoRemoveNode(&agraph->nodeList, stack[iNode], true)) {
					agoAddLogEntry(&agraph->ref, VX_FAILURE, "ERROR: agoOptimizeDramaRemoveNodeMerge: agoRemoveNode(*,%s) failed\n", stack[iNode]->akernel->name);
					return -1;
				}
				stack[iNode] = 0;
			}
			// make only one change at a time
			return 1;
		}
	}

	// try special case node mapping
	for (AgoNode * node = agraph->nodeList.head; node; node = node->next)
	{
		AgoKernel * kernel = node->akernel;
		if (kernel->id == VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_NEAREST || kernel->id == VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR ||
			kernel->id == VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR_REPLICATE || kernel->id == VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_BILINEAR_CONSTANT ||
			kernel->id == VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_AREA)
		{
			AgoNode * childnode = NULL;
			AgoData * oImg = node->paramList[0];
			AgoData * iImg = node->paramList[1];
			// 1:1 scale image is same as channel copy
			vx_float32 offset = (kernel->id == VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_AREA) ? -0.5f : 0.0f;
			if ((iImg->u.img.width == oImg->u.img.width) && (iImg->u.img.height == oImg->u.img.height)) {
				// replace the node with VX_KERNEL_AMD_CHANNEL_COPY_U8_U8
				childnode = agoCreateNode(agraph, VX_KERNEL_AMD_CHANNEL_COPY_U8_U8);
				childnode->paramList[0] = oImg;
				childnode->paramList[1] = iImg;
			}
			// approaximate AREA interpolation mode with scale factors not greater than 1.0f with BILINEAR
			else if (kernel->id == VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_AREA && !((iImg->u.img.width > oImg->u.img.width) && (iImg->u.img.height > oImg->u.img.height))) {
				// replace the node with VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_NEAREST
				childnode = agoCreateNode(agraph, VX_KERNEL_AMD_SCALE_IMAGE_U8_U8_NEAREST);
				childnode->paramList[0] = oImg;
				childnode->paramList[1] = iImg;
			}
			if (childnode) {
				// transfer configuration from node to childnode
				childnode->attr_affinity = node->attr_affinity;
				agoImportNodeConfig(childnode, node);
				debug_printf("INFO: agoOptimizeDramaRemoveNodeMerge: added node %s\n", childnode->akernel->name);
				// remove the original node
				debug_printf("INFO: agoOptimizeDramaRemoveNodeMerge: removing node %s\n", node->akernel->name);
				if (agoRemoveNode(&agraph->nodeList, node, true)) {
					agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: agoOptimizeDramaRemoveNodeMerge: agoRemoveNode(*,%s) failed\n", node->akernel->name);
					return -1;
				}
				// verify the node
				if (agoVerifyNode(childnode)) {
					return -1;
				}
				// make only one change at a time
				return 1;
			}
		}
	}

	// no changes happened to the graph
	return 0;
}

int agoOptimizeDramaRemoveImageU8toU1(AgoGraph * agraph)
{
	int status = 0;
	// browse through all virtual data in the graph for VX_DF_IMAGE_U8 objects
	// that can be potentially converted into VX_DF_IMAGE_U1_AMD
	for (AgoData * adata = agraph->dataList.head; adata; adata = adata->next) {
		if (adata->ref.type == VX_TYPE_IMAGE &&
			adata->u.img.format == VX_DF_IMAGE_U8 && 
			adata->inputUsageCount >= 1 && 
			adata->outputUsageCount == 1 && 
			adata->inoutUsageCount == 0)
		{
			bool U8toU1_possible = true;

			// loop through all connected images, such as ROI
			AgoData * pdata = adata->u.img.roiMasterImage ? adata->u.img.roiMasterImage : adata;
			for (AgoData * data = agraph->dataList.head; data && U8toU1_possible; data = data->next)
			{
				if (data->ref.type == VX_TYPE_IMAGE && (data == adata || data->u.img.roiMasterImage == pdata))
				{
					// if ROI, make sure start_x and end_x are multiple of 8
					if (data->u.img.isROI && ((data->u.img.rect_roi.start_x & 7) || (data->u.img.rect_roi.end_x & 7))) {
						// can not convert it to U1 since ROI accesses on non-byte boundaries
						U8toU1_possible = false;
						break;
					}
					// make sure all the nodes that access this data can be converted to use VX_DF_IMAGE_U1_AMD
					for (AgoNode * anode = agraph->nodeList.head; anode; anode = anode->next) {
						vx_int32 arg_index = -1;
						for (vx_uint32 i = 0; i < anode->paramCount; i++) {
							if (anode->paramList[i] == data) {
								arg_index = i;
								break;
							}
						}
						// check if data is used by anode
						if (arg_index >= 0) {
							// check if anode is part of U8toU1 conversion rule
							bool matched = false;
							for (vx_uint32 rule = 0; rule < s_U8toU1_rule_count; rule++) {
								if (s_U8toU1_rule[rule].find_kernel_id == anode->akernel->id &&
									s_U8toU1_rule[rule].arg_index == arg_index)
								{
									matched = true;
									break;
								}
							}
							if (!matched) {
								// data is used by nodes that are not in U8toU1 conversion rule
								U8toU1_possible = false;
								break;
							}
						}
					}
				}
			}

			// if U8toU1_possible is TRUE:
			// - replace adata image type from VX_DF_IMAGE_U8 to VX_DF_IMAGE_U1_AMD
			// - change node type to use VX_DF_IMAGE_U1_AMD instead of VX_DF_IMAGE_U8
			if (U8toU1_possible) {
				// loop through all connected images, such as ROI
				AgoData * pdata = adata->u.img.roiMasterImage ? adata->u.img.roiMasterImage : adata;
				for (AgoData * data = agraph->dataList.head; data && U8toU1_possible; data = data->next)
				{
					if (data->ref.type == VX_TYPE_IMAGE && (data == adata || data->u.img.roiMasterImage == pdata))
					{
						data->u.img.format = VX_DF_IMAGE_U1_AMD;
						for (AgoNode * anode = agraph->nodeList.head; anode; anode = anode->next) {
							vx_int32 arg_index = -1;
							for (vx_uint32 i = 0; i < anode->paramCount; i++) {
								if (anode->paramList[i] == data) {
									arg_index = i;
									break;
								}
							}
							// check if data is used by anode
							if (arg_index >= 0) {
								// check if anode is part of U8toU1 conversion rule
								for (vx_uint32 rule = 0; rule < s_U8toU1_rule_count; rule++) {
									if (s_U8toU1_rule[rule].find_kernel_id == anode->akernel->id &&
										s_U8toU1_rule[rule].arg_index == arg_index)
									{
										anode->akernel = agoFindKernelByEnum(agraph->ref.context, s_U8toU1_rule[rule].replace_kernel_id);
										if (!anode->akernel) {
											agoAddLogEntry(&anode->ref, VX_FAILURE, "ERROR: agoOptimizeDramaRemoveImageU8toU1: agoFindKernelByEnum(0x%08x) failed for rule:%d\n", s_U8toU1_rule[rule].replace_kernel_id, rule);
											return -1;
										}
										break;
									}
								}
							}
						}
					}
				}
				// mark that graph has been modified
				status = 1;
			}
		}
	}
	return status;
}

int agoOptimizeDramaRemove(AgoGraph * agraph)
{
#if ENABLE_DEBUG_MESSAGES > 1
	int iteration = 0;
#endif
	for (int graphGotModified = !0; agraph->nodeList.head && graphGotModified;)
	{
		// check and mark data usage
		agoOptimizeDramaMarkDataUsage(agraph);

#if ENABLE_DEBUG_MESSAGES > 1
		printf("************************************************************************** agoOptimizeDramaRemove: ITER %04d\n", ++iteration);
		agoWriteGraph(agraph, NULL, 0, stdout, "[agoOptimizeDramaRemove]");
#endif

		if (!(agraph->optimizer_flags & AGO_GRAPH_OPTIMIZER_FLAG_NO_REMOVE_COPY_NODES)) {
			// try removing COPY nodes with virtual buffers
			if ((graphGotModified = agoOptimizeDramaRemoveCopyNodes(agraph)) < 0)
				return -1;
			if (graphGotModified)
				continue;
		}

		if (!(agraph->optimizer_flags & AGO_GRAPH_OPTIMIZER_FLAG_NO_REMOVE_UNUSED_OUTPUTS)) {
			// try remove nodes who's outputs are not used
			if ((graphGotModified = agoOptimizeDramaRemoveNodesWithUnusedOutputs(agraph)) < 0)
				return -1;
			if (graphGotModified)
				continue;
		}

		if (!(agraph->optimizer_flags & AGO_GRAPH_OPTIMIZER_FLAG_NO_NODE_MERGE)) {
			// try merging nodes that will further result in removal of redundancies
			if ((graphGotModified = agoOptimizeDramaRemoveNodeMerge(agraph)) < 0)
				return -1;
			if (graphGotModified)
				continue;
		}

		if (!(agraph->optimizer_flags & AGO_GRAPH_OPTIMIZER_FLAG_NO_CONVERT_8BIT_TO_1BIT)) {
			// try converting VX_DF_IMAGE_U8 images to VX_DF_IMAGE_U1_AMD images
			if ((graphGotModified = agoOptimizeDramaRemoveImageU8toU1(agraph)) < 0)
				return -1;
			if (graphGotModified)
				continue;
		}

		graphGotModified = 0;
	}
	return 0;
}
