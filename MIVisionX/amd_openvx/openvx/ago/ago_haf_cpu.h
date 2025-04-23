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


#ifndef __ago_haf_cpu_h__
#define __ago_haf_cpu_h__

#include <VX/vx.h>

#define TWOPI			6.283185307f
#define PI				3.1415926535898f
#define CAST_S16(x)		(int16_t)((x) < -32768 ? -32768 : (x) > 32767 ? 32767 : (x))
#define atan2_p0        (0.273*0.3183098862f)
#define atan2_p1		(0.9997878412794807f*57.29577951308232f)
#define atan2_p3		(-0.3258083974640975f*57.29577951308232f)
#define atan2_p5		(0.1555786518463281f*57.29577951308232f)
#define atan2_p7		(-0.04432655554792128f*57.29577951308232f)


typedef struct {
	vx_uint16 x;
	vx_uint16 y;
} ago_coord2d_ushort_t;

typedef struct {
	vx_int16 x;
	vx_int16 y;
} ago_coord2d_short_t;

typedef struct {
	vx_int32 x;
	vx_int32 y;
} ago_coord2d_int_t;

typedef struct {
	vx_float32 x;
	vx_float32 y;
} ago_coord2d_float_t;

typedef struct {
	vx_float32 matrix[3][2];
} ago_affine_matrix_t;

typedef struct {
	vx_float32 matrix[3][3];
} ago_perspective_matrix_t;

typedef struct AgoConfigScaleMatrix ago_scale_matrix_t;

typedef struct {
	vx_int16   x; // x-coordinate
	vx_int16   y; // y-coordinate
	vx_float32 s; // stregnth
} ago_keypoint_xys_t;

typedef struct {
	vx_uint32      width;
	vx_uint32      height;
	vx_uint32      strideInBytes;
	vx_uint8     * pImage;
	vx_bool        imageAlreadyComputed;
} ago_pyramid_u8_t;

typedef struct {
	vx_uint32  sampleCount;
	vx_float32 sum;
	vx_float32 sumSquared;
} ago_meanstddev_data_t;

typedef struct {
	vx_int32 min;
	vx_int32 max;
} ago_minmaxloc_data_t;

typedef struct {
	vx_float32 x;
	vx_float32 y;
} ago_keypoint_t;

typedef struct {
	vx_uint32 width;
	vx_uint32 height;
	vx_uint32 cellSize;
	vx_uint32 gridBufSize;
} ago_harris_grid_header_t;

int HafCpu_Not_U8_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Not_U8_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Not_U1_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Not_U1_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Lut_U8_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLut
	);
int HafCpu_Lut_S16_S16
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * pLut,
		vx_uint32     offset
	);
int HafCpu_Threshold_U8_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	);
int HafCpu_Threshold_U8_U8_Range
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      lower,
		vx_uint8      upper
	);
int HafCpu_Threshold_U1_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	);
int HafCpu_Threshold_U1_U8_Range
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      lower,
		vx_uint8      upper
	);
int HafCpu_Threshold_U8_S16_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16      threshold
	);
int HafCpu_Threshold_U8_S16_Range
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16      lower,
		vx_int16      upper
	);
int HafCpu_ThresholdNot_U8_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	);
int HafCpu_ThresholdNot_U8_U8_Range
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      lower,
		vx_uint8      upper
	);
int HafCpu_ThresholdNot_U1_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	);
int HafCpu_ThresholdNot_U1_U8_Range
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      lower,
		vx_uint8      upper
	);
int HafCpu_ThresholdNot_U8_S16_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	);
int HafCpu_ThresholdNot_U8_S16_Range
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      lower,
		vx_uint8      upper
	);
int HafCpu_ColorDepth_U8_S16_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int32      shift
	);
int HafCpu_ColorDepth_U8_S16_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int32      shift
	);
int HafCpu_ColorDepth_S16_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int32      shift
	);
int HafCpu_Add_U8_U8U8_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Add_U8_U8U8_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Sub_U8_U8U8_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Sub_U8_U8U8_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Mul_U8_U8U8_Wrap_Trunc
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_U8_U8U8_Wrap_Round
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_U8_U8U8_Sat_Trunc
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_U8_U8U8_Sat_Round
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_And_U8_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_And_U8_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_And_U8_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_And_U1_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_And_U1_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_And_U1_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Or_U8_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Or_U8_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Or_U8_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Or_U1_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Or_U1_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Or_U1_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xor_U8_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xor_U8_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xor_U8_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xor_U1_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xor_U1_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xor_U1_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nand_U8_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nand_U8_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nand_U8_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nand_U1_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nand_U1_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nand_U1_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nor_U8_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nor_U8_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nor_U8_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nor_U1_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nor_U1_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Nor_U1_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xnor_U8_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xnor_U8_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xnor_U8_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xnor_U1_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xnor_U1_U8U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Xnor_U1_U1U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_AbsDiff_U8_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_AccumulateWeighted_U8_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_float32    alpha
	);
int HafCpu_Add_S16_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Sub_S16_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Mul_S16_U8U8_Wrap_Trunc
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_U8U8_Wrap_Round
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_U8U8_Sat_Trunc
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_U8U8_Sat_Round
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Add_S16_S16U8_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Add_S16_S16U8_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Accumulate_S16_S16U8_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Sub_S16_S16U8_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Sub_S16_S16U8_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Mul_S16_S16U8_Wrap_Trunc
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_S16U8_Wrap_Round
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_S16U8_Sat_Trunc
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_S16U8_Sat_Round
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_AccumulateSquared_S16_S16U8_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint32     shift
	);
int HafCpu_Sub_S16_U8S16_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Sub_S16_U8S16_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_AbsDiff_S16_S16S16_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Add_S16_S16S16_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Add_S16_S16S16_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Sub_S16_S16S16_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Sub_S16_S16S16_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_Mul_S16_S16S16_Wrap_Trunc
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_S16S16_Wrap_Round
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_S16S16_Sat_Trunc
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Mul_S16_S16S16_Sat_Round
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_int16    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_float32    scale
	);
int HafCpu_Magnitude_S16_S16S16
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pMagImage,
		vx_uint32     magImageStrideInBytes,
		vx_int16    * pGxImage,
		vx_uint32     gxImageStrideInBytes,
		vx_int16    * pGyImage,
		vx_uint32     gyImageStrideInBytes
	);
int HafCpu_Phase_U8_S16S16
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pPhaseImage,
		vx_uint32     phaseImageStrideInBytes,
		vx_int16    * pGxImage,
		vx_uint32     gxImageStrideInBytes,
		vx_int16    * pGyImage,
		vx_uint32     gyImageStrideInBytes
	);
int HafCpu_MemSet_U8
	(
		vx_size       count,
		vx_uint8    * pDstBuf,
		vx_uint8      value
	);
int HafCpu_MemSet_U16
	(
		vx_size       count,
		vx_uint16   * pDstBuf,
		vx_uint16     value
	);
int HafCpu_MemSet_U24
	(
		vx_size       count,
		vx_uint8    * pDstBuf,
		vx_uint32     value
	);
int HafCpu_MemSet_U32
	(
		vx_size       count,
		vx_uint32   * pDstBuf,
		vx_uint32     value
	);
int HafCpu_BinaryCopy_U8_U8
	(
		vx_size       size,
		vx_uint8    * pDstBuf,
		vx_uint8    * pSrcBuf
	);
int HafCpu_ChannelCopy_U8_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_BufferCopyDisperseInDst
	(
		vx_uint32	  dstWidth,
		vx_uint32	  dstHeight,
		vx_uint32	  pixelSizeInBytes,
		vx_uint8	* pDstImage,
		vx_uint32	  dstImageStrideYInBytes,
		vx_uint32	  dstImageStrideXInBytes,
		vx_uint8	* pSrcImage,
		vx_uint32	  srcImageStrideYInBytes
	);
int HafCpu_BufferCopyDisperseInSrc
	(
		vx_uint32	  dstWidth,
		vx_uint32	  dstHeight,
		vx_uint32	  pixelSizeInBytes,
		vx_uint8	* pDstImage,
		vx_uint32	  dstImageStrideYInBytes,
		vx_uint8	* pSrcImage,
		vx_uint32	  srcImageStrideYInBytes,
		vx_uint32	  srcImageStrideXInBytes
	);
int HafCpu_ChannelCopy_U8_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelCopy_U1_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelCopy_U1_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U16_Pos0
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U16_Pos1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U24_Pos0
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U24_Pos1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U24_Pos2
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U32_Pos0
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U32_Pos1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U32_Pos2
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8_U32_Pos3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8U8U8_U24
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage0,
		vx_uint8    * pDstImage1,
		vx_uint8    * pDstImage2,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8U8U8_U32
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage0,
		vx_uint8    * pDstImage1,
		vx_uint8    * pDstImage2,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelExtract_U8U8U8U8_U32
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage0,
		vx_uint8    * pDstImage1,
		vx_uint8    * pDstImage2,
		vx_uint8    * pDstImage3,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ChannelCombine_U16_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes
	);
int HafCpu_ChannelCombine_U24_U8U8U8_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_ChannelCombine_U32_U8U8U8_UYVY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_ChannelCombine_U32_U8U8U8_YUYV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	);
int HafCpu_ChannelCombine_U32_U8U8U8U8_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_uint8    * pSrcImage3,
		vx_uint32     srcImage3StrideInBytes
	);
int HafCpu_ColorConvert_RGB_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_RGB_UYVY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_RGB_YUYV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_RGB_IYUV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcYImage,
		vx_uint32     srcYImageStrideInBytes,
		vx_uint8    * pSrcUImage,
		vx_uint32     srcUImageStrideInBytes,
		vx_uint8    * pSrcVImage,
		vx_uint32     srcVImageStrideInBytes
	);
int HafCpu_ColorConvert_RGB_NV12
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcLumaImage,
		vx_uint32     srcLumaImageStrideInBytes,
		vx_uint8    * pSrcChromaImage,
		vx_uint32     srcChromaImageStrideInBytes
	);
int HafCpu_ColorConvert_RGB_NV21
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcLumaImage,
		vx_uint32     srcLumaImageStrideInBytes,
		vx_uint8    * pSrcChromaImage,
		vx_uint32     srcChromaImageStrideInBytes
	);
int HafCpu_ColorConvert_RGBX_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_RGBX_UYVY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_RGBX_YUYV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_RGBX_IYUV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcYImage,
		vx_uint32     srcYImageStrideInBytes,
		vx_uint8    * pSrcUImage,
		vx_uint32     srcUImageStrideInBytes,
		vx_uint8    * pSrcVImage,
		vx_uint32     srcVImageStrideInBytes
	);
int HafCpu_ColorConvert_RGBX_NV12
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcLumaImage,
		vx_uint32     srcLumaImageStrideInBytes,
		vx_uint8    * pSrcChromaImage,
		vx_uint32     srcChromaImageStrideInBytes
	);
int HafCpu_ColorConvert_RGBX_NV21
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcLumaImage,
		vx_uint32     srcLumaImageStrideInBytes,
		vx_uint8    * pSrcChromaImage,
		vx_uint32     srcChromaImageStrideInBytes
	);
int HafCpu_ColorConvert_YUV4_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_YUV4_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ScaleUp2x2_U8_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_FormatConvert_UV_UV12
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcChromaImage,
		vx_uint32     srcChromaImageStrideInBytes
	);
int HafCpu_ColorConvert_IYUV_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_IYUV_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_FormatConvert_IYUV_UYVY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_FormatConvert_IYUV_YUYV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_FormatConvert_IUV_UV12
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcChromaImage,
		vx_uint32     srcChromaImageStrideInBytes
	);
int HafCpu_ColorConvert_NV12_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstLumaImage,
		vx_uint32     dstLumaImageStrideInBytes,
		vx_uint8    * pDstChromaImage,
		vx_uint32     dstChromaImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_NV12_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstLumaImage,
		vx_uint32     dstLumaImageStrideInBytes,
		vx_uint8    * pDstChromaImage,
		vx_uint32     dstChromaImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_FormatConvert_NV12_UYVY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstLumaImage,
		vx_uint32     dstLumaImageStrideInBytes,
		vx_uint8    * pDstChromaImage,
		vx_uint32     dstChromaImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_FormatConvert_NV12_YUYV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstLumaImage,
		vx_uint32     dstLumaImageStrideInBytes,
		vx_uint8    * pDstChromaImage,
		vx_uint32     dstChromaImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_FormatConvert_UV12_IUV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstChromaImage,
		vx_uint32     dstChromaImageStrideInBytes,
		vx_uint8    * pSrcUImage,
		vx_uint32     srcUImageStrideInBytes,
		vx_uint8    * pSrcVImage,
		vx_uint32     srcVImageStrideInBytes
	);
int HafCpu_ColorConvert_Y_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_Y_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_U_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_U_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_V_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_V_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_IU_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_IU_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_IV_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_IV_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_IUV_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_IUV_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_UV12_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImageChroma,
		vx_uint32     dstImageChromaStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_ColorConvert_UV12_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImageChroma,
		vx_uint32     dstImageChromaStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Box_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pScratch
	);
int HafCpu_Dilate_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Erode_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Median_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Gaussian_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	);
int HafCpu_ScaleGaussianHalf_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	);
int HafCpu_ScaleGaussianHalf_U8_U8_5x5
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		bool		  sampleFirstRow,
		bool		  sampleFirstColumn,
		vx_uint8	* pScratch
	);
int HafCpu_ScaleGaussianOrb_U8_U8_5x5
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pLocalData
	);
int HafCpu_Convolve_U8_U8_3xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size		  convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_U8_U8_5xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size		  convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_U8_U8_7xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size		  convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_U8_U8_9xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size		  convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_U8_U8_MxN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_uint32     convolutionWidth,
		vx_uint32     convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_S16_U8_3xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size		  convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_S16_U8_5xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size		  convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_S16_U8_7xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size		  convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_S16_U8_9xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size		  convolutionHeight,
		vx_int32      shift
	);
int HafCpu_Convolve_S16_U8_MxN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_uint32     convolutionWidth,
		vx_uint32     convolutionHeight,
		vx_int32      shift
	);
int HafCpu_SobelMagnitude_S16_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstMagImage,
		vx_uint32     dstMagImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_SobelPhase_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstPhaseImage,
		vx_uint32     dstPhaseImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	);
int HafCpu_SobelMagnitudePhase_S16U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstMagImage,
		vx_uint32     dstMagImageStrideInBytes,
		vx_uint8    * pDstPhaseImage,
		vx_uint32     dstPhaseImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Sobel_S16S16_U8_3x3_GXY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstGxImage,
		vx_uint32     dstGxImageStrideInBytes,
		vx_int16    * pDstGyImage,
		vx_uint32     dstGyImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	);
int HafCpu_Sobel_S16_U8_3x3_GX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstGxImage,
		vx_uint32     dstGxImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	);
int HafCpu_Sobel_S16_U8_3x3_GY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstGyImage,
		vx_uint32     dstGyImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	);
int HafCpu_Dilate_U1_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Erode_U1_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Dilate_U1_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Erode_U1_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Dilate_U8_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Erode_U8_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_FastCorners_XY_U8_Supression
	(
		vx_uint32       capacityOfDstCorner,
		vx_keypoint_t   dstCorner[],
		vx_uint32     * pDstCornerCount,
		vx_uint32       srcWidth,
		vx_uint32       srcHeight,
		vx_uint8      * pSrcImage,
		vx_uint32       srcImageStrideInBytes,
		vx_float32      strength_threshold,
		vx_uint8	  * pScratch
	);
int HafCpu_FastCorners_XY_U8_NoSupression
	(
		vx_uint32       capacityOfDstCorner,
		vx_keypoint_t   dstCorner[],
		vx_uint32     * pDstCornerCount,
		vx_uint32       srcWidth,
		vx_uint32       srcHeight,
		vx_uint8      * pSrcImage,
		vx_uint32       srcImageStrideInBytes,
		vx_float32      strength_threshold
	);
int HafCpu_HarrisSobel_HG3_U8_3x3
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	);
int HafCpu_HarrisSobel_HG3_U8_5x5
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	);
int HafCpu_HarrisSobel_HG3_U8_7x7
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	);
int HafCpu_HarrisScore_HVC_HG3_3x3
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	);
int HafCpu_HarrisScore_HVC_HG3_5x5
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	);
int HafCpu_HarrisScore_HVC_HG3_7x7
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	);
int HafCpu_NonMaxSupp_XY_ANY_3x3
	(
		vx_uint32            capacityOfList,
		ago_keypoint_xys_t * dstList,
		vx_uint32          * pDstListCount,
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_float32         * pSrcImg,
		vx_uint32            srcStrideInBytes
	);
int HafCpu_HarrisMergeSortAndPick_XY_XYS
	(
		vx_uint32                  capacityOfDstCorner,
		vx_keypoint_t            * dstCorner,
		vx_uint32                * pDstCornerCount,
		ago_keypoint_xys_t       * srcList,
		vx_uint32                  srcListCount,
		vx_float32                 min_distance,
		ago_harris_grid_header_t * gridInfo,
		ago_coord2d_short_t      * gridBuf
	);
int HafCpu_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper,
		vx_uint8			 * pScratch
	);
int HafCpu_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	);
int HafCpu_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	);
int HafCpu_CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	);
int HafCpu_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	);
int HafCpu_CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	);
int HafCpu_CannySobel_U16_U8_3x3_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	);
int HafCpu_CannySobel_U16_U8_3x3_L2NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	);
int HafCpu_CannySobel_U16_U8_5x5_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	);
int HafCpu_CannySobel_U16_U8_5x5_L2NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	);
int HafCpu_CannySobel_U16_U8_7x7_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	);
int HafCpu_CannySobel_U16_U8_7x7_L2NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	);
int HafCpu_CannySuppThreshold_U8XY_U16_3x3
	(
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32            * pxyStackTop,
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDst,
		vx_uint32              dstStrideInBytes,
		vx_uint16            * pSrc,
		vx_uint32              srcStrideInBytes,
		vx_uint16               hyst_lower,
		vx_uint16               hyst_upper
	);
int HafCpu_Remap_U8_U8_Nearest
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              srcWidth,
		vx_uint32              srcHeight,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		ago_coord2d_ushort_t * pMap,
		vx_uint32              mapStrideInBytes
	);
int HafCpu_Remap_U8_U8_Nearest_Constant
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              srcWidth,
		vx_uint32              srcHeight,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		ago_coord2d_ushort_t * pMap,
		vx_uint32              mapStrideInBytes,
		vx_uint8               border
	);
int HafCpu_Remap_U8_U8_Bilinear
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              srcWidth,
		vx_uint32              srcHeight,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		ago_coord2d_ushort_t * pMap,
		vx_uint32              mapStrideInBytes
	);
int HafCpu_Remap_U8_U8_Bilinear_Constant
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              srcWidth,
		vx_uint32              srcHeight,
		vx_uint8             * pSrcImage,
		vx_uint32              srcImageStrideInBytes,
		ago_coord2d_ushort_t * pMap,
		vx_uint32              mapStrideInBytes,
		vx_uint8               border
	);
int HafCpu_WarpAffine_U8_U8_Nearest
	(
		vx_uint32             dstWidth,
		vx_uint32             dstHeight,
		vx_uint8            * pDstImage,
		vx_uint32             dstImageStrideInBytes,
		vx_uint32             srcWidth,
		vx_uint32             srcHeight,
		vx_uint8            * pSrcImage,
		vx_uint32             srcImageStrideInBytes,
		ago_affine_matrix_t * matrix,
		vx_uint8			* pLocalData
	);
int HafCpu_WarpAffine_U8_U8_Nearest_Constant
	(
		vx_uint32             dstWidth,
		vx_uint32             dstHeight,
		vx_uint8            * pDstImage,
		vx_uint32             dstImageStrideInBytes,
		vx_uint32             srcWidth,
		vx_uint32             srcHeight,
		vx_uint8            * pSrcImage,
		vx_uint32             srcImageStrideInBytes,
		ago_affine_matrix_t * matrix,
		vx_uint8              border,
		vx_rectangle_t        rect_valid,
		vx_uint8			* pLocalData
	);
int HafCpu_WarpAffine_U8_U8_Bilinear
	(
		vx_uint32             dstWidth,
		vx_uint32             dstHeight,
		vx_uint8            * pDstImage,
		vx_uint32             dstImageStrideInBytes,
		vx_uint32             srcWidth,
		vx_uint32             srcHeight,
		vx_uint8            * pSrcImage,
		vx_uint32             srcImageStrideInBytes,
		ago_affine_matrix_t * matrix,
		vx_uint8			* pLocalData
	);
int HafCpu_WarpAffine_U8_U8_Bilinear_Constant
	(
		vx_uint32             dstWidth,
		vx_uint32             dstHeight,
		vx_uint8            * pDstImage,
		vx_uint32             dstImageStrideInBytes,
		vx_uint32             srcWidth,
		vx_uint32             srcHeight,
		vx_uint8            * pSrcImage,
		vx_uint32             srcImageStrideInBytes,
		ago_affine_matrix_t * matrix,
		vx_uint8              border,
		vx_uint8			* pLocalData
	);
int HafCpu_WarpPerspective_U8_U8_Nearest
	(
		vx_uint32                  dstWidth,
		vx_uint32                  dstHeight,
		vx_uint8                 * pDstImage,
		vx_uint32                  dstImageStrideInBytes,
		vx_uint32                  srcWidth,
		vx_uint32                  srcHeight,
		vx_uint8                 * pSrcImage,
		vx_uint32                  srcImageStrideInBytes,
		ago_perspective_matrix_t * matrix,
		vx_uint8				 * pLocalData
	);
int HafCpu_WarpPerspective_U8_U8_Nearest_Constant
	(
		vx_uint32                  dstWidth,
		vx_uint32                  dstHeight,
		vx_uint8                 * pDstImage,
		vx_uint32                  dstImageStrideInBytes,
		vx_uint32                  srcWidth,
		vx_uint32                  srcHeight,
		vx_uint8                 * pSrcImage,
		vx_uint32                  srcImageStrideInBytes,
		ago_perspective_matrix_t * matrix,
		vx_uint8                   border,
		vx_uint8				 * pLocalData
	);
int HafCpu_WarpPerspective_U8_U8_Bilinear
	(
		vx_uint32                  dstWidth,
		vx_uint32                  dstHeight,
		vx_uint8                 * pDstImage,
		vx_uint32                  dstImageStrideInBytes,
		vx_uint32                  srcWidth,
		vx_uint32                  srcHeight,
		vx_uint8                 * pSrcImage,
		vx_uint32                  srcImageStrideInBytes,
		ago_perspective_matrix_t * matrix,
		vx_uint8				 * pLocalData
	);
int HafCpu_WarpPerspective_U8_U8_Bilinear_Constant
	(
		vx_uint32                  dstWidth,
		vx_uint32                  dstHeight,
		vx_uint8                 * pDstImage,
		vx_uint32                  dstImageStrideInBytes,
		vx_uint32                  srcWidth,
		vx_uint32                  srcHeight,
		vx_uint8                 * pSrcImage,
		vx_uint32                  srcImageStrideInBytes,
		ago_perspective_matrix_t * matrix,
		vx_uint8                   border,
		vx_uint8				 * pLocalData
	);
int HafCpu_ScaleImage_U8_U8_Nearest
	(
		vx_uint32            dstWidth,
		vx_uint32            dstHeight,
		vx_uint8           * pDstImage,
		vx_uint32            dstImageStrideInBytes,
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes,
		ago_scale_matrix_t * matrix
	);
int HafCpu_ScaleImage_U8_U8_Nearest_Constant
	(
		vx_uint32            dstWidth,
		vx_uint32            dstHeight,
		vx_uint8           * pDstImage,
		vx_uint32            dstImageStrideInBytes,
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes,
		ago_scale_matrix_t * matrix,
		vx_uint8             border
	);
int HafCpu_ScaleImage_U8_U8_Bilinear
	(
		vx_uint32            dstWidth,
		vx_uint32            dstHeight,
		vx_uint8           * pDstImage,
		vx_uint32            dstImageStrideInBytes,
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes,
		ago_scale_matrix_t * matrix
	);
int HafCpu_ScaleImage_U8_U8_Bilinear_Replicate
	(
		vx_uint32            dstWidth,
		vx_uint32            dstHeight,
		vx_uint8           * pDstImage,
		vx_uint32            dstImageStrideInBytes,
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes,
		ago_scale_matrix_t * matrix
	);
int HafCpu_ScaleImage_U8_U8_Bilinear_Constant
	(
		vx_uint32            dstWidth,
		vx_uint32            dstHeight,
		vx_uint8           * pDstImage,
		vx_uint32            dstImageStrideInBytes,
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes,
		ago_scale_matrix_t * matrix,
		vx_uint8             border
	);
int HafCpu_ScaleImage_U8_U8_Area
	(
		vx_uint32            dstWidth,
		vx_uint32            dstHeight,
		vx_uint8           * pDstImage,
		vx_uint32            dstImageStrideInBytes,
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes,
		ago_scale_matrix_t * matrix
	);
int HafCpu_OpticalFlowPyrLK_XY_XY_Generic
(
	vx_keypoint_t      newKeyPoint[],
	vx_float32         pyramidScale,
	vx_uint32          pyramidLevelCount,
	ago_pyramid_u8_t * oldPyramid,
	ago_pyramid_u8_t * newPyramid,
	vx_uint32          keyPointCount,
	vx_keypoint_t      oldKeyPoint[],
	vx_keypoint_t      newKeyPointEstimate[],
	vx_enum            termination,
	vx_float32         epsilon,
	vx_uint32          num_iterations,
	vx_bool            use_initial_estimate,
	vx_uint32		   dataStrideInBytes,
	vx_uint8		 * DataPtr,
	vx_int32		   window_dimension
);

int HafCpu_HarrisMergeSortAndPick_XY_HVC
	(
		vx_uint32         capacityOfDstCorner,
		vx_keypoint_t     dstCorner[],
		vx_uint32       * pDstCornerCount,
		vx_uint32         srcWidth,
		vx_uint32         srcHeight,
		vx_float32      * pSrcVc,
		vx_uint32         srcVcStrideInBytes,
		vx_float32        min_distance
	);
int HafCpu_FastCornerMerge_XY_XY
	(
		vx_uint32       capacityOfDstCorner,
		vx_keypoint_t   dstCorner[],
		vx_uint32     * pDstCornerCount,
		vx_uint32		numSrcCornerBuffers,
		vx_keypoint_t * pSrcCorners[],
		vx_uint32       numSrcCorners[]
	);
int HafCpu_CannyEdgeTrace_U8_U8
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[]
	);
int HafCpu_CannyEdgeTrace_U8_U8XY
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32              xyStackTop
	);
int HafCpu_IntegralImage_U32_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint32   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Histogram_DATA_U8
	(
		vx_uint32     dstHist[],
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_HistogramFixedBins_DATA_U8
	(
		vx_uint32     dstHist[],
		vx_uint32     distBinCount,
		vx_uint32     distOffset,
		vx_uint32     distRange,
		vx_uint32     distWindow,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_MeanStdDev_DATA_U8
	(
		vx_float32  * pSum,
		vx_float32  * pSumOfSquared,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_MeanStdDev_DATA_U1
	(
		vx_float32  * pSum,
		vx_float32  * pSumOfSquared,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_Equalize_DATA_DATA
	(
		vx_uint8    * pLut,
		vx_uint32     numPartitions,
		vx_uint32   * pPartSrcHist[]
	);
int HafCpu_HistogramMerge_DATA_DATA
	(
		vx_uint32     dstHist[],
		vx_uint32     numPartitions,
		vx_uint32   * pPartSrcHist[]
	);
int HafCpu_MeanStdDevMerge_DATA_DATA
	(
		vx_float32  * mean,
		vx_float32  * stddev,
		vx_uint32	  totalSampleCount,
		vx_uint32     numPartitions,
		vx_float32    partSum[],
		vx_float32    partSumOfSquared[]
	);
int HafCpu_MinMax_DATA_U8
	(
		vx_int32    * pDstMinValue,
		vx_int32    * pDstMaxValue,
		vx_uint32     srcWidth,
		vx_uint32     srcWeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_MinMax_DATA_S16
	(
		vx_int32    * pDstMinValue,
		vx_int32    * pDstMaxValue,
		vx_uint32     srcWidth,
		vx_uint32     srcWeight,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	);
int HafCpu_MinMaxMerge_DATA_DATA
	(
		vx_int32    * pDstMinValue,
		vx_int32    * pDstMaxValue,
		vx_uint32     numDataPartitions,
		vx_int32      srcMinValue[],
		vx_int32      srcMaxValue[]
	);
int HafCpu_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Min
	(
		vx_uint32          * pMinLocCount,
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Max
	(
		vx_uint32          * pMaxLocCount,
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_U8DATA_Loc_None_Count_MinMax
	(
		vx_uint32          * pMinLocCount,
		vx_uint32          * pMaxLocCount,
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_Min
	(
		vx_uint32          * pMinLocCount,
		vx_uint32            capacityOfMinLocList,
		vx_coordinates2d_t   minLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_MinMax
	(
		vx_uint32          * pMinLocCount,
		vx_uint32          * pMaxLocCount,
		vx_uint32            capacityOfMinLocList,
		vx_coordinates2d_t   minLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_Max
	(
		vx_uint32          * pMaxLocCount,
		vx_uint32            capacityOfMaxLocList,
		vx_coordinates2d_t   maxLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_MinMax
	(
		vx_uint32          * pMinLocCount,
		vx_uint32          * pMaxLocCount,
		vx_uint32            capacityOfMaxLocList,
		vx_coordinates2d_t   maxLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_U8DATA_Loc_MinMax_Count_MinMax
	(
		vx_uint32          * pMinLocCount,
		vx_uint32          * pMaxLocCount,
		vx_uint32            capacityOfMinLocList,
		vx_coordinates2d_t   minLocList[],
		vx_uint32            capacityOfMaxLocList,
		vx_coordinates2d_t   maxLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_uint8           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Min
	(
		vx_uint32          * pMinLocCount,
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_int16           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Max
	(
		vx_uint32          * pMaxLocCount,
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_int16           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_S16DATA_Loc_None_Count_MinMax
	(
		vx_uint32          * pMinLocCount,
		vx_uint32          * pMaxLocCount,
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_int16           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_Min
	(
		vx_uint32          * pMinLocCount,
		vx_uint32            capacityOfMinLocList,
		vx_coordinates2d_t   minLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_int16           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_MinMax
	(
		vx_uint32          * pMinLocCount,
		vx_uint32          * pMaxLocCount,
		vx_uint32            capacityOfMinLocList,
		vx_coordinates2d_t   minLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_int16           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_Max
	(
		vx_uint32          * pMaxLocCount,
		vx_uint32            capacityOfMaxLocList,
		vx_coordinates2d_t   maxLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_int16           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_MinMax
	(
		vx_uint32          * pMinLocCount,
		vx_uint32          * pMaxLocCount,
		vx_uint32            capacityOfMaxLocList,
		vx_coordinates2d_t   maxLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_int16           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLoc_DATA_S16DATA_Loc_MinMax_Count_MinMax
	(
		vx_uint32          * pMinLocCount,
		vx_uint32          * pMaxLocCount,
		vx_uint32            capacityOfMinLocList,
		vx_coordinates2d_t   minLocList[],
		vx_uint32            capacityOfMaxLocList,
		vx_coordinates2d_t   maxLocList[],
		vx_int32           * pDstMinValue,
		vx_int32           * pDstMaxValue,
		vx_uint32            numDataPartitions,
		vx_int32             srcMinValue[],
		vx_int32             srcMaxValue[],
		vx_uint32            srcWidth,
		vx_uint32            srcHeight,
		vx_int16           * pSrcImage,
		vx_uint32            srcImageStrideInBytes
	);
int HafCpu_MinMaxLocMerge_DATA_DATA
	(
		vx_uint32          * pDstLocCount,
		vx_uint32            capacityOfDstLocList,
		vx_coordinates2d_t   dstLocList[],
		vx_uint32            numDataPartitions,
		vx_uint32            partLocCount[],
		vx_coordinates2d_t * partLocList[]
	);

// helper functions for phase
float HafCpu_FastAtan2_deg
(
	vx_int16	  Gx,
	vx_int16      Gy
);

float HafCpu_FastAtan2_rad
(
	vx_int16	  Gx,
	vx_int16      Gy
);

int HafCpu_FastAtan2_Canny
(
	vx_int16	  Gx,
	vx_int16      Gy
);

int HafCpu_WeightedAverage_U8_U8U8
(
    vx_image img1, 
    vx_float32 alpha, 
    vx_image img2, 
    vx_image output
);

int HafCpu_NonLinearFilter_DATA_DATADATA
(
	vx_int32 function, 
	vx_image src, 
	vx_matrix mask, 
	vx_image dst, 
	vx_border_t *border
);

int HafCpu_LaplacianPyramid_DATA_DATA_DATA
(
	vx_node node,
	vx_image input, 
	vx_pyramid laplacian,
	vx_image output
);

int HafCpu_LaplacianReconstruct_DATA_DATA_DATA
(
	vx_node node, 
	vx_pyramid laplacian,
	vx_image input,
	vx_image output
);
#endif // __ago_haf_cpu_h__
