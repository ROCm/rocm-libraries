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

int HafCpu_ColorConvert_IU_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

int HafCpu_ColorConvert_IU_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

int HafCpu_ColorConvert_IV_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

int HafCpu_ColorConvert_IV_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

int HafCpu_ColorConvert_UV12_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImageChroma,
		vx_uint32     dstImageChromaStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

int HafCpu_ColorConvert_UV12_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImageChroma,
		vx_uint32     dstImageChromaStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

int HafCpu_CannyEdgeTrace_U8_U8
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[]
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

