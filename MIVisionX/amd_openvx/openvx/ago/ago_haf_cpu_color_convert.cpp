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

DECL_ALIGN(16) unsigned char dataColorConvert[16 * 26] ATTR_ALIGN(16) = {
	  1,   3,   5,   7,   9,  11,  13,  15, 255, 255, 255, 255, 255, 255, 255, 255,		// UYVY to IYUV - Y; UV12 to IUV - V (lower); NV21 to IYUV - U; UYVY to NV12 - Y; YUYV to NV12 - UV
	  0,   4,   8,  12, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// UYVY to IYUV - U
	  2,   6,  10,  14, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// UYVY to IYUV - V
	  0,   2,   4,   6,   8,  10,  12,  14, 255, 255, 255, 255, 255, 255, 255, 255,		// YUYV to IYUV - Y; UV12 to IUV - U (lower); NV21 to IYUV - V; UYVY to NV12 - UV; YUYV to NV12 - Y
	  1,   5,   9,  13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// YUYV to IYUV - U
	  3,   7,  11,  15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// YUYV to IYUV - V
	  0,   0,   2,   2,   4,   4,   6,   6,   8,   8,  10,  10,  12,  12,  14,  14,		// UV12 to UV - U; NV21 to YUV4 - V
	  1,   1,   3,   3,   5,   5,   7,   7,   9,   9,  11,  11,  13,  13,  15,  15,		// VV12 to UV - V; NV21 to YUV4 - U
	  0,   1,   2,   4,   5,   6,   8,   9,  10,  12,  13,  14, 255, 255, 255, 255,		// RGBX to RGB - First 16 bytes of RGBX to first 16 bytes of RGB
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   1,   2,   4,		// RGBX to RGB - Second 16 bytes of RGBX to first 16 bytes of RGB
	  5,   6,   8,   9,  10,  12,  13,  14, 255, 255, 255, 255, 255, 255, 255, 255,		// RGBX to RGB - Second 16 bytes of RGBX to second 16 bytes of RGB
	255, 255, 255, 255, 255, 255, 255, 255,   0,   1,   2,   4,   5,   6,   8,   9,		// RGBX to RGB - Third 16 bytes of RGBX to second 16 bytes of RGB
	 10,  12,  13,  14, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// RGBX to RGB - Third 16 bytes of RGBX to third 16 bytes of RGB
	255, 255, 255, 255,   0,   1,   2,   4,   5,   6,   8,   9,  10,  12,  13,  14,		// RGBX to RGB - Fourth 16 bytes of RGBX to third 16 bytes of RGB
	  0,   1,   2, 255,   3,   4,   5, 255,   6,   7,   8, 255,   9,  10,  11, 255,		// RGB to RGBX - First 16 bytes of RGB to first 16 bytes of RGBX
	 12,  13,  14, 255,  15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// RGB to RGBX - First 16 bytes of RGB to second 16 bytes of RGBX
	255, 255, 255, 255, 255,   0,   1, 255,   2,   3,   4, 255,   5,   6,   7, 255,		// RGB to RGBX - Second 16 bytes of RGB to second 16 bytes of RGBX
	  8,   9,  10, 255,  11,  12,  13, 255,  14,  15, 255, 255, 255, 255, 255, 255,		// RGB to RGBX - Second 16 bytes of RGB to third 16 bytes of RGBX
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 255,   1,   2,   3, 255,		// RGB to RGBX - Third 16 bytes of RGB to third 16 bytes of RGBX
	  4,   5,   6, 255,   7,   8,   9, 255,  10,  11,  12, 255,  13,  14,  15, 255,		// RGB to RGBX - Third 16 bytes of RGB to fourth 16 bytes of RGBX
	  0,   0,   0, 255,   0,   0,   0, 255,   0,   0,   0, 255,   0,   0,   0, 255,		// RGB to RGBX - Mask to fill in 255 for X positions
	  0,   3,   6,   9,  12,  15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// RGB to single plane extraction
	  2,   5,   8,  11,  14, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// RGB to single plane extraction
	  1,   4,   7,  10,  13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 	// RGB to single plane extraction
	255, 255, 255, 255, 255, 255, 255, 255,   1,   3,   5,   7,   9,  11,  13,  15,		// UV12 to IUV - V (upper)
	255, 255, 255, 255, 255, 255, 255, 255,   0,   2,   4,   6,   8,  10,  12,  14 		// UV12 to IUV - U (upper)
};

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
	)
{
	unsigned char *pLocalSrc, *pLocalDstY, *pLocalDstU, *pLocalDstV;
	unsigned char *pLocalSrcNextRow, *pLocalDstYNextRow;

	__m128i * tbl = (__m128i*) dataColorConvert;
	__m128i maskY = _mm_load_si128(tbl);
	__m128i maskU = _mm_load_si128(tbl + 1);
	__m128i maskV = _mm_load_si128(tbl + 2);
	__m128i pixels0, pixels1, pixels0_NextRow, pixels1_NextRow, temp0, temp1;

	bool isAligned = (((intptr_t(pDstYImage) & intptr_t(pDstUImage) & intptr_t(pDstVImage)) & 7) == ((intptr_t(pDstYImage) | intptr_t(pDstUImage) | intptr_t(pDstVImage)) & 7));		// Check for 8 byte alignment
	isAligned = isAligned & ((intptr_t(pDstYImage) & 8) == 0);					// Y image should be 16 byte aligned or have same alignment as the Chroma planes

	if (isAligned)
	{
		int prefixWidth = intptr_t(pDstYImage) & 15;
		prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
		int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time
		int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalSrcNextRow = (unsigned char *)pSrcImage + srcImageStrideInBytes;
			pLocalDstY = (unsigned char *)pDstYImage;
			pLocalDstYNextRow = (unsigned char *)pDstYImage + dstYImageStrideInBytes;
			pLocalDstU = (unsigned char *)pDstUImage;
			pLocalDstV = (unsigned char *)pDstVImage;

			for (int x = 0; x < prefixWidth; x+=2)
			{
				*pLocalDstU++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// U
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstV++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// V
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
			}

			int width = alignedWidth >> 4;												// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels0_NextRow = _mm_loadu_si128((__m128i *) pLocalSrcNextRow);
				pixels1_NextRow = _mm_loadu_si128((__m128i *) (pLocalSrcNextRow + 16));

				temp0 = _mm_shuffle_epi8(pixels0, maskY);								// Y plane, bytes 0..7
				temp1 = _mm_shuffle_epi8(pixels1, maskY);								// Y plane, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_store_si128((__m128i *) pLocalDstY, temp0);

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskY);						// Y plane - next row, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_shuffle_epi8(pixels0_NextRow, maskY);						// Y plane - next row, bytes 0..7
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_store_si128((__m128i *) pLocalDstYNextRow, temp0);

				temp1 = _mm_shuffle_epi8(pixels1, maskU);								// U plane, intermideate bytes 4..7
				pixels1 = _mm_shuffle_epi8(pixels1, maskV);								// V plane, intermideate bytes 4..7
				temp1 = _mm_slli_si128(temp1, 4);
				pixels1 = _mm_slli_si128(pixels1, 4);

				temp0 = _mm_shuffle_epi8(pixels0, maskU);								// U plane, intermideate bytes 0..3
				pixels0 = _mm_shuffle_epi8(pixels0, maskV);								// V plane, intermideate bytes 0..3
				temp0 = _mm_or_si128(temp0, temp1);										// U plane, intermideate bytes 0..7
				pixels0 = _mm_or_si128(pixels0, pixels1);								// V plane, intermideate bytes 0..7
				
				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskU);						// U plane - next row, intermideate bytes 4..7
				pixels1_NextRow = _mm_shuffle_epi8(pixels1_NextRow, maskV);				// V plane - next row, intermideate bytes 4..7
				temp1 = _mm_slli_si128(temp1, 4);
				pixels1_NextRow = _mm_slli_si128(pixels1_NextRow, 4);

				pixels1 = _mm_shuffle_epi8(pixels0_NextRow, maskU);						// U plane - next row, intermideate bytes 0..3
				pixels0_NextRow = _mm_shuffle_epi8(pixels0_NextRow, maskV);				// V plane - next row, intermideate bytes 0..3
				temp1 = _mm_or_si128(temp1, pixels1);									// U plane - next row, intermideate bytes 0..7
				pixels0_NextRow = _mm_or_si128(pixels0_NextRow, pixels1_NextRow);		// V plane - next row, intermideate bytes 0..7

				__m128i avg0, avg1;
				avg0 = _mm_avg_epu8(temp0, temp1);										// U plane, bytes 0..7
				// to fix it up by subtracting 1 if it got rounded up
				avg0 = _mm_sub_epi8(avg0, _mm_and_si128(_mm_xor_si128(temp0, temp1), _mm_set1_epi8(1)));
				*((int64_t *)pLocalDstU) = M128I(avg0).m128i_i64[0];
				avg1 = _mm_avg_epu8(pixels0, pixels0_NextRow);							// V plane, bytes 0..7
				// to fix it up by subtracting 1 if it got rounded up
				avg1 = _mm_sub_epi8(avg1, _mm_and_si128(_mm_xor_si128(pixels0, pixels0_NextRow), _mm_set1_epi8(1)));
				*((int64_t *)pLocalDstV) = M128I(avg1).m128i_i64[0];

				pLocalSrc += 32;
				pLocalSrcNextRow += 32;
				pLocalDstY += 16;
				pLocalDstYNextRow += 16;
				pLocalDstU += 8;
				pLocalDstV += 8;
				width--;
			}

			for (int x = 0; x < postfixWidth; x+=2)
			{
				*pLocalDstU++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// U
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstV++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// V
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
			}

			pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);				// Advance by 2 rows
			pDstYImage += (dstYImageStrideInBytes + dstYImageStrideInBytes);			// Advance by 2 rows
			pDstUImage += dstUImageStrideInBytes;
			pDstVImage += dstVImageStrideInBytes;

			height -= 2;
		}
	}
	else
	{
		int postfixWidth = (int)dstWidth & 15;
		int alignedWidth = (int)dstWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalSrcNextRow = (unsigned char *)pSrcImage + srcImageStrideInBytes;
			pLocalDstY = (unsigned char *)pDstYImage;
			pLocalDstYNextRow = (unsigned char *)pDstYImage + dstYImageStrideInBytes;
			pLocalDstU = (unsigned char *)pDstUImage;
			pLocalDstV = (unsigned char *)pDstVImage;

			int width = alignedWidth >> 4;												// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels0_NextRow = _mm_loadu_si128((__m128i *) pLocalSrcNextRow);
				pixels1_NextRow = _mm_loadu_si128((__m128i *) (pLocalSrcNextRow + 16));

				temp0 = _mm_shuffle_epi8(pixels0, maskY);								// Y plane, bytes 0..7
				temp1 = _mm_shuffle_epi8(pixels1, maskY);								// Y plane, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_storeu_si128((__m128i *) pLocalDstY, temp0);

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskY);						// Y plane - next row, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_shuffle_epi8(pixels0_NextRow, maskY);						// Y plane - next row, bytes 0..7
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_storeu_si128((__m128i *) pLocalDstYNextRow, temp0);

				temp1 = _mm_shuffle_epi8(pixels1, maskU);								// U plane, intermideate bytes 4..7
				pixels1 = _mm_shuffle_epi8(pixels1, maskV);								// V plane, intermideate bytes 4..7
				temp1 = _mm_slli_si128(temp1, 4);
				pixels1 = _mm_slli_si128(pixels1, 4);

				temp0 = _mm_shuffle_epi8(pixels0, maskU);								// U plane, intermideate bytes 0..3
				pixels0 = _mm_shuffle_epi8(pixels0, maskV);								// V plane, intermideate bytes 0..3
				temp0 = _mm_or_si128(temp0, temp1);										// U plane, intermideate bytes 0..7
				pixels0 = _mm_or_si128(pixels0, pixels1);								// V plane, intermideate bytes 0..7

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskU);						// U plane - next row, intermideate bytes 4..7
				pixels1_NextRow = _mm_shuffle_epi8(pixels1_NextRow, maskV);				// V plane - next row, intermideate bytes 4..7
				temp1 = _mm_slli_si128(temp1, 4);
				pixels1_NextRow = _mm_slli_si128(pixels1_NextRow, 4);

				pixels1 = _mm_shuffle_epi8(pixels0_NextRow, maskU);						// U plane - next row, intermideate bytes 0..3
				pixels0_NextRow = _mm_shuffle_epi8(pixels0_NextRow, maskV);				// V plane - next row, intermideate bytes 0..3
				temp1 = _mm_or_si128(temp1, pixels1);									// U plane - next row, intermideate bytes 0..7
				pixels0_NextRow = _mm_or_si128(pixels0_NextRow, pixels1_NextRow);		// V plane - next row, intermideate bytes 0..7

				__m128i avg0, avg1;
				avg0 = _mm_avg_epu8(temp0, temp1);										// U plane, bytes 0..7
				// to fix it up by subtracting 1 if it got rounded up
				avg0 = _mm_sub_epi8(avg0, _mm_and_si128(_mm_xor_si128(temp0, temp1), _mm_set1_epi8(1)));
				*((int64_t *)pLocalDstU) = M128I(avg0).m128i_i64[0];
				avg1 = _mm_avg_epu8(pixels0, pixels0_NextRow);							// V plane, bytes 0..7
				// to fix it up by subtracting 1 if it got rounded up
				avg1 = _mm_sub_epi8(avg1, _mm_and_si128(_mm_xor_si128(pixels0, pixels0_NextRow), _mm_set1_epi8(1)));
				*((int64_t *)pLocalDstV) = M128I(avg1).m128i_i64[0];

				pLocalSrc += 32;
				pLocalSrcNextRow += 32;
				pLocalDstY += 16;
				pLocalDstYNextRow += 16;
				pLocalDstU += 8;
				pLocalDstV += 8;
				width--;
			}

			for (int x = 0; x < postfixWidth; x+=2)
			{
				*pLocalDstU++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// U
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstV++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// V
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
			}

			pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);				// Advance by 2 rows
			pDstYImage += (dstYImageStrideInBytes + dstYImageStrideInBytes);			// Advance by 2 rows
			pDstUImage += dstUImageStrideInBytes;
			pDstVImage += dstVImageStrideInBytes;

			height -= 2;
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_RGB_UYVY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~7;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i shufMask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 5, 4, 2, 1, 0);
	__m128i tempI, row;
	__m128 Y0, Y1, U, V;

	// BT709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care
	__m128 const128 = _mm_set1_ps(128.0f);

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < alignedWidth; width += 8)
		{
			row = _mm_loadu_si128((__m128i *)pLocalSrc);

			for (int i = 0; i < 4; i++)
			{
				tempI = _mm_shuffle_epi8(row, _mm_set1_epi32((int)0xFFFFFF00));
				U = _mm_cvtepi32_ps(tempI);								// U U U U
				U = _mm_sub_ps(U, const128);
				row = _mm_srli_si128(row, 1);
				tempI = _mm_shuffle_epi8(row, _mm_set1_epi32((int)0xFFFFFF00));
				Y0 = _mm_cvtepi32_ps(tempI);							// Y0 Y0 Y0 Y0
				row = _mm_srli_si128(row, 1);
				tempI = _mm_shuffle_epi8(row, _mm_set1_epi32((int)0xFFFFFF00));
				V = _mm_cvtepi32_ps(tempI);								// V V V V
				V = _mm_sub_ps(V, const128);
				row = _mm_srli_si128(row, 1);
				tempI = _mm_shuffle_epi8(row, _mm_set1_epi32((int)0xFFFFFF00));
				Y1 = _mm_cvtepi32_ps(tempI);							// Y1 Y1 Y1 Y1
				row = _mm_srli_si128(row, 1);

				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 0
				Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 1

				// Convert RGB01 to U8
				tempI = _mm_packus_epi32(_mm_cvttps_epi32(Y0), _mm_cvttps_epi32(Y1));
				tempI = _mm_packus_epi16(tempI, tempI);
				tempI = _mm_shuffle_epi8(tempI, shufMask);
				_mm_storeu_si128((__m128i *)(pLocalDst + 6 * i), tempI);
			}

			pLocalSrc += 16;
			pLocalDst += 24;
		}

		for (int width = 0; width < postfixWidth; width += 2)
		{
			float Ypix1, Ypix2, Upix, Vpix, Rpix, Gpix, Bpix;
			Upix  = (float)(*pLocalSrc++) - 128.0f;
			Ypix1 = (float)(*pLocalSrc++);
			Vpix  = (float)(*pLocalSrc++) - 128.0f;
			Ypix2 = (float)(*pLocalSrc++);

			Rpix = fminf(fmaxf(Ypix1 + (Vpix * 1.5748f), 0.0f), 255.0f);
			Gpix = fminf(fmaxf(Ypix1 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
			Bpix = fminf(fmaxf(Ypix1 + (Upix * 1.8556f), 0.0f), 255.0f);
			
			*pLocalDst++ = (vx_uint8)Rpix;
			*pLocalDst++ = (vx_uint8)Gpix;
			*pLocalDst++ = (vx_uint8)Bpix;
			
			Rpix = fminf(fmaxf(Ypix2 + (Vpix * 1.5748f), 0.0f), 255.0f);
			Gpix = fminf(fmaxf(Ypix2 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
			Bpix = fminf(fmaxf(Ypix2 + (Upix * 1.8556f), 0.0f), 255.0f);

			*pLocalDst++ = (vx_uint8)Rpix;
			*pLocalDst++ = (vx_uint8)Gpix;
			*pLocalDst++ = (vx_uint8)Bpix;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_RGB_YUYV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~7;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i shufMask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 5, 4, 2, 1, 0);
	__m128i tempI, row;
	__m128 Y0, Y1, U, V;

	// BT709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care
	__m128 const128 = _mm_set1_ps(128.0f);

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < alignedWidth; width += 8)
		{
			row = _mm_loadu_si128((__m128i *)pLocalSrc);

			for (int i = 0; i < 4; i++)
			{
				tempI = _mm_shuffle_epi8(row, _mm_set1_epi32((int)0xFFFFFF00));
				Y0 = _mm_cvtepi32_ps(tempI);							// Y0 Y0 Y0 Y0
				row = _mm_srli_si128(row, 1);
				tempI = _mm_shuffle_epi8(row, _mm_set1_epi32((int)0xFFFFFF00));
				U = _mm_cvtepi32_ps(tempI);								// U U U U
				U = _mm_sub_ps(U, const128);
				row = _mm_srli_si128(row, 1);
				tempI = _mm_shuffle_epi8(row, _mm_set1_epi32((int)0xFFFFFF00));
				Y1 = _mm_cvtepi32_ps(tempI);							// Y1 Y1 Y1 Y1
				row = _mm_srli_si128(row, 1);
				tempI = _mm_shuffle_epi8(row, _mm_set1_epi32((int)0xFFFFFF00));
				V = _mm_cvtepi32_ps(tempI);								// V V V V
				V = _mm_sub_ps(V, const128);
				row = _mm_srli_si128(row, 1);

				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 0
				Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 1

				// Convert RGB01 to U8
				tempI = _mm_packus_epi32(_mm_cvttps_epi32(Y0), _mm_cvttps_epi32(Y1));
				tempI = _mm_packus_epi16(tempI, tempI);
				tempI = _mm_shuffle_epi8(tempI, shufMask);
				_mm_storeu_si128((__m128i *)(pLocalDst + 6 * i), tempI);
			}

			pLocalSrc += 16;
			pLocalDst += 24;
		}
		
		for (int width = 0; width < postfixWidth; width += 2)
		{
			float Ypix1, Ypix2, Upix, Vpix, Rpix, Gpix, Bpix;
			Ypix1 = (float)(*pLocalSrc++);
			Upix = (float)(*pLocalSrc++) - 128.0f;
			Ypix2 = (float)(*pLocalSrc++);
			Vpix = (float)(*pLocalSrc++) - 128.0f;

			Rpix = fminf(fmaxf(Ypix1 + (Vpix * 1.5748f), 0.0f), 255.0f);
			Gpix = fminf(fmaxf(Ypix1 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
			Bpix = fminf(fmaxf(Ypix1 + (Upix * 1.8556f), 0.0f), 255.0f);

			*pLocalDst++ = (vx_uint8)Rpix;
			*pLocalDst++ = (vx_uint8)Gpix;
			*pLocalDst++ = (vx_uint8)Bpix;

			Rpix = fminf(fmaxf(Ypix2 + (Vpix * 1.5748f), 0.0f), 255.0f);
			Gpix = fminf(fmaxf(Ypix2 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
			Bpix = fminf(fmaxf(Ypix2 + (Upix * 1.8556f), 0.0f), 255.0f);

			*pLocalDst++ = (vx_uint8)Rpix;
			*pLocalDst++ = (vx_uint8)Gpix;
			*pLocalDst++ = (vx_uint8)Bpix;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	alignedWidth = (alignedWidth > 15) ? alignedWidth-16: 0;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128 Y00, Y01, Y10, Y11, U, V;
	__m128i Y0pix, Y1pix;
	__m128i shufMask = _mm_set_epi8(-1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);

	// BT 709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care

	vx_uint8 Upixels[8], Vpixels[8];

	for (int height = 0; height < (int)dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrcY = pSrcYImage;
		vx_uint8 * pLocalSrcU = pSrcUImage;
		vx_uint8 * pLocalSrcV = pSrcVImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)	// Process 16 pixels at a time
		{
			Y0pix = _mm_loadu_si128((__m128i *) pLocalSrcY);
			Y1pix = _mm_loadu_si128((__m128i *) (pLocalSrcY + srcYImageStrideInBytes));
			*((int64_t *)Upixels) = *((int64_t *)pLocalSrcU);
			*((int64_t *)Vpixels) = *((int64_t *)pLocalSrcV);

			for (int i = 0; i < 4; i++)
			{
				// For pixels 00, 01
				//			  10, 11
				U = _mm_cvtepi32_ps(_mm_set1_epi32((int)Upixels[2*i]));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				V = _mm_cvtepi32_ps(_mm_set1_epi32((int)Vpixels[2*i]));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);

				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 00
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 01
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 10
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 11

				__m128i tempI0 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB00, RGB01 to U8
				__m128i tempI1 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB10, RGB11 to U8

				// For pixels 02, 03
				//			  12, 13
				U = _mm_cvtepi32_ps(_mm_set1_epi32((int)Upixels[2*i + 1]));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				V = _mm_cvtepi32_ps(_mm_set1_epi32((int)Vpixels[2*i + 1]));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);

				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 02
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 03
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 12
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 13

				__m128i tempI2 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB02, RGB03 to U8
				tempI0 = _mm_packus_epi16(tempI0, tempI2);
				tempI0 = _mm_shuffle_epi8(tempI0, shufMask);
				_mm_storeu_si128((__m128i *)pLocalDst, tempI0);

				__m128i tempI3 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB12, RGB13 to U8
				tempI1 = _mm_packus_epi16(tempI1, tempI3);
				tempI1 = _mm_shuffle_epi8(tempI1, shufMask);
				_mm_storeu_si128((__m128i *)(pLocalDst + dstImageStrideInBytes), tempI1);
				pLocalDst += 12;
			}

			pLocalSrcY += 16;
			pLocalSrcU += 8;
			pLocalSrcV += 8;
		}

		for (int width = 0; width < (postfixWidth >> 1); width++)		// Processing two pixels at a time in a row
		{
			float Ypix, Rpix, Gpix, Bpix;

			Ypix = (float)(*pLocalSrcY);
			Rpix = (float)(*pLocalSrcV++) - 128.0f;
			Bpix = (float)(*pLocalSrcU++) - 128.0f;

			Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
			Rpix *= 1.5748f;
			Bpix *= 1.8556f;

			*pLocalDst = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcY + 1));
			*(pLocalDst + 3) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 4) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 5) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcY + srcYImageStrideInBytes));
			*(pLocalDst + dstImageStrideInBytes + 0) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcY + srcYImageStrideInBytes + 1));
			*(pLocalDst + dstImageStrideInBytes + 3) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 4) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 5) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			pLocalSrcY += 2;
			pLocalDst += 6;
		}
		pSrcYImage += (srcYImageStrideInBytes + srcYImageStrideInBytes);
		pSrcUImage += srcUImageStrideInBytes;
		pSrcVImage += srcVImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes + dstImageStrideInBytes);
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	alignedWidth = (alignedWidth > 15) ? alignedWidth-16: 0;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128 Y00, Y01, Y10, Y11, U, V;
	__m128i Y0pix, Y1pix, UVpix;
	__m128i shufMask = _mm_set_epi8(-1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);

	// BT 709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care

	for (int height = 0; height < (int)dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrcLuma = pSrcLumaImage;
		vx_uint8 * pLocalSrcChroma = pSrcChromaImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)	// Process 16 pixels at a time
		{
			Y0pix = _mm_loadu_si128((__m128i *) pLocalSrcLuma);
			Y1pix = _mm_loadu_si128((__m128i *) (pLocalSrcLuma + srcLumaImageStrideInBytes));
			UVpix = _mm_loadu_si128((__m128i *) pLocalSrcChroma);

			for (int i = 0; i < 4; i++)
			{
				// For pixels 00, 01
				//			  10, 11
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				U = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				V = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 00
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 01
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 10
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 11

				__m128i tempI0 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB00, RGB01 to U8
				__m128i tempI1 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB10, RGB11 to U8

				// For pixels 02, 03
				//			  12, 13
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				U = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				V = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 02
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 03
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 12
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 13

				__m128i tempI2 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB02, RGB03 to U8
				tempI0 = _mm_packus_epi16(tempI0, tempI2);
				tempI0 = _mm_shuffle_epi8(tempI0, shufMask);
				_mm_storeu_si128((__m128i *)pLocalDst, tempI0);

				__m128i tempI3 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB12, RGB13 to U8
				tempI1 = _mm_packus_epi16(tempI1, tempI3);
				tempI1 = _mm_shuffle_epi8(tempI1, shufMask);
				_mm_storeu_si128((__m128i *)(pLocalDst + dstImageStrideInBytes), tempI1);
				pLocalDst += 12;
			}
			pLocalSrcLuma += 16;
			pLocalSrcChroma += 16;
		}

		for (int width = 0; width < (postfixWidth >> 1); width++)		// Processing two pixels at a time in a row
		{
			float Ypix, Rpix, Gpix, Bpix;

			Ypix = (float)(*pLocalSrcLuma);
			Bpix = (float)(*pLocalSrcChroma++) - 128.0f;
			Rpix = (float)(*pLocalSrcChroma++) - 128.0f;

			Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
			Rpix *= 1.5748f;
			Bpix *= 1.8556f;

			*pLocalDst = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcLuma + 1));
			*(pLocalDst + 3) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 4) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 5) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcLuma + srcLumaImageStrideInBytes));
			*(pLocalDst + dstImageStrideInBytes + 0) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcLuma + srcLumaImageStrideInBytes + 1));
			*(pLocalDst + dstImageStrideInBytes + 3) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 4) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 5) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			pLocalSrcLuma += 2;
			pLocalDst += 6;
		}
		pSrcLumaImage += (srcLumaImageStrideInBytes + srcLumaImageStrideInBytes);
		pSrcChromaImage += srcChromaImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes + dstImageStrideInBytes);
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	alignedWidth = (alignedWidth > 15) ? alignedWidth-16: 0;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128 Y00, Y01, Y10, Y11, U, V;
	__m128i Y0pix, Y1pix, UVpix;
	__m128i shufMask = _mm_set_epi8(-1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);

	// BT 709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care

	for (int height = 0; height < (int)dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrcLuma = pSrcLumaImage;
		vx_uint8 * pLocalSrcChroma = pSrcChromaImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)	// Process 16 pixels at a time
		{
			Y0pix = _mm_loadu_si128((__m128i *) pLocalSrcLuma);
			Y1pix = _mm_loadu_si128((__m128i *) (pLocalSrcLuma + srcLumaImageStrideInBytes));
			UVpix = _mm_loadu_si128((__m128i *) pLocalSrcChroma);

			for (int i = 0; i < 4; i++)
			{
				// For pixels 00, 01
				//			  10, 11
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				V = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 00
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 01
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 10
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 11

				__m128i tempI0 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB00, RGB01 to U8
				__m128i tempI1 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB10, RGB11 to U8

				// For pixels 02, 03
				//			  12, 13
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				V = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 02
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 03
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 12
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 13

				__m128i tempI2 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB02, RGB03 to U8
				tempI0 = _mm_packus_epi16(tempI0, tempI2);
				tempI0 = _mm_shuffle_epi8(tempI0, shufMask);
				_mm_storeu_si128((__m128i *)pLocalDst, tempI0);

				__m128i tempI3 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB12, RGB13 to U8
				tempI1 = _mm_packus_epi16(tempI1, tempI3);
				tempI1 = _mm_shuffle_epi8(tempI1, shufMask);
				_mm_storeu_si128((__m128i *)(pLocalDst + dstImageStrideInBytes), tempI1);
				pLocalDst += 12;
			}
			pLocalSrcLuma += 16;
			pLocalSrcChroma += 16;
		}

		for (int width = 0; width < (postfixWidth >> 1); width++)		// Processing two pixels at a time in a row
		{
			float Ypix, Rpix, Gpix, Bpix;

			Ypix = (float)(*pLocalSrcLuma);
			Rpix = (float)(*pLocalSrcChroma++) - 128.0f;
			Bpix = (float)(*pLocalSrcChroma++) - 128.0f;

			Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
			Rpix *= 1.5748f;
			Bpix *= 1.8556f;

			*pLocalDst = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcLuma + 1));
			*(pLocalDst + 3) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 4) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 5) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcLuma + srcLumaImageStrideInBytes));
			*(pLocalDst + dstImageStrideInBytes + 0) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			Ypix = (float)(*(pLocalSrcLuma + srcLumaImageStrideInBytes + 1));
			*(pLocalDst + dstImageStrideInBytes + 3) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 4) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 5) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);

			pLocalSrcLuma += 2;
			pLocalDst += 6;
		}
		pSrcLumaImage += (srcLumaImageStrideInBytes + srcLumaImageStrideInBytes);
		pSrcChromaImage += srcChromaImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes + dstImageStrideInBytes);
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_RGBX_UYVY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~7;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i mask = _mm_set1_epi32((int)0xFFFFFF00);						// 255 255 255 0 255 255 255 0 255 255 255 0 255 255 255 0
	__m128i tempI, row, RGB0, RGB1;
	__m128 Y0, Y1, U, V;

	// BT 709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care
	__m128 const128 = _mm_set1_ps(128.0f);

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < alignedWidth; width += 8)
		{
			row = _mm_load_si128((__m128i *)pLocalSrc);

			// Pixels 0,1
			tempI = _mm_shuffle_epi8(row, mask);
			U = _mm_cvtepi32_ps(tempI);								// U0 U0 U0 U0
			U = _mm_sub_ps(U, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y0 = _mm_cvtepi32_ps(tempI);							// Y0 Y0 Y0 Y0
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			V = _mm_cvtepi32_ps(tempI);								// V0 V0 V0 V0
			V = _mm_sub_ps(V, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y1 = _mm_cvtepi32_ps(tempI);							// Y1 Y1 Y1 Y1
			row = _mm_srli_si128(row, 1);

			U = _mm_mul_ps(U, weights_U2RGB);
			V = _mm_mul_ps(V, weights_V2RGB);
			U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
			Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 0
			Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 1

			tempI = _mm_cvttps_epi32(Y0);
			RGB0 = _mm_cvttps_epi32(Y1);
			RGB0 = _mm_packus_epi32(tempI, RGB0);					// X1 B1 G1 R1 X0 B0 G0 R0 (words)

			// Pixels 2,3
			tempI = _mm_shuffle_epi8(row, mask);
			U = _mm_cvtepi32_ps(tempI);								// U1 U1 U1 U1
			U = _mm_sub_ps(U, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y0 = _mm_cvtepi32_ps(tempI);							// Y2 Y2 Y2 Y2
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			V = _mm_cvtepi32_ps(tempI);								// V1 V1 V1 V1
			V = _mm_sub_ps(V, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y1 = _mm_cvtepi32_ps(tempI);							// Y3 Y3 Y3 Y3
			row = _mm_srli_si128(row, 1);

			U = _mm_mul_ps(U, weights_U2RGB);
			V = _mm_mul_ps(V, weights_V2RGB);
			U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
			Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 2
			Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 3

			tempI = _mm_cvttps_epi32(Y0);
			RGB1 = _mm_cvttps_epi32(Y1);
			RGB1 = _mm_packus_epi32(tempI, RGB1);					// X3 B3 G3 R3 X2 B2 G2 R2 (words)
			RGB0 = _mm_packus_epi16(RGB0, RGB1);					// X3 B3 G3 R3 X2 B2 G2 R2 X1 B1 G1 R1 X0 B0 G0 R0 (bytes)

			// Pixels 4,5
			tempI = _mm_shuffle_epi8(row, mask);
			U = _mm_cvtepi32_ps(tempI);								// U2 U2 U2 U2
			U = _mm_sub_ps(U, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y0 = _mm_cvtepi32_ps(tempI);							// Y4 Y4 Y4 Y4
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			V = _mm_cvtepi32_ps(tempI);								// V2 V2 V2 V2
			V = _mm_sub_ps(V, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y1 = _mm_cvtepi32_ps(tempI);							// Y5 Y5 Y5 Y5
			row = _mm_srli_si128(row, 1);

			U = _mm_mul_ps(U, weights_U2RGB);
			V = _mm_mul_ps(V, weights_V2RGB);
			U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
			Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 4
			Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 5

			tempI = _mm_cvttps_epi32(Y0);
			RGB1 = _mm_cvttps_epi32(Y1);
			RGB1 = _mm_packus_epi32(tempI, RGB1);					// X5 B5 G5 R5 X4 B4 G4 R4 (words)

			// Pixels 6,7
			tempI = _mm_shuffle_epi8(row, mask);
			U = _mm_cvtepi32_ps(tempI);								// U3 U3 U3 U3
			U = _mm_sub_ps(U, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y0 = _mm_cvtepi32_ps(tempI);							// Y6 Y6 Y6 Y6
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			V = _mm_cvtepi32_ps(tempI);								// V3 V3 V3 V3
			V = _mm_sub_ps(V, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y1 = _mm_cvtepi32_ps(tempI);							// Y7 Y7 Y7 Y7

			U = _mm_mul_ps(U, weights_U2RGB);
			V = _mm_mul_ps(V, weights_V2RGB);
			U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
			Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 6
			Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 7

			tempI = _mm_cvttps_epi32(Y0);
			row = _mm_cvttps_epi32(Y1);
			row = _mm_packus_epi32(tempI, row);						// X7 B7 G7 R7 X6 B6 G6 R6 (words)
			RGB1 = _mm_packus_epi16(RGB1, row);						// X7 B7 G7 R7 X6 B6 G6 R6 X5 B5 G5 R5 X4 B4 G4 R4 (bytes)

			// Make the X component value 255
			RGB0 = _mm_or_si128(RGB0, _mm_set1_epi32((int)0xFF000000));
			RGB1 = _mm_or_si128(RGB1, _mm_set1_epi32((int)0xFF000000));

			_mm_storeu_si128((__m128i *)pLocalDst, RGB0);
			_mm_storeu_si128((__m128i *)(pLocalDst + 16), RGB1);

			pLocalSrc += 16;
			pLocalDst += 32;
		}

		for (int width = 0; width < postfixWidth; width += 2)
		{
			float Ypix1, Ypix2, Upix, Vpix, Rpix, Gpix, Bpix;
			Upix = (float)(*pLocalSrc++) - 128.0f;
			Ypix1 = (float)(*pLocalSrc++);
			Vpix = (float)(*pLocalSrc++) - 128.0f;
			Ypix2 = (float)(*pLocalSrc++);

			Rpix = fminf(fmaxf(Ypix1 + (Vpix * 1.5748f), 0.0f), 255.0f);
			Gpix = fminf(fmaxf(Ypix1 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
			Bpix = fminf(fmaxf(Ypix1 + (Upix * 1.8556f), 0.0f), 255.0f);

			*pLocalDst++ = (vx_uint8)Rpix;
			*pLocalDst++ = (vx_uint8)Gpix;
			*pLocalDst++ = (vx_uint8)Bpix;
			*pLocalDst++ = (vx_uint8)255;

			Rpix = fminf(fmaxf(Ypix2 + (Vpix * 1.5748f), 0.0f), 255.0f);
			Gpix = fminf(fmaxf(Ypix2 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
			Bpix = fminf(fmaxf(Ypix2 + (Upix * 1.8556f), 0.0f), 255.0f);

			*pLocalDst++ = (vx_uint8)Rpix;
			*pLocalDst++ = (vx_uint8)Gpix;
			*pLocalDst++ = (vx_uint8)Bpix;
			*pLocalDst++ = (vx_uint8)255;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_RGBX_YUYV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~7;
	int postfixWidth = (int)dstWidth - alignedWidth;
	
	__m128i mask = _mm_set1_epi32((int)0xFFFFFF00);						// 255 255 255 0 255 255 255 0 255 255 255 0 255 255 255 0
	__m128i tempI, row, RGB0, RGB1;
	__m128 Y0, Y1, U, V;

	// BT 709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care
	__m128 const128 = _mm_set1_ps(128.0f);

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < alignedWidth; width += 8)
		{
			row = _mm_load_si128((__m128i *)pLocalSrc);

			// Pixels 0,1
			tempI = _mm_shuffle_epi8(row, mask);
			Y0 = _mm_cvtepi32_ps(tempI);							// Y0 Y0 Y0 Y0
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			U = _mm_cvtepi32_ps(tempI);								// U0 U0 U0 U0
			U = _mm_sub_ps(U, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y1 = _mm_cvtepi32_ps(tempI);							// Y1 Y1 Y1 Y1
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			V = _mm_cvtepi32_ps(tempI);								// V0 V0 V0 V0
			V = _mm_sub_ps(V, const128);
			row = _mm_srli_si128(row, 1);

			U = _mm_mul_ps(U, weights_U2RGB);
			V = _mm_mul_ps(V, weights_V2RGB);
			U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
			Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 0
			Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 1

			tempI = _mm_cvttps_epi32(Y0);
			RGB0 = _mm_cvttps_epi32(Y1);
			RGB0 = _mm_packus_epi32(tempI, RGB0);					// X1 B1 G1 R1 X0 B0 G0 R0 (words)

			// Pixels 2,3
			tempI = _mm_shuffle_epi8(row, mask);
			Y0 = _mm_cvtepi32_ps(tempI);							// Y2 Y2 Y2 Y2
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			U = _mm_cvtepi32_ps(tempI);								// U1 U1 U1 U1
			U = _mm_sub_ps(U, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y1 = _mm_cvtepi32_ps(tempI);							// Y3 Y3 Y3 Y3
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			V = _mm_cvtepi32_ps(tempI);								// V1 V1 V1 V1
			V = _mm_sub_ps(V, const128);
			row = _mm_srli_si128(row, 1);

			U = _mm_mul_ps(U, weights_U2RGB);
			V = _mm_mul_ps(V, weights_V2RGB);
			U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
			Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 2
			Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 3

			tempI = _mm_cvttps_epi32(Y0);
			RGB1 = _mm_cvttps_epi32(Y1);
			RGB1 = _mm_packus_epi32(tempI, RGB1);					// X3 B3 G3 R3 X2 B2 G2 R2 (words)
			RGB0 = _mm_packus_epi16(RGB0, RGB1);					// X3 B3 G3 R3 X2 B2 G2 R2 X1 B1 G1 R1 X0 B0 G0 R0 (bytes)

			// Pixels 4,5
			tempI = _mm_shuffle_epi8(row, mask);
			Y0 = _mm_cvtepi32_ps(tempI);							// Y4 Y4 Y4 Y4
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			U = _mm_cvtepi32_ps(tempI);								// U2 U2 U2 U2
			U = _mm_sub_ps(U, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y1 = _mm_cvtepi32_ps(tempI);							// Y5 Y5 Y5 Y5
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			V = _mm_cvtepi32_ps(tempI);								// V2 V2 V2 V2
			V = _mm_sub_ps(V, const128);
			row = _mm_srli_si128(row, 1);

			U = _mm_mul_ps(U, weights_U2RGB);
			V = _mm_mul_ps(V, weights_V2RGB);
			U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
			Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 4
			Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 5

			tempI = _mm_cvttps_epi32(Y0);
			RGB1 = _mm_cvttps_epi32(Y1);
			RGB1 = _mm_packus_epi32(tempI, RGB1);					// X5 B5 G5 R5 X4 B4 G4 R4 (words)

			// Pixels 6,7
			tempI = _mm_shuffle_epi8(row, mask);
			Y0 = _mm_cvtepi32_ps(tempI);							// Y6 Y6 Y6 Y6
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			U = _mm_cvtepi32_ps(tempI);								// U3 U3 U3 U3
			U = _mm_sub_ps(U, const128);
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			Y1 = _mm_cvtepi32_ps(tempI);							// Y7 Y7 Y7 Y7
			row = _mm_srli_si128(row, 1);
			tempI = _mm_shuffle_epi8(row, mask);
			V = _mm_cvtepi32_ps(tempI);								// V0 V0 V0 V3
			V = _mm_sub_ps(V, const128);

			U = _mm_mul_ps(U, weights_U2RGB);
			V = _mm_mul_ps(V, weights_V2RGB);
			U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
			Y0 = _mm_add_ps(Y0, U);									// RGB for pixel 6
			Y1 = _mm_add_ps(Y1, U);									// RGB for pixel 7

			tempI = _mm_cvttps_epi32(Y0);
			row = _mm_cvttps_epi32(Y1);
			row = _mm_packus_epi32(tempI, row);						// X7 B7 G7 R7 X6 B6 G6 R6 (words)
			RGB1 = _mm_packus_epi16(RGB1, row);						// X7 B7 G7 R7 X6 B6 G6 R6 X5 B5 G5 R5 X4 B4 G4 R4 (bytes)

			// Make the X component value 255
			RGB0 = _mm_or_si128(RGB0, _mm_set1_epi32((int)0xFF000000));
			RGB1 = _mm_or_si128(RGB1, _mm_set1_epi32((int)0xFF000000));

			_mm_storeu_si128((__m128i *)pLocalDst, RGB0);
			_mm_storeu_si128((__m128i *)(pLocalDst + 16), RGB1);

			pLocalSrc += 16;
			pLocalDst += 32;
		}

		for (int width = 0; width < postfixWidth; width += 2)
		{
			float Ypix1, Ypix2, Upix, Vpix, Rpix, Gpix, Bpix;
			Ypix1 = (float)(*pLocalSrc++);
			Upix = (float)(*pLocalSrc++) - 128.0f;
			Ypix2 = (float)(*pLocalSrc++);
			Vpix = (float)(*pLocalSrc++) - 128.0f;

			Rpix = fminf(fmaxf(Ypix1 + (Vpix * 1.5748f), 0.0f), 255.0f);
			Gpix = fminf(fmaxf(Ypix1 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
			Bpix = fminf(fmaxf(Ypix1 + (Upix * 1.8556f), 0.0f), 255.0f);

			*pLocalDst++ = (vx_uint8)Rpix;
			*pLocalDst++ = (vx_uint8)Gpix;
			*pLocalDst++ = (vx_uint8)Bpix;
			*pLocalDst++ = (vx_uint8)255;

			Rpix = fminf(fmaxf(Ypix2 + (Vpix * 1.5748f), 0.0f), 255.0f);
			Gpix = fminf(fmaxf(Ypix2 - (Upix * 0.1873f) - (Vpix * 0.4681f), 0.0f), 255.0f);
			Bpix = fminf(fmaxf(Ypix2 + (Upix * 1.8556f), 0.0f), 255.0f);

			*pLocalDst++ = (vx_uint8)Rpix;
			*pLocalDst++ = (vx_uint8)Gpix;
			*pLocalDst++ = (vx_uint8)Bpix;
			*pLocalDst++ = (vx_uint8)255;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128 Y00, Y01, Y10, Y11, U, V;
	__m128i Y0pix, Y1pix;
	__m128i shufMask = _mm_set_epi8(-1, 14, 13, 12, -1, 10, 9, 8, -1, 6, 5, 4, -1, 2, 1, 0);

	// BT 709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care

	vx_uint8 Upixels[8], Vpixels[8];

	for (int height = 0; height < (int)dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrcY = pSrcYImage;
		vx_uint8 * pLocalSrcU = pSrcUImage;
		vx_uint8 * pLocalSrcV = pSrcVImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)	// Process 16 pixels at a time
		{
			Y0pix = _mm_loadu_si128((__m128i *) pLocalSrcY);
			Y1pix = _mm_loadu_si128((__m128i *) (pLocalSrcY + srcYImageStrideInBytes));
			*((int64_t *)Upixels) = *((int64_t *)pLocalSrcU);
			*((int64_t *)Vpixels) = *((int64_t *)pLocalSrcV);

			for (int i = 0; i < 4; i++)
			{
				// For pixels 00, 01
				//			  10, 11
				U = _mm_cvtepi32_ps(_mm_set1_epi32((int)Upixels[2 * i]));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				V = _mm_cvtepi32_ps(_mm_set1_epi32((int)Vpixels[2 * i]));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);

				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 00
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 01
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 10
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 11

				__m128i tempI0 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB00, RGB01 to U8
				__m128i tempI1 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB10, RGB11 to U8

				// For pixels 02, 03
				//			  12, 13
				U = _mm_cvtepi32_ps(_mm_set1_epi32((int)Upixels[2 * i + 1]));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				V = _mm_cvtepi32_ps(_mm_set1_epi32((int)Vpixels[2 * i + 1]));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);

				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 02
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 03
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 12
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 13

				__m128i tempI2 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB02, RGB03 to U8
				tempI0 = _mm_packus_epi16(tempI0, tempI2);
				tempI0 = _mm_shuffle_epi8(tempI0, shufMask);
				tempI0 = _mm_or_si128(tempI0, _mm_set1_epi32((int)0xFF000000));
				_mm_storeu_si128((__m128i *)pLocalDst, tempI0);

				__m128i tempI3 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB12, RGB13 to U8
				tempI1 = _mm_packus_epi16(tempI1, tempI3);
				tempI1 = _mm_shuffle_epi8(tempI1, shufMask);
				tempI1 = _mm_or_si128(tempI1, _mm_set1_epi32((int)0xFF000000));
				_mm_storeu_si128((__m128i *)(pLocalDst + dstImageStrideInBytes), tempI1);
				
				pLocalDst += 16;
			}
			pLocalSrcY += 16;
			pLocalSrcU += 8;
			pLocalSrcV += 8;
		}

		for (int width = 0; width < (postfixWidth >> 1); width ++)		// Processing two pixels at a time in a row
		{
			float Ypix, Rpix, Gpix, Bpix;

			Ypix = (float)(*pLocalSrcY);
			Rpix = (float)(*pLocalSrcV++) - 128.0f;
			Bpix = (float)(*pLocalSrcU++) - 128.0f;

			Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
			Rpix *= 1.5748f;
			Bpix *= 1.8556f;

			*pLocalDst = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + 3) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcY + 1));
			*(pLocalDst + 4) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 5) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 6) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + 7) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcY + srcYImageStrideInBytes));
			*(pLocalDst + dstImageStrideInBytes + 0) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 3) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcY + srcYImageStrideInBytes + 1));
			*(pLocalDst + dstImageStrideInBytes + 4) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 5) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 6) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 7) = (vx_uint8)255;

			pLocalSrcY += 2;
			pLocalDst += 8;
		}
		pSrcYImage += (srcYImageStrideInBytes + srcYImageStrideInBytes);
		pSrcUImage += srcUImageStrideInBytes;
		pSrcVImage += srcVImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes + dstImageStrideInBytes);
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128 Y00, Y01, Y10, Y11, U, V;
	__m128i Y0pix, Y1pix, UVpix;
	__m128i shufMask = _mm_set_epi8(-1, 14, 13, 12, -1, 10, 9, 8, -1, 6, 5, 4, -1, 2, 1, 0);

	// BT 709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care

	for (int height = 0; height < (int)dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrcLuma = pSrcLumaImage;
		vx_uint8 * pLocalSrcChroma = pSrcChromaImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)	// Process 16 pixels at a time
		{
			Y0pix = _mm_loadu_si128((__m128i *) pLocalSrcLuma);
			Y1pix = _mm_loadu_si128((__m128i *) (pLocalSrcLuma + srcLumaImageStrideInBytes));
			UVpix = _mm_loadu_si128((__m128i *) pLocalSrcChroma);

			for (int i = 0; i < 4; i++)
			{
				// For pixels 00, 01
				//			  10, 11
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				U = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				V = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 00
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 01
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 10
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 11

				__m128i tempI0 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB00, RGB01 to U8
				__m128i tempI1 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB10, RGB11 to U8

				// For pixels 02, 03
				//			  12, 13
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				U = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				V = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 02
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 03
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 12
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 13

				__m128i tempI2 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB02, RGB03 to U8
				tempI0 = _mm_packus_epi16(tempI0, tempI2);
				tempI0 = _mm_shuffle_epi8(tempI0, shufMask);
				tempI0 = _mm_or_si128(tempI0, _mm_set1_epi32(0xFF000000));
				_mm_storeu_si128((__m128i *)pLocalDst, tempI0);

				__m128i tempI3 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB12, RGB13 to U8
				tempI1 = _mm_packus_epi16(tempI1, tempI3);
				tempI1 = _mm_shuffle_epi8(tempI1, shufMask);
				tempI1 = _mm_or_si128(tempI1, _mm_set1_epi32(0xFF000000));
				_mm_storeu_si128((__m128i *)(pLocalDst + dstImageStrideInBytes), tempI1);
				
				pLocalDst += 16;
			}
			pLocalSrcLuma += 16;
			pLocalSrcChroma += 16;
		}

		for (int width = 0; width < (postfixWidth >> 1); width ++)		// Processing two pixels at a time in a row
		{
			float Ypix, Rpix, Gpix, Bpix;

			Ypix = (float)(*pLocalSrcLuma);
			Bpix = (float)(*pLocalSrcChroma++) - 128.0f;
			Rpix = (float)(*pLocalSrcChroma++) - 128.0f;

			Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
			Rpix *= 1.5748f;
			Bpix *= 1.8556f;

			*pLocalDst = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + 3) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcLuma + 1));
			*(pLocalDst + 4) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 5) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 6) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + 7) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcLuma + srcLumaImageStrideInBytes));
			*(pLocalDst + dstImageStrideInBytes + 0) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 3) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcLuma + srcLumaImageStrideInBytes + 1));
			*(pLocalDst + dstImageStrideInBytes + 4) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 5) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 6) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 7) = (vx_uint8)255;

			pLocalSrcLuma += 2;
			pLocalDst += 8;
		}
		pSrcLumaImage += (srcLumaImageStrideInBytes + srcLumaImageStrideInBytes);
		pSrcChromaImage += srcChromaImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes + dstImageStrideInBytes);
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128 Y00, Y01, Y10, Y11, U, V;
	__m128i Y0pix, Y1pix, UVpix;
	__m128i shufMask = _mm_set_epi8(-1, 14, 13, 12, -1, 10, 9, 8, -1, 6, 5, 4, -1, 2, 1, 0);

	// BT 709 conversion factors
	__m128 weights_U2RGB = _mm_set_ps(0.0f, 1.8556f, -0.1873f, 0.0f);		// x R G B, The most significant float is don't care
	__m128 weights_V2RGB = _mm_set_ps(0.0f, 0.0f, -0.4681f, 1.5748f);		// x R G B, The most significant float is don't care

	for (int height = 0; height < (int)dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrcLuma = pSrcLumaImage;
		vx_uint8 * pLocalSrcChroma = pSrcChromaImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)	// Process 16 pixels at a time
		{
			Y0pix = _mm_loadu_si128((__m128i *) pLocalSrcLuma);
			Y1pix = _mm_loadu_si128((__m128i *) (pLocalSrcLuma + srcLumaImageStrideInBytes));
			UVpix = _mm_loadu_si128((__m128i *) pLocalSrcChroma);

			for (int i = 0; i < 4; i++)
			{
				// For pixels 00, 01
				//			  10, 11
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				V = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 00
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 01
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 10
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 11

				__m128i tempI0 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB00, RGB01 to U8
				__m128i tempI1 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB10, RGB11 to U8

				// For pixels 02, 03
				//			  12, 13
				Y00 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y01 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y0pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y0pix = _mm_srli_si128(Y0pix, 1);
				Y10 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				Y11 = _mm_cvtepi32_ps(_mm_shuffle_epi8(Y1pix, _mm_set1_epi32((int)0xFFFFFF00)));
				Y1pix = _mm_srli_si128(Y1pix, 1);
				V = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				V = _mm_sub_ps(V, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_cvtepi32_ps(_mm_shuffle_epi8(UVpix, _mm_set1_epi32((int)0xFFFFFF00)));
				U = _mm_sub_ps(U, _mm_set1_ps(128.0f));
				UVpix = _mm_srli_si128(UVpix, 1);
				U = _mm_mul_ps(U, weights_U2RGB);
				V = _mm_mul_ps(V, weights_V2RGB);
				U = _mm_add_ps(U, V);									// weights_U*U + weights_V*V
				Y00 = _mm_add_ps(Y00, U);								// RGB for pixel 02
				Y01 = _mm_add_ps(Y01, U);								// RGB for pixel 03
				Y10 = _mm_add_ps(Y10, U);								// RGB for pixel 12
				Y11 = _mm_add_ps(Y11, U);								// RGB for pixel 13

				__m128i tempI2 = _mm_packus_epi32(_mm_cvttps_epi32(Y00), _mm_cvttps_epi32(Y01));	// Convert RGB02, RGB03 to U8
				tempI0 = _mm_packus_epi16(tempI0, tempI2);
				tempI0 = _mm_shuffle_epi8(tempI0, shufMask);
				tempI0 = _mm_or_si128(tempI0, _mm_set1_epi32(0xFF000000));
				_mm_storeu_si128((__m128i *)pLocalDst, tempI0);

				__m128i tempI3 = _mm_packus_epi32(_mm_cvttps_epi32(Y10), _mm_cvttps_epi32(Y11));	// Convert RGB12, RGB13 to U8
				tempI1 = _mm_packus_epi16(tempI1, tempI3);
				tempI1 = _mm_shuffle_epi8(tempI1, shufMask);
				tempI1 = _mm_or_si128(tempI1, _mm_set1_epi32(0xFF000000));
				_mm_storeu_si128((__m128i *)(pLocalDst + dstImageStrideInBytes), tempI1);

				pLocalDst += 16;
			}
			pLocalSrcLuma += 16;
			pLocalSrcChroma += 16;
		}

		for (int width = 0; width < (postfixWidth >> 1); width ++)		// Processing two pixels at a time in a row
		{
			float Ypix, Rpix, Gpix, Bpix;

			Ypix = (float)(*pLocalSrcLuma);
			Rpix = (float)(*pLocalSrcChroma++) - 128.0f;
			Bpix = (float)(*pLocalSrcChroma++) - 128.0f;

			Gpix = (Bpix * 0.1873f) + (Rpix * 0.4681f);
			Rpix *= 1.5748f;
			Bpix *= 1.8556f;

			*pLocalDst = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + 3) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcLuma + 1));
			*(pLocalDst + 4) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + 5) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + 6) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + 7) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcLuma + srcLumaImageStrideInBytes));
			*(pLocalDst + dstImageStrideInBytes + 0) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 1) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 2) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 3) = (vx_uint8)255;

			Ypix = (float)(*(pLocalSrcLuma + srcLumaImageStrideInBytes + 1));
			*(pLocalDst + dstImageStrideInBytes + 4) = (vx_uint8)fminf(fmaxf(Ypix + Rpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 5) = (vx_uint8)fminf(fmaxf(Ypix - Gpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 6) = (vx_uint8)fminf(fmaxf(Ypix + Bpix, 0.0f), 255.0f);
			*(pLocalDst + dstImageStrideInBytes + 7) = (vx_uint8)255;

			pLocalSrcLuma += 2;
			pLocalDst += 8;
		}
		pSrcLumaImage += (srcLumaImageStrideInBytes + srcLumaImageStrideInBytes);
		pSrcChromaImage += srcChromaImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes + dstImageStrideInBytes);
	}
	return AGO_SUCCESS;
}

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
	)
{
	unsigned char *pLocalSrc, *pLocalDstY, *pLocalDstU, *pLocalDstV;
	unsigned char *pLocalSrcNextRow, *pLocalDstYNextRow;

	__m128i * tbl = (__m128i*) dataColorConvert;
	__m128i maskY = _mm_load_si128(tbl + 3);
	__m128i maskU = _mm_load_si128(tbl + 4);
	__m128i maskV = _mm_load_si128(tbl + 5);
	__m128i pixels0, pixels1, pixels0_NextRow, pixels1_NextRow, temp0, temp1;

	bool isAligned = (((intptr_t(pDstYImage) & intptr_t(pDstUImage) & intptr_t(pDstVImage)) & 7) == ((intptr_t(pDstYImage) | intptr_t(pDstUImage) | intptr_t(pDstVImage)) & 7));		// Check for 8 byte alignment
	isAligned = isAligned & ((intptr_t(pDstYImage) & 8) == 0);					// Y image should be 16 byte aligned or have same alignment as the Chroma planes

	if (isAligned)
	{
		int prefixWidth = intptr_t(pDstYImage) & 15;
		prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
		int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time
		int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalSrcNextRow = (unsigned char *)pSrcImage + srcImageStrideInBytes;
			pLocalDstY = (unsigned char *)pDstYImage;
			pLocalDstYNextRow = (unsigned char *)pDstYImage + dstYImageStrideInBytes;
			pLocalDstU = (unsigned char *)pDstUImage;
			pLocalDstV = (unsigned char *)pDstVImage;

			for (int x = 0; x < prefixWidth; x+=2)
			{
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstU++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// U
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstV++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// V
			}

			int width = alignedWidth >> 4;												// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels0_NextRow = _mm_loadu_si128((__m128i *) pLocalSrcNextRow);
				pixels1_NextRow = _mm_loadu_si128((__m128i *) (pLocalSrcNextRow + 16));

				temp0 = _mm_shuffle_epi8(pixels0, maskY);								// Y plane, bytes 0..7
				temp1 = _mm_shuffle_epi8(pixels1, maskY);								// Y plane, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_store_si128((__m128i *) pLocalDstY, temp0);

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskY);						// Y plane - next row, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_shuffle_epi8(pixels0_NextRow, maskY);						// Y plane - next row, bytes 0..7
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_store_si128((__m128i *) pLocalDstYNextRow, temp0);

				temp1 = _mm_shuffle_epi8(pixels1, maskU);								// U plane, intermideate bytes 4..7
				pixels1 = _mm_shuffle_epi8(pixels1, maskV);								// V plane, intermideate bytes 4..7
				temp1 = _mm_slli_si128(temp1, 4);
				pixels1 = _mm_slli_si128(pixels1, 4);

				temp0 = _mm_shuffle_epi8(pixels0, maskU);								// U plane, intermideate bytes 0..3
				pixels0 = _mm_shuffle_epi8(pixels0, maskV);								// V plane, intermideate bytes 0..3
				temp0 = _mm_or_si128(temp0, temp1);										// U plane, intermideate bytes 0..7
				pixels0 = _mm_or_si128(pixels0, pixels1);								// V plane, intermideate bytes 0..7
				
				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskU);						// U plane - next row, intermideate bytes 4..7
				pixels1_NextRow = _mm_shuffle_epi8(pixels1_NextRow, maskV);				// V plane - next row, intermideate bytes 4..7
				temp1 = _mm_slli_si128(temp1, 4);
				pixels1_NextRow = _mm_slli_si128(pixels1_NextRow, 4);

				pixels1 = _mm_shuffle_epi8(pixels0_NextRow, maskU);						// U plane - next row, intermideate bytes 0..3
				pixels0_NextRow = _mm_shuffle_epi8(pixels0_NextRow, maskV);				// V plane - next row, intermideate bytes 0..3
				temp1 = _mm_or_si128(temp1, pixels1);									// U plane - next row, intermideate bytes 0..7
				pixels0_NextRow = _mm_or_si128(pixels0_NextRow, pixels1_NextRow);		// V plane - next row, intermideate bytes 0..7

				__m128i avg0, avg1;
				avg0 = _mm_avg_epu8(temp0, temp1);										// U plane, bytes 0..7
				// to fix it up by subtracting 1 if it got rounded up
				avg0 = _mm_sub_epi8(avg0, _mm_and_si128(_mm_xor_si128(temp0, temp1), _mm_set1_epi8(1)));
				*((int64_t *)pLocalDstU) = M128I(avg0).m128i_i64[0];
				avg1 = _mm_avg_epu8(pixels0, pixels0_NextRow);							// V plane, bytes 0..7
				// to fix it up by subtracting 1 if it got rounded up
				avg1 = _mm_sub_epi8(avg1, _mm_and_si128(_mm_xor_si128(pixels0, pixels0_NextRow), _mm_set1_epi8(1)));
				*((int64_t *)pLocalDstV) = M128I(avg1).m128i_i64[0];

				pLocalSrc += 32;
				pLocalSrcNextRow += 32;
				pLocalDstY += 16;
				pLocalDstYNextRow += 16;
				pLocalDstU += 8;
				pLocalDstV += 8;
				width--;
			}

			for (int x = 0; x < postfixWidth; x+=2)
			{
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstU++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// U
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstV++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// V
			}

			pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);				// Advance by 2 rows
			pDstYImage += (dstYImageStrideInBytes + dstYImageStrideInBytes);			// Advance by 2 rows
			pDstUImage += dstUImageStrideInBytes;
			pDstVImage += dstVImageStrideInBytes;

			height -= 2;
		}
	}
	else
	{
		int postfixWidth = (int)dstWidth & 15;
		int alignedWidth = (int)dstWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalSrcNextRow = (unsigned char *)pSrcImage + srcImageStrideInBytes;
			pLocalDstY = (unsigned char *)pDstYImage;
			pLocalDstYNextRow = (unsigned char *)pDstYImage + dstYImageStrideInBytes;
			pLocalDstU = (unsigned char *)pDstUImage;
			pLocalDstV = (unsigned char *)pDstVImage;

			int width = alignedWidth >> 4;												// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels0_NextRow = _mm_loadu_si128((__m128i *) pLocalSrcNextRow);
				pixels1_NextRow = _mm_loadu_si128((__m128i *) (pLocalSrcNextRow + 16));

				temp0 = _mm_shuffle_epi8(pixels0, maskY);								// Y plane, bytes 0..7
				temp1 = _mm_shuffle_epi8(pixels1, maskY);								// Y plane, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_storeu_si128((__m128i *) pLocalDstY, temp0);

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskY);						// Y plane - next row, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_shuffle_epi8(pixels0_NextRow, maskY);						// Y plane - next row, bytes 0..7
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_storeu_si128((__m128i *) pLocalDstYNextRow, temp0);

				temp1 = _mm_shuffle_epi8(pixels1, maskU);								// U plane, intermideate bytes 4..7
				pixels1 = _mm_shuffle_epi8(pixels1, maskV);								// V plane, intermideate bytes 4..7
				temp1 = _mm_slli_si128(temp1, 4);
				pixels1 = _mm_slli_si128(pixels1, 4);

				temp0 = _mm_shuffle_epi8(pixels0, maskU);								// U plane, intermideate bytes 0..3
				pixels0 = _mm_shuffle_epi8(pixels0, maskV);								// V plane, intermideate bytes 0..3
				temp0 = _mm_or_si128(temp0, temp1);										// U plane, intermideate bytes 0..7
				pixels0 = _mm_or_si128(pixels0, pixels1);								// V plane, intermideate bytes 0..7
				
				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskU);						// U plane - next row, intermideate bytes 4..7
				pixels1_NextRow = _mm_shuffle_epi8(pixels1_NextRow, maskV);				// V plane - next row, intermideate bytes 4..7
				temp1 = _mm_slli_si128(temp1, 4);
				pixels1_NextRow = _mm_slli_si128(pixels1_NextRow, 4);

				pixels1 = _mm_shuffle_epi8(pixels0_NextRow, maskU);						// U plane - next row, intermideate bytes 0..3
				pixels0_NextRow = _mm_shuffle_epi8(pixels0_NextRow, maskV);				// V plane - next row, intermideate bytes 0..3
				temp1 = _mm_or_si128(temp1, pixels1);									// U plane - next row, intermideate bytes 0..7
				pixels0_NextRow = _mm_or_si128(pixels0_NextRow, pixels1_NextRow);		// V plane - next row, intermideate bytes 0..7

				__m128i avg0, avg1;
				avg0 = _mm_avg_epu8(temp0, temp1);										// U plane, bytes 0..7
				// to fix it up by subtracting 1 if it got rounded up
				avg0 = _mm_sub_epi8(avg0, _mm_and_si128(_mm_xor_si128(temp0, temp1), _mm_set1_epi8(1)));
				_mm_storeu_si128((__m128i *) pLocalDstU, avg0);						// Only lower 8 bytes valid
				avg1 = _mm_avg_epu8(pixels0, pixels0_NextRow);							// V plane, bytes 0..7
				// to fix it up by subtracting 1 if it got rounded up
				avg1 = _mm_sub_epi8(avg1, _mm_and_si128(_mm_xor_si128(pixels0, pixels0_NextRow), _mm_set1_epi8(1)));
				_mm_storeu_si128((__m128i *) pLocalDstV, avg1);						// Only lower 8 bytes valid

				pLocalSrc += 32;
				pLocalSrcNextRow += 32;
				pLocalDstY += 16;
				pLocalDstYNextRow += 16;
				pLocalDstU += 8;
				pLocalDstV += 8;
				width--;
			}

			for (int x = 0; x < postfixWidth; x+=2)
			{
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstU++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// U
				*pLocalDstY++ = *pLocalSrc++;											// Y
				*pLocalDstYNextRow++ = *pLocalSrcNextRow++;								// Y - next row
				*pLocalDstV++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;				// V
			}

			pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);				// Advance by 2 rows
			pDstYImage += (dstYImageStrideInBytes + dstYImageStrideInBytes);			// Advance by 2 rows
			pDstUImage += dstUImageStrideInBytes;
			pDstVImage += dstVImageStrideInBytes;

			height -= 2;
		}
	}
	return AGO_SUCCESS;
}

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
	)
{
	unsigned char *pLocalSrc, *pLocalDstLuma, *pLocalDstChroma;
	unsigned char *pLocalSrcNextRow, *pLocalDstLumaNextRow;

	__m128i * tbl = (__m128i*) dataColorConvert;
	__m128i maskLuma = _mm_load_si128(tbl);
	__m128i maskChroma = _mm_load_si128(tbl + 3);
	__m128i pixels0, pixels1, pixels0_NextRow, pixels1_NextRow, temp0, temp1;

	bool isAligned = ((intptr_t(pDstLumaImage) & 15) == (intptr_t(pDstChromaImage) & 15));

	if (isAligned)													// Optimized routine for both dst images at same alignment
	{
		int prefixWidth = intptr_t(pDstLumaImage) & 15;
		prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
		int postfixWidth = ((int) dstWidth - prefixWidth) & 15;
		int alignedWidth = (int) dstWidth - prefixWidth - postfixWidth;

		int height = (int) dstHeight;
		while (height > 0)
		{
			pLocalSrc = (unsigned char *) pSrcImage;
			pLocalDstLuma = (unsigned char *) pDstLumaImage;
			pLocalDstChroma = (unsigned char *) pDstChromaImage;
			pLocalSrcNextRow = (unsigned char *) pSrcImage + srcImageStrideInBytes;
			pLocalDstLumaNextRow = (unsigned char *) pDstLumaImage + dstLumaImageStrideInBytes;
			
			for (int x = 0; x < prefixWidth; x += 2)
			{
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// U
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// V
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
			}

			int width = alignedWidth >> 4;												// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels0_NextRow = _mm_loadu_si128((__m128i *) pLocalSrcNextRow);
				pixels1_NextRow = _mm_loadu_si128((__m128i *) (pLocalSrcNextRow + 16));

				temp0 = _mm_shuffle_epi8(pixels0, maskLuma);							// Y plane, bytes 0..7
				temp1 = _mm_shuffle_epi8(pixels1, maskLuma);							// Y plane, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_store_si128((__m128i *) pLocalDstLuma, temp0);

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskLuma);					// Y plane - next row, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_shuffle_epi8(pixels0_NextRow, maskLuma);					// Y plane - next row, bytes 0..7
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_store_si128((__m128i *) pLocalDstLumaNextRow, temp0);

				pixels0 = _mm_shuffle_epi8(pixels0, maskChroma);						// Chroma plane, bytes 0..7
				pixels0_NextRow = _mm_shuffle_epi8(pixels0_NextRow, maskChroma);		// Chroma plane - Next row, bytes 0..7
				pixels1 = _mm_shuffle_epi8(pixels1, maskChroma);						// Chroma plane, bytes 8..15
				pixels1_NextRow = _mm_shuffle_epi8(pixels1_NextRow, maskChroma);		// Chroma plane - Next row, bytes 8..15
				
				pixels1 = _mm_slli_si128(pixels1, 8);
				pixels1_NextRow = _mm_slli_si128(pixels1_NextRow, 8);
				pixels0 = _mm_or_si128(pixels0, pixels1);
				pixels0_NextRow = _mm_or_si128(pixels0_NextRow, pixels1_NextRow);
				__m128i avg;
				avg = _mm_avg_epu8(pixels0, pixels0_NextRow);
				// to fix it up by subtracting 1 if it got rounded up
				avg = _mm_sub_epi8(avg, _mm_and_si128(_mm_xor_si128(pixels0, pixels0_NextRow), _mm_set1_epi8(1)));
				_mm_store_si128((__m128i *) pLocalDstChroma, avg);

				pLocalSrc += 32;
				pLocalSrcNextRow += 32;
				pLocalDstLuma += 16;
				pLocalDstLumaNextRow += 16;
				pLocalDstChroma += 16;
				width--;
			}

			for (int x = 0; x < postfixWidth; x += 2)
			{
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// U
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// V
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
			}

			pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);				// Advance by 2 rows
			pDstLumaImage += (dstLumaImageStrideInBytes + dstLumaImageStrideInBytes);	// Advance by 2 rows
			pDstChromaImage += dstChromaImageStrideInBytes;

			height -= 2;
		}
		
	}
	else
	{
		int postfixWidth = (int)dstWidth & 15;
		int alignedWidth = (int)dstWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height > 0)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDstLuma = (unsigned char *)pDstLumaImage;
			pLocalDstChroma = (unsigned char *)pDstChromaImage;
			pLocalSrcNextRow = (unsigned char *)pSrcImage + srcImageStrideInBytes;
			pLocalDstLumaNextRow = (unsigned char *)pDstLumaImage + dstLumaImageStrideInBytes;

			int width = alignedWidth >> 4;												// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels0_NextRow = _mm_loadu_si128((__m128i *) pLocalSrcNextRow);
				pixels1_NextRow = _mm_loadu_si128((__m128i *) (pLocalSrcNextRow + 16));

				temp0 = _mm_shuffle_epi8(pixels0, maskLuma);							// Y plane, bytes 0..7
				temp1 = _mm_shuffle_epi8(pixels1, maskLuma);							// Y plane, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_storeu_si128((__m128i *) pLocalDstLuma, temp0);

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskLuma);					// Y plane - next row, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_shuffle_epi8(pixels0_NextRow, maskLuma);					// Y plane - next row, bytes 0..7
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_storeu_si128((__m128i *) pLocalDstLumaNextRow, temp0);

				pixels0 = _mm_shuffle_epi8(pixels0, maskChroma);						// Chroma plane, bytes 0..7
				pixels0_NextRow = _mm_shuffle_epi8(pixels0_NextRow, maskChroma);		// Chroma plane - Next row, bytes 0..7
				pixels1 = _mm_shuffle_epi8(pixels1, maskChroma);						// Chroma plane, bytes 8..15
				pixels1_NextRow = _mm_shuffle_epi8(pixels1_NextRow, maskChroma);		// Chroma plane - Next row, bytes 8..15

				pixels1 = _mm_slli_si128(pixels1, 8);
				pixels1_NextRow = _mm_slli_si128(pixels1_NextRow, 8);
				pixels0 = _mm_or_si128(pixels0, pixels1);
				pixels0_NextRow = _mm_or_si128(pixels0_NextRow, pixels1_NextRow);
				__m128i avg;
				avg = _mm_avg_epu8(pixels0, pixels0_NextRow);
				// to fix it up by subtracting 1 if it got rounded up
				avg = _mm_sub_epi8(avg, _mm_and_si128(_mm_xor_si128(pixels0, pixels0_NextRow), _mm_set1_epi8(1)));
				_mm_storeu_si128((__m128i *) pLocalDstChroma, avg);

				pLocalSrc += 32;
				pLocalSrcNextRow += 32;
				pLocalDstLuma += 16;
				pLocalDstLumaNextRow += 16;
				pLocalDstChroma += 16;
				width--;
			}

			for (int x = 0; x < postfixWidth; x += 2)
			{
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// U
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// V
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
			}

			pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);				// Advance by 2 rows
			pDstLumaImage += (dstLumaImageStrideInBytes + dstLumaImageStrideInBytes);	// Advance by 2 rows
			pDstChromaImage += dstChromaImageStrideInBytes;

			height -= 2;
		}	
	}
	return AGO_SUCCESS;
}

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
	)
{
	unsigned char *pLocalSrc, *pLocalDstLuma, *pLocalDstChroma;
	unsigned char *pLocalSrcNextRow, *pLocalDstLumaNextRow;

	__m128i * tbl = (__m128i*) dataColorConvert;
	__m128i maskLuma = _mm_load_si128(tbl + 3);
	__m128i maskChroma = _mm_load_si128(tbl);
	__m128i pixels0, pixels1, pixels0_NextRow, pixels1_NextRow, temp0, temp1;

	bool isAligned = ((intptr_t(pDstLumaImage) & 15) == (intptr_t(pDstChromaImage) & 15));

	if (isAligned)													// Optimized routine for both dst images at same alignment
	{
		int prefixWidth = intptr_t(pDstLumaImage) & 15;
		prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
		int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
		int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height > 0)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDstLuma = (unsigned char *)pDstLumaImage;
			pLocalDstChroma = (unsigned char *)pDstChromaImage;
			pLocalSrcNextRow = (unsigned char *)pSrcImage + srcImageStrideInBytes;
			pLocalDstLumaNextRow = (unsigned char *)pDstLumaImage + dstLumaImageStrideInBytes;

			for (int x = 0; x < prefixWidth; x += 2)
			{
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// U
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// V
			}

			int width = alignedWidth >> 4;												// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels0_NextRow = _mm_loadu_si128((__m128i *) pLocalSrcNextRow);
				pixels1_NextRow = _mm_loadu_si128((__m128i *) (pLocalSrcNextRow + 16));

				temp0 = _mm_shuffle_epi8(pixels0, maskLuma);							// Y plane, bytes 0..7
				temp1 = _mm_shuffle_epi8(pixels1, maskLuma);							// Y plane, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_store_si128((__m128i *) pLocalDstLuma, temp0);

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskLuma);					// Y plane - next row, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_shuffle_epi8(pixels0_NextRow, maskLuma);					// Y plane - next row, bytes 0..7
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_store_si128((__m128i *) pLocalDstLumaNextRow, temp0);

				pixels0 = _mm_shuffle_epi8(pixels0, maskChroma);						// Chroma plane, bytes 0..7
				pixels0_NextRow = _mm_shuffle_epi8(pixels0_NextRow, maskChroma);		// Chroma plane - Next row, bytes 0..7
				pixels1 = _mm_shuffle_epi8(pixels1, maskChroma);						// Chroma plane, bytes 8..15
				pixels1_NextRow = _mm_shuffle_epi8(pixels1_NextRow, maskChroma);		// Chroma plane - Next row, bytes 8..15

				pixels1 = _mm_slli_si128(pixels1, 8);
				pixels1_NextRow = _mm_slli_si128(pixels1_NextRow, 8);
				pixels0 = _mm_or_si128(pixels0, pixels1);
				pixels0_NextRow = _mm_or_si128(pixels0_NextRow, pixels1_NextRow);
				__m128i avg;
				avg = _mm_avg_epu8(pixels0, pixels0_NextRow);
				// to fix it up by subtracting 1 if it got rounded up
				avg = _mm_sub_epi8(avg, _mm_and_si128(_mm_xor_si128(pixels0, pixels0_NextRow), _mm_set1_epi8(1)));
				_mm_store_si128((__m128i *) pLocalDstChroma, avg);

				pLocalSrc += 32;
				pLocalSrcNextRow += 32;
				pLocalDstLuma += 16;
				pLocalDstLumaNextRow += 16;
				pLocalDstChroma += 16;
				width--;
			}

			for (int x = 0; x < postfixWidth; x += 2)
			{
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// U
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// V				
			}

			pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);				// Advance by 2 rows
			pDstLumaImage += (dstLumaImageStrideInBytes + dstLumaImageStrideInBytes);	// Advance by 2 rows
			pDstChromaImage += dstChromaImageStrideInBytes;

			height -= 2;
		}

	}
	else
	{
		int postfixWidth = (int)dstWidth & 15;
		int alignedWidth = (int)dstWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height > 0)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDstLuma = (unsigned char *)pDstLumaImage;
			pLocalDstChroma = (unsigned char *)pDstChromaImage;
			pLocalSrcNextRow = (unsigned char *)pSrcImage + srcImageStrideInBytes;
			pLocalDstLumaNextRow = (unsigned char *)pDstLumaImage + dstLumaImageStrideInBytes;

			int width = alignedWidth >> 4;												// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels0_NextRow = _mm_loadu_si128((__m128i *) pLocalSrcNextRow);
				pixels1_NextRow = _mm_loadu_si128((__m128i *) (pLocalSrcNextRow + 16));

				temp0 = _mm_shuffle_epi8(pixels0, maskLuma);							// Y plane, bytes 0..7
				temp1 = _mm_shuffle_epi8(pixels1, maskLuma);							// Y plane, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_storeu_si128((__m128i *) pLocalDstLuma, temp0);

				temp1 = _mm_shuffle_epi8(pixels1_NextRow, maskLuma);					// Y plane - next row, bytes 8..15
				temp1 = _mm_slli_si128(temp1, 8);
				temp0 = _mm_shuffle_epi8(pixels0_NextRow, maskLuma);					// Y plane - next row, bytes 0..7
				temp0 = _mm_or_si128(temp0, temp1);
				_mm_storeu_si128((__m128i *) pLocalDstLumaNextRow, temp0);

				pixels0 = _mm_shuffle_epi8(pixels0, maskChroma);						// Chroma plane, bytes 0..7
				pixels0_NextRow = _mm_shuffle_epi8(pixels0_NextRow, maskChroma);		// Chroma plane - Next row, bytes 0..7
				pixels1 = _mm_shuffle_epi8(pixels1, maskChroma);						// Chroma plane, bytes 8..15
				pixels1_NextRow = _mm_shuffle_epi8(pixels1_NextRow, maskChroma);		// Chroma plane - Next row, bytes 8..15

				pixels1 = _mm_slli_si128(pixels1, 8);
				pixels1_NextRow = _mm_slli_si128(pixels1_NextRow, 8);
				pixels0 = _mm_or_si128(pixels0, pixels1);
				pixels0_NextRow = _mm_or_si128(pixels0_NextRow, pixels1_NextRow);
				__m128i avg;
				avg = _mm_avg_epu8(pixels0, pixels0_NextRow);
				// to fix it up by subtracting 1 if it got rounded up
				avg = _mm_sub_epi8(avg, _mm_and_si128(_mm_xor_si128(pixels0, pixels0_NextRow), _mm_set1_epi8(1)));
				_mm_store_si128((__m128i *) pLocalDstChroma, avg);

				pLocalSrc += 32;
				pLocalSrcNextRow += 32;
				pLocalDstLuma += 16;
				pLocalDstLumaNextRow += 16;
				pLocalDstChroma += 16;
				width--;
			}

			for (int x = 0; x < postfixWidth; x += 2)
			{
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// U
				*pLocalDstLuma++ = *pLocalSrc++;										// Y
				*pLocalDstLumaNextRow++ = *pLocalSrcNextRow++;							// Y - next row
				*pLocalDstChroma++ = (*pLocalSrc++ + *pLocalSrcNextRow++) >> 1;			// V
			}

			pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);				// Advance by 2 rows
			pDstLumaImage += (dstLumaImageStrideInBytes + dstLumaImageStrideInBytes);	// Advance by 2 rows
			pDstChromaImage += dstChromaImageStrideInBytes;

			height -= 2;
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_RGB_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataColorConvert;

	__m128i mask_1_1 = _mm_load_si128(tbl + 8);				// First 16 bytes -RGBX- to first 16 bytes - RGB
	__m128i mask_1_2 = _mm_load_si128(tbl + 9);				// Second 16 bytes -RGBX- to first 16 bytes - RGB
	__m128i mask_2_2 = _mm_load_si128(tbl + 10);			// Second 16 bytes -RGBX- to second 16 bytes - RGB
	__m128i mask_2_3 = _mm_load_si128(tbl + 11);			// Third 16 bytes -RGBX- to second 16 bytes - RGB
	__m128i mask_3_3 = _mm_load_si128(tbl + 12);			// Third 16 bytes -RGBX- to third 16 bytes - RGB
	__m128i mask_3_4 = _mm_load_si128(tbl + 13);			// Fourth 16 bytes -RGBX- to third 16 bytes - RGB
	__m128i pixels1, pixels2, pixels3, pixels4, temp;

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = (vx_uint8 *)pSrcImage;
		vx_uint8 * pLocalDst = (vx_uint8 *)pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			pixels1 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			pixels3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));
			pixels4 = _mm_loadu_si128((__m128i *)(pLocalSrc + 48));

			pixels4 = _mm_shuffle_epi8(pixels4, mask_3_4);
			temp = _mm_shuffle_epi8(pixels3, mask_3_3);
			pixels4 = _mm_or_si128(pixels4, temp);

			pixels3 = _mm_shuffle_epi8(pixels3, mask_2_3);
			temp = _mm_shuffle_epi8(pixels2, mask_2_2);
			pixels3 = _mm_or_si128(pixels3, temp);

			pixels2 = _mm_shuffle_epi8(pixels2, mask_1_2);
			temp = _mm_shuffle_epi8(pixels1, mask_1_1);
			pixels2 = _mm_or_si128(pixels2, temp);

			_mm_storeu_si128((__m128i *)pLocalDst, pixels2);
			_mm_storeu_si128((__m128i *)(pLocalDst + 16), pixels3);
			_mm_storeu_si128((__m128i *)(pLocalDst + 32), pixels4);

			pLocalDst += 48;
			pLocalSrc += 64;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			*pLocalDst++ = *pLocalSrc++;
			*pLocalDst++ = *pLocalSrc++;
			*pLocalDst++ = *pLocalSrc++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_RGBX_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 2;														// 4 bytes = 1 pixel
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	unsigned char *pLocalSrc, *pLocalDst;
	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	__m128i * tbl = (__m128i*) dataColorConvert;

	__m128i mask_1_1 = _mm_load_si128(tbl + 14);			// First 16 bytes -RGB- to first 16 bytes - RGBX
	__m128i mask_2_1 = _mm_load_si128(tbl + 15);			// First 16 bytes -RGB- to second 16 bytes - RGBX
	__m128i mask_2_2 = _mm_load_si128(tbl + 16);			// Second 16 bytes -RGB- to second 16 bytes - RGBX
	__m128i mask_3_2 = _mm_load_si128(tbl + 17);			// Second 16 bytes -RGB- to third 16 bytes - RGBX
	__m128i mask_3_3 = _mm_load_si128(tbl + 18);			// Third 16 bytes -RGB- to third 16 bytes - RGBX
	__m128i mask_4_3 = _mm_load_si128(tbl + 19);			// Third 16 bytes -RGB- to fourth 16 bytes - RGBX
	__m128i mask_fill = _mm_load_si128(tbl + 20);			// Fill in 255 at the X positions
	__m128i pixels1, pixels2, pixels3, pixels4, temp;

	int height = (int) dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *) pSrcImage;
		pLocalDst = (unsigned char *) pDstImage;
		for (int x = 0; x < prefixWidth; x++)
		{
			*pLocalDst++ = *pLocalSrc++;					// R
			*pLocalDst++ = *pLocalSrc++;					// G
			*pLocalDst++ = *pLocalSrc++;					// B
			*pLocalDst++ = (unsigned char)255;
		}

		pLocalSrc_xmm = (__m128i *) pLocalSrc;
		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);				// 16 pixels processed at a time
		while (width)
		{
			pixels1 = _mm_loadu_si128(pLocalSrc_xmm++);
			pixels2 = _mm_loadu_si128(pLocalSrc_xmm++);
			pixels3 = _mm_loadu_si128(pLocalSrc_xmm++);

			pixels4 = _mm_shuffle_epi8(pixels3, mask_4_3);

			pixels3 = _mm_shuffle_epi8(pixels3, mask_3_3);
			temp = _mm_shuffle_epi8(pixels2, mask_3_2);
			pixels3 = _mm_or_si128(pixels3, temp);

			pixels2 = _mm_shuffle_epi8(pixels2, mask_2_2);
			temp = _mm_shuffle_epi8(pixels1, mask_2_1);
			pixels2 = _mm_or_si128(pixels2, temp);

			pixels1 = _mm_shuffle_epi8(pixels1, mask_1_1);

			pixels1 = _mm_or_si128(pixels1, mask_fill);
			pixels2 = _mm_or_si128(pixels2, mask_fill);
			pixels3 = _mm_or_si128(pixels3, mask_fill);
			pixels4 = _mm_or_si128(pixels4, mask_fill);

			_mm_store_si128(pLocalDst_xmm++, pixels1);
			_mm_store_si128(pLocalDst_xmm++, pixels2);
			_mm_store_si128(pLocalDst_xmm++, pixels3);
			_mm_store_si128(pLocalDst_xmm++, pixels4);

			width--;
		}

		pLocalSrc = (unsigned char *) pLocalSrc_xmm;
		pLocalDst = (unsigned char *) pLocalDst_xmm;
		for (int x = 0; x < postfixWidth; x++)
		{
			*pLocalDst++ = *pLocalSrc++;					// R
			*pLocalDst++ = *pLocalSrc++;					// G
			*pLocalDst++ = *pLocalSrc++;					// B
			*pLocalDst++ = (unsigned char)255;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~3;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataColorConvert;
	__m128i mask = _mm_load_si128(tbl + 14);		// 0 B3 G3 R3 0 B2 G2 R2 0 B1 G1 R1 0 B0 G0 R0
	__m128i cvtmask = _mm_set1_epi32(255);			// 0 0 0 FF 0 0 0 FF 0 0 0 FF 0 0 0 FF
	__m128i row0, row1, tempI;
	__m128 Y0, U0, V0, Y1, U1, V1, weights_toY, weights_toU, weights_toV, temp, temp2;
	__m128 const128 = _mm_set1_ps(128.0f);

	DECL_ALIGN(16) unsigned int Ybuf[8] ATTR_ALIGN(16);
	DECL_ALIGN(16) unsigned short Ubuf[8] ATTR_ALIGN(16);
	DECL_ALIGN(16) unsigned short Vbuf[8] ATTR_ALIGN(16);

	for (int height = 0; height < (int) dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDstY = pDstYImage;
		vx_uint8 * pLocalDstU = pDstUImage;
		vx_uint8 * pLocalDstV = pDstVImage;

		for (int width = 0; width < (alignedWidth >> 2); width++)
		{
			row0 = _mm_loadu_si128((__m128i*)(pLocalSrc));
			row1 = _mm_loadu_si128((__m128i*)(pLocalSrc + srcImageStrideInBytes));

			row0 = _mm_shuffle_epi8(row0, mask);
			row1 = _mm_shuffle_epi8(row1, mask);

			// R0..R3
			weights_toY = _mm_set_ps1(0.2126f);
			weights_toU = _mm_set_ps1(-0.1146f);
			weights_toV = _mm_set_ps1(0.5f);
			tempI = _mm_and_si128(row0, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			Y0 = _mm_mul_ps(temp, weights_toY);
			U0 = _mm_mul_ps(temp, weights_toU);
			V0 = _mm_mul_ps(temp, weights_toV);

			tempI = _mm_and_si128(row1, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			Y1 = _mm_mul_ps(temp, weights_toY);
			U1 = _mm_mul_ps(temp, weights_toU);
			V1 = _mm_mul_ps(temp, weights_toV);

			// G0..G3
			weights_toY = _mm_set_ps1(0.7152f);
			weights_toU = _mm_set_ps1(-0.3854f);
			weights_toV = _mm_set_ps1(-0.4542f);
			row0 = _mm_srli_si128(row0, 1);
			tempI = _mm_and_si128(row0, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y0 = _mm_add_ps(Y0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U0 = _mm_add_ps(U0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V0 = _mm_add_ps(V0, temp2);

			row1 = _mm_srli_si128(row1, 1);
			tempI = _mm_and_si128(row1, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y1 = _mm_add_ps(Y1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U1 = _mm_add_ps(U1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V1 = _mm_add_ps(V1, temp2);

			// B0..B3
			weights_toY = _mm_set_ps1(0.0722f);
			weights_toU = _mm_set_ps1(0.5f);
			weights_toV = _mm_set_ps1(-0.0458f);
			row0 = _mm_srli_si128(row0, 1);
			tempI = _mm_and_si128(row0, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y0 = _mm_add_ps(Y0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U0 = _mm_add_ps(U0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V0 = _mm_add_ps(V0, temp2);

			row1 = _mm_srli_si128(row1, 1);
			tempI = _mm_and_si128(row1, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y1 = _mm_add_ps(Y1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U1 = _mm_add_ps(U1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V1 = _mm_add_ps(V1, temp2);

			tempI = _mm_cvttps_epi32(Y0);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			row1 = _mm_cvttps_epi32(Y1);
			row1 = _mm_packus_epi32(row1, row1);
			row1 = _mm_packus_epi16(row1, row1);
			_mm_store_si128((__m128i *)Ybuf, tempI);
			_mm_store_si128((__m128i *)(Ybuf + 4), row1);

			// u00 u01 u02 u03
			// u10 u11 u12 u13
			U0 = _mm_add_ps(U0, const128);
			U1 = _mm_add_ps(U1, const128);
			tempI = _mm_cvttps_epi32(U0);
			tempI = _mm_packus_epi32(tempI, tempI);
			row1 = _mm_cvttps_epi32(U1);
			row1 = _mm_packus_epi32(row1, row1);
			tempI = _mm_avg_epu16(tempI, row1);			// Average u00, u10; u01, u11 ...
			//tempI = _mm_haddd_epu16(tempI);					// TBD: XOP instruction - not supported on all platforms
			tempI = _mm_hadd_epi16(tempI,tempI);				// Average horizontally
			tempI = _mm_cvtepi16_epi32(tempI);
			row0 = _mm_set1_epi16(1);
			tempI = _mm_add_epi16(tempI, row0);
			tempI = _mm_srli_epi16(tempI, 1);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			_mm_store_si128((__m128i *)Ubuf, tempI);

			// v00 v01 v02 v03
			// v10 v11 v12 v13
			V0 = _mm_add_ps(V0, const128);
			V1 = _mm_add_ps(V1, const128);
			tempI = _mm_cvttps_epi32(V0);
			tempI = _mm_packus_epi32(tempI, tempI);
			row1 = _mm_cvttps_epi32(V1);
			row1 = _mm_packus_epi32(row1, row1);
			tempI = _mm_avg_epu16(tempI, row1);			// Average u00, u10; u01, u11 ...
			//tempI = _mm_haddd_epu16(tempI);					// TBD: XOP instruction - not supported on all platforms
			tempI = _mm_hadd_epi16(tempI, tempI);				// Average horizontally
			tempI = _mm_cvtepi16_epi32(tempI);
			tempI = _mm_add_epi16(tempI, row0);
			tempI = _mm_srli_epi16(tempI, 1);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			_mm_store_si128((__m128i *)Vbuf, tempI);

			*(unsigned int *)(pLocalDstY) = Ybuf[0];
			*(unsigned int *)(pLocalDstY + dstYImageStrideInBytes) = Ybuf[4];
			*(unsigned short *)(pLocalDstU) = Ubuf[0];
			*(unsigned short *)(pLocalDstV) = Vbuf[0];

			pLocalSrc += 12;
			pLocalDstY += 4;
			pLocalDstU += 2;
			pLocalDstV += 2;
		}

		for (int width = 0; width < postfixWidth; width += 2)
		{
			float R = (float)*(pLocalSrc);
			float G = (float)*(pLocalSrc + 1);
			float B = (float)*(pLocalSrc + 2);

			*pLocalDstY = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
			float U = (R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f;
			float V = (R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f;

			R = (float)*(pLocalSrc + 3);
			G = (float)*(pLocalSrc + 4);
			B = (float)*(pLocalSrc + 5);

			*(pLocalDstY + 1) = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			R = (float)*(pLocalSrc + srcImageStrideInBytes);
			G = (float)*(pLocalSrc + srcImageStrideInBytes + 1);
			B = (float)*(pLocalSrc + srcImageStrideInBytes + 2);

			*(pLocalDstY + dstYImageStrideInBytes) = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			R = (float)*(pLocalSrc + srcImageStrideInBytes + 3);
			G = (float)*(pLocalSrc + srcImageStrideInBytes + 4);
			B = (float)*(pLocalSrc + srcImageStrideInBytes + 5);

			*(pLocalDstY + dstYImageStrideInBytes + 1) = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			U /= 4.0f;	V /= 4.0f;

			*pLocalDstU++ = (vx_uint8)U;
			*pLocalDstV++ = (vx_uint8)V;

			pLocalSrc += 6;
			pLocalDstY += 2;
		}

		pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);
		pDstYImage += (dstYImageStrideInBytes + dstYImageStrideInBytes);
		pDstUImage += dstUImageStrideInBytes;
		pDstVImage += dstVImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~3;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataColorConvert;
	__m128i mask = _mm_load_si128(tbl + 14);		// 0 B3 G3 R3 0 B2 G2 R2 0 B1 G1 R1 0 B0 G0 R0
	__m128i cvtmask = _mm_set1_epi32(255);			// 0 0 0 FF 0 0 0 FF 0 0 0 FF 0 0 0 FF
	__m128i row0, row1, tempI;
	__m128 Y0, U0, V0, Y1, U1, V1, weights_toY, weights_toU, weights_toV, temp, temp2;
	__m128 const128 = _mm_set1_ps(128.0f);

	DECL_ALIGN(16) unsigned int Ybuf[8] ATTR_ALIGN(16);
	DECL_ALIGN(16) unsigned short Ubuf[8] ATTR_ALIGN(16);
	DECL_ALIGN(16) unsigned short Vbuf[8] ATTR_ALIGN(16);

	for (int height = 0; height < (int)dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDstLuma = pDstLumaImage;
		vx_uint8 * pLocalDstChroma = pDstChromaImage;

		for (int width = 0; width < (alignedWidth >> 2); width++)
		{
			row0 = _mm_loadu_si128((__m128i*)(pLocalSrc));
			row1 = _mm_loadu_si128((__m128i*)(pLocalSrc + srcImageStrideInBytes));

			row0 = _mm_shuffle_epi8(row0, mask);
			row1 = _mm_shuffle_epi8(row1, mask);

			// R0..R3
			weights_toY = _mm_set_ps1(0.2126f);
			weights_toU = _mm_set_ps1(-0.1146f);
			weights_toV = _mm_set_ps1(0.5f);
			tempI = _mm_and_si128(row0, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			Y0 = _mm_mul_ps(temp, weights_toY);
			U0 = _mm_mul_ps(temp, weights_toU);
			V0 = _mm_mul_ps(temp, weights_toV);

			tempI = _mm_and_si128(row1, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			Y1 = _mm_mul_ps(temp, weights_toY);
			U1 = _mm_mul_ps(temp, weights_toU);
			V1 = _mm_mul_ps(temp, weights_toV);

			// G0..G3
			weights_toY = _mm_set_ps1(0.7152f);
			weights_toU = _mm_set_ps1(-0.3854f);
			weights_toV = _mm_set_ps1(-0.4542f);
			row0 = _mm_srli_si128(row0, 1);
			tempI = _mm_and_si128(row0, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y0 = _mm_add_ps(Y0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U0 = _mm_add_ps(U0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V0 = _mm_add_ps(V0, temp2);

			row1 = _mm_srli_si128(row1, 1);
			tempI = _mm_and_si128(row1, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y1 = _mm_add_ps(Y1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U1 = _mm_add_ps(U1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V1 = _mm_add_ps(V1, temp2);

			// B0..B3
			weights_toY = _mm_set_ps1(0.0722f);
			weights_toU = _mm_set_ps1(0.5f);
			weights_toV = _mm_set_ps1(-0.0458f);
			row0 = _mm_srli_si128(row0, 1);
			tempI = _mm_and_si128(row0, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y0 = _mm_add_ps(Y0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U0 = _mm_add_ps(U0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V0 = _mm_add_ps(V0, temp2);

			row1 = _mm_srli_si128(row1, 1);
			tempI = _mm_and_si128(row1, cvtmask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y1 = _mm_add_ps(Y1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U1 = _mm_add_ps(U1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V1 = _mm_add_ps(V1, temp2);

			tempI = _mm_cvttps_epi32(Y0);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			row1 = _mm_cvttps_epi32(Y1);
			row1 = _mm_packus_epi32(row1, row1);
			row1 = _mm_packus_epi16(row1, row1);
			_mm_store_si128((__m128i *)Ybuf, tempI);
			_mm_store_si128((__m128i *)(Ybuf + 4), row1);

			// u00 u01 u02 u03
			// u10 u11 u12 u13
			U0 = _mm_add_ps(U0, const128);
			U1 = _mm_add_ps(U1, const128);
			tempI = _mm_cvttps_epi32(U0);
			tempI = _mm_packus_epi32(tempI, tempI);
			row1 = _mm_cvttps_epi32(U1);
			row1 = _mm_packus_epi32(row1, row1);
			tempI = _mm_avg_epu16(tempI, row1);			// Average u00, u10; u01, u11 ...
			//tempI = _mm_haddd_epu16(tempI);					// TBD: XOP instruction - not supported on all platforms
			tempI = _mm_hadd_epi16(tempI, tempI);				// Average horizontally
			tempI = _mm_cvtepi16_epi32(tempI);
			row0 = _mm_set1_epi16(1);
			tempI = _mm_add_epi16(tempI, row0);
			tempI = _mm_srli_epi16(tempI, 1);
			tempI = _mm_packus_epi16(tempI, tempI);
			_mm_store_si128((__m128i *)Ubuf, tempI);

			// v00 v01 v02 v03
			// v10 v11 v12 v13
			V0 = _mm_add_ps(V0, const128);
			V1 = _mm_add_ps(V1, const128);
			tempI = _mm_cvttps_epi32(V0);
			tempI = _mm_packus_epi32(tempI, tempI);
			row1 = _mm_cvttps_epi32(V1);
			row1 = _mm_packus_epi32(row1, row1);
			tempI = _mm_avg_epu16(tempI, row1);			// Average u00, u10; u01, u11 ...
			//tempI = _mm_haddd_epu16(tempI);					// TBD: XOP instruction - not supported on all platforms
			tempI = _mm_hadd_epi16(tempI, tempI);				// Average horizontally
			tempI = _mm_cvtepi16_epi32(tempI);
			tempI = _mm_add_epi16(tempI, row0);
			tempI = _mm_srli_epi16(tempI, 1);
			tempI = _mm_packus_epi16(tempI, tempI);
			_mm_store_si128((__m128i *)Vbuf, tempI);

			*(unsigned int *)(pLocalDstLuma) = Ybuf[0];
			*(unsigned int *)(pLocalDstLuma + dstLumaImageStrideInBytes) = Ybuf[4];
			*(unsigned int *)(pLocalDstChroma) = Ubuf[0] | (Vbuf[0] << 8) | (Ubuf[1] << 16) | (Vbuf[1] << 24);

			pLocalSrc += 12;
			pLocalDstLuma += 4;
			pLocalDstChroma += 4;
		}

		for (int width = 0; width < postfixWidth; width += 2)
		{
			float R = (float)*(pLocalSrc);
			float G = (float)*(pLocalSrc + 1);
			float B = (float)*(pLocalSrc + 2);

			*pLocalDstLuma = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
			float U = (R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f;
			float V = (R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f;

			R = (float)*(pLocalSrc + 3);
			G = (float)*(pLocalSrc + 4);
			B = (float)*(pLocalSrc + 5);

			*(pLocalDstLuma + 1) = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			R = (float)*(pLocalSrc + srcImageStrideInBytes);
			G = (float)*(pLocalSrc + srcImageStrideInBytes + 1);
			B = (float)*(pLocalSrc + srcImageStrideInBytes + 2);

			*(pLocalDstLuma + dstLumaImageStrideInBytes) = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			R = (float)*(pLocalSrc + srcImageStrideInBytes + 3);
			G = (float)*(pLocalSrc + srcImageStrideInBytes + 4);
			B = (float)*(pLocalSrc + srcImageStrideInBytes + 5);

			*(pLocalDstLuma + dstLumaImageStrideInBytes + 1) = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			U /= 4.0f;	V /= 4.0f;

			*pLocalDstChroma++ = (vx_uint8)U;
			*pLocalDstChroma++ = (vx_uint8)V;

			pLocalSrc += 6;
			pLocalDstLuma += 2;
		}
		pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);
		pDstLumaImage += (dstLumaImageStrideInBytes + dstLumaImageStrideInBytes);
		pDstChromaImage += dstChromaImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_Y_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataColorConvert;

	__m128i pixels0, pixels1, pixels2, R, G, B;
	__m128i mask1 = _mm_load_si128(tbl + 21);
	__m128i mask2 = _mm_load_si128(tbl + 22);
	__m128i mask3 = _mm_load_si128(tbl + 23);
	__m128 weights_R = _mm_set_ps1((float) 0.2126);
	__m128 weights_G = _mm_set_ps1((float) 0.7152);
	__m128 weights_B = _mm_set_ps1((float) 0.0722);
	__m128 temp, Y;

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstYImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			pixels0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));

			R = _mm_shuffle_epi8(pixels0, mask1);						// 0 0 0 0 0 0 0 0 0 0 R5 R4 R3 R2 R1 R0
			G = _mm_shuffle_epi8(pixels0, mask3);						// 0 0 0 0 0 0 0 0 0 0 0 G4 G3 G2 G1 G0
			B = _mm_shuffle_epi8(pixels0, mask2);						// 0 0 0 0 0 0 0 0 0 0 0 B4 B3 B2 B1 B0

			pixels0 = _mm_shuffle_epi8(pixels1, mask2);					// 0 0 0 0 0 0 0 0 0 0 0 0 R10 R9 R8 R7 R6
			pixels0 = _mm_slli_si128(pixels0, 6);
			R = _mm_or_si128(R, pixels0);								// 0 0 0 0 0 R10 R9 R8 R7 R6 R5 R4 R3 R2 R1 R0
			pixels0 = _mm_shuffle_epi8(pixels1, mask1);					// 0 0 0 0 0 0 0 0 0 0 G10 G9 G8 G7 G6 G5
			pixels0 = _mm_slli_si128(pixels0, 5);
			G = _mm_or_si128(G, pixels0);								// 0 0 0 0 0 G10 G9 G8 G7 G6 G5 G4 G3 G2 G1 G0
			pixels0 = _mm_shuffle_epi8(pixels1, mask3);					// 0 0 0 0 0 0 0 0 0 0 0 B9 B8 B7 B6 B5
			pixels0 = _mm_slli_si128(pixels0, 5);
			B = _mm_or_si128(B, pixels0);								// 0 0 0 0 0 0 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0

			pixels0 = _mm_shuffle_epi8(pixels2, mask3);					// 0 0 0 0 0 0 0 0 0 0 0 R15 R14 R13 R12 R11
			pixels0 = _mm_slli_si128(pixels0, 11);
			R = _mm_or_si128(R, pixels0);								// R15 R14 R13 R12 R11 R10 R9 R8 R7 R6 R5 R4 R3 R2 R1 R0
			pixels0 = _mm_shuffle_epi8(pixels2, mask2);					// 0 0 0 0 0 0 0 0 0 0 0 G15 G14 G13 G12 G11
			pixels0 = _mm_slli_si128(pixels0, 11);
			G = _mm_or_si128(G, pixels0);								// G15 G14 G13 G12 G11 G10 G9 G8 G7 G6 G5 G4 G3 G2 G1 G0
			pixels0 = _mm_shuffle_epi8(pixels2, mask1);					// 0 0 0 0 0 0 0 0 0 0 B15 B14 B13 B12 B11 B10
			pixels0 = _mm_slli_si128(pixels0, 10);
			B = _mm_or_si128(B, pixels0);								// B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0

			// For pixels 0..3
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R0..R3
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G0..G3
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B0..B3
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_cvttps_epi32(Y);

			// For pixels 4..7
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R4..R7
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G4..G7
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B4..B7
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);
			pixels0 = _mm_packus_epi32(pixels0, pixels1);

			// For pixels 8..11
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R8..R11
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G8..G11
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B8..B11
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);

			// For pixels 12..15
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R12..R15
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G12..G15
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B12..B15
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvttps_epi32(Y);
			pixels1 = _mm_packus_epi32(pixels1, pixels2);

			pixels0 = _mm_packus_epi16(pixels0, pixels1);
			_mm_storeu_si128((__m128i *)pLocalDst, pixels0);

			pLocalSrc += 48;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			float R = (float)*pLocalSrc++;
			float G = (float)*pLocalSrc++;
			float B = (float)*pLocalSrc++;

			*pLocalDst++ = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722f));
		}

		pSrcImage += srcImageStrideInBytes;
		pDstYImage += dstYImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_U_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataColorConvert;

	__m128i pixels0, pixels1, pixels2, R, G, B;
	__m128i mask1 = _mm_load_si128(tbl + 21);
	__m128i mask2 = _mm_load_si128(tbl + 22);
	__m128i mask3 = _mm_load_si128(tbl + 23);
	__m128i offset = _mm_set1_epi32((int) 128);
	__m128 weights_R = _mm_set_ps1((float) -0.1146);
	__m128 weights_G = _mm_set_ps1((float) -0.3854);
	__m128 weights_B = _mm_set_ps1((float) 0.5);
	__m128 temp, Y;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstUImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			pixels0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));

			R = _mm_shuffle_epi8(pixels0, mask1);						// 0 0 0 0 0 0 0 0 0 0 R5 R4 R3 R2 R1 R0
			G = _mm_shuffle_epi8(pixels0, mask3);						// 0 0 0 0 0 0 0 0 0 0 0 G4 G3 G2 G1 G0
			B = _mm_shuffle_epi8(pixels0, mask2);						// 0 0 0 0 0 0 0 0 0 0 0 B4 B3 B2 B1 B0

			pixels0 = _mm_shuffle_epi8(pixels1, mask2);					// 0 0 0 0 0 0 0 0 0 0 0 0 R10 R9 R8 R7 R6
			pixels0 = _mm_slli_si128(pixels0, 6);
			R = _mm_or_si128(R, pixels0);								// 0 0 0 0 0 R10 R9 R8 R7 R6 R5 R4 R3 R2 R1 R0
			pixels0 = _mm_shuffle_epi8(pixels1, mask1);					// 0 0 0 0 0 0 0 0 0 0 G10 G9 G8 G7 G6 G5
			pixels0 = _mm_slli_si128(pixels0, 5);
			G = _mm_or_si128(G, pixels0);								// 0 0 0 0 0 G10 G9 G8 G7 G6 G5 G4 G3 G2 G1 G0
			pixels0 = _mm_shuffle_epi8(pixels1, mask3);					// 0 0 0 0 0 0 0 0 0 0 0 B9 B8 B7 B6 B5
			pixels0 = _mm_slli_si128(pixels0, 5);
			B = _mm_or_si128(B, pixels0);								// 0 0 0 0 0 0 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0

			pixels0 = _mm_shuffle_epi8(pixels2, mask3);					// 0 0 0 0 0 0 0 0 0 0 0 R15 R14 R13 R12 R11
			pixels0 = _mm_slli_si128(pixels0, 11);
			R = _mm_or_si128(R, pixels0);								// R15 R14 R13 R12 R11 R10 R9 R8 R7 R6 R5 R4 R3 R2 R1 R0
			pixels0 = _mm_shuffle_epi8(pixels2, mask2);					// 0 0 0 0 0 0 0 0 0 0 0 G15 G14 G13 G12 G11
			pixels0 = _mm_slli_si128(pixels0, 11);
			G = _mm_or_si128(G, pixels0);								// G15 G14 G13 G12 G11 G10 G9 G8 G7 G6 G5 G4 G3 G2 G1 G0
			pixels0 = _mm_shuffle_epi8(pixels2, mask1);					// 0 0 0 0 0 0 0 0 0 0 B15 B14 B13 B12 B11 B10
			pixels0 = _mm_slli_si128(pixels0, 10);
			B = _mm_or_si128(B, pixels0);								// B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0

			// For pixels 0..3
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R0..R3
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G0..G3
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B0..B3
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_cvttps_epi32(Y);
			pixels0 = _mm_add_epi32(pixels0, offset);

			// For pixels 4..7
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R4..R7
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G4..G7
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B4..B7
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);
			pixels1 = _mm_add_epi32(pixels1, offset);
			pixels0 = _mm_packus_epi32(pixels0, pixels1);
			
			// For pixels 8..11
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R8..R11
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G8..G11
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B8..B11
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);
			pixels1 = _mm_add_epi32(pixels1, offset);

			// For pixels 12..15
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R12..R15
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G12..G15
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B12..B15
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvttps_epi32(Y);
			pixels2 = _mm_add_epi32(pixels2, offset);
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			
			pixels0 = _mm_packus_epi16(pixels0, pixels1);
			_mm_storeu_si128((__m128i *)pLocalDst, pixels0);

			pLocalSrc += 48;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			float R = (float)*pLocalSrc++;
			float G = (float)*pLocalSrc++;
			float B = (float)*pLocalSrc++;

			*pLocalDst++ = (vx_uint8)((R * -0.1146f) + (G * -0.3854) + (B * 0.5f) + 128.0f);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstUImage += dstUImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_V_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataColorConvert;

	__m128i pixels0, pixels1, pixels2, R, G, B;
	__m128i mask1 = _mm_load_si128(tbl + 21);
	__m128i mask2 = _mm_load_si128(tbl + 22);
	__m128i mask3 = _mm_load_si128(tbl + 23);
	__m128i offset = _mm_set1_epi32((int)128);
	__m128 weights_R = _mm_set_ps1((float) 0.5);
	__m128 weights_G = _mm_set_ps1((float)-0.4542);
	__m128 weights_B = _mm_set_ps1((float)-0.0458);
	__m128 temp, Y;

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstVImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			pixels0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));

			R = _mm_shuffle_epi8(pixels0, mask1);						// 0 0 0 0 0 0 0 0 0 0 R5 R4 R3 R2 R1 R0
			G = _mm_shuffle_epi8(pixels0, mask3);						// 0 0 0 0 0 0 0 0 0 0 0 G4 G3 G2 G1 G0
			B = _mm_shuffle_epi8(pixels0, mask2);						// 0 0 0 0 0 0 0 0 0 0 0 B4 B3 B2 B1 B0

			pixels0 = _mm_shuffle_epi8(pixels1, mask2);					// 0 0 0 0 0 0 0 0 0 0 0 0 R10 R9 R8 R7 R6
			pixels0 = _mm_slli_si128(pixels0, 6);
			R = _mm_or_si128(R, pixels0);								// 0 0 0 0 0 R10 R9 R8 R7 R6 R5 R4 R3 R2 R1 R0
			pixels0 = _mm_shuffle_epi8(pixels1, mask1);					// 0 0 0 0 0 0 0 0 0 0 G10 G9 G8 G7 G6 G5
			pixels0 = _mm_slli_si128(pixels0, 5);
			G = _mm_or_si128(G, pixels0);								// 0 0 0 0 0 G10 G9 G8 G7 G6 G5 G4 G3 G2 G1 G0
			pixels0 = _mm_shuffle_epi8(pixels1, mask3);					// 0 0 0 0 0 0 0 0 0 0 0 B9 B8 B7 B6 B5
			pixels0 = _mm_slli_si128(pixels0, 5);
			B = _mm_or_si128(B, pixels0);								// 0 0 0 0 0 0 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0

			pixels0 = _mm_shuffle_epi8(pixels2, mask3);					// 0 0 0 0 0 0 0 0 0 0 0 R15 R14 R13 R12 R11
			pixels0 = _mm_slli_si128(pixels0, 11);
			R = _mm_or_si128(R, pixels0);								// R15 R14 R13 R12 R11 R10 R9 R8 R7 R6 R5 R4 R3 R2 R1 R0
			pixels0 = _mm_shuffle_epi8(pixels2, mask2);					// 0 0 0 0 0 0 0 0 0 0 0 G15 G14 G13 G12 G11
			pixels0 = _mm_slli_si128(pixels0, 11);
			G = _mm_or_si128(G, pixels0);								// G15 G14 G13 G12 G11 G10 G9 G8 G7 G6 G5 G4 G3 G2 G1 G0
			pixels0 = _mm_shuffle_epi8(pixels2, mask1);					// 0 0 0 0 0 0 0 0 0 0 B15 B14 B13 B12 B11 B10
			pixels0 = _mm_slli_si128(pixels0, 10);
			B = _mm_or_si128(B, pixels0);								// B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0

			// For pixels 0..3
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R0..R3
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G0..G3
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B0..B3
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_cvttps_epi32(Y);
			pixels0 = _mm_add_epi32(pixels0, offset);

			// For pixels 4..7
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R4..R7
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G4..G7
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B4..B7
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);
			pixels1 = _mm_add_epi32(pixels1, offset);
			pixels0 = _mm_packus_epi32(pixels0, pixels1);

			// For pixels 8..11
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R8..R11
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G8..G11
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B8..B11
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);
			pixels1 = _mm_add_epi32(pixels1, offset);

			// For pixels 12..15
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);
			pixels2 = _mm_cvtepu8_epi32(R);
			temp = _mm_cvtepi32_ps(pixels2);							// R12..R15
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_cvtepu8_epi32(G);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_G);							// G12..G15
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvtepu8_epi32(B);
			temp = _mm_cvtepi32_ps(pixels2);
			temp = _mm_mul_ps(temp, weights_B);							// B12..B15
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvttps_epi32(Y);
			pixels2 = _mm_add_epi32(pixels2, offset);
			pixels1 = _mm_packus_epi32(pixels1, pixels2);

			pixels0 = _mm_packus_epi16(pixels0, pixels1);
			_mm_storeu_si128((__m128i *)pLocalDst, pixels0);

			pLocalSrc += 48;
			pLocalDst += 16;
		}
		
		for (int width = 0; width < postfixWidth; width++)
		{
			float R = (float)*pLocalSrc++;
			float G = (float)*pLocalSrc++;
			float B = (float)*pLocalSrc++;

			*pLocalDst++ = (vx_uint8)((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstVImage += dstVImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_Y_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstYImage,
		vx_uint32     dstYImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;
	
	__m128i pixels0, pixels1, pixels2, pixels3, tempI;
	__m128i mask = _mm_set_epi8((char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF);
	__m128 weights_R = _mm_set_ps1((float) 0.2126);
	__m128 weights_G = _mm_set_ps1((float) 0.7152);
	__m128 weights_B = _mm_set_ps1((float) 0.0722);
	__m128 temp, Y;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstYImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			pixels0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));
			pixels3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 48));

			// For pixels 0..3
			tempI = _mm_and_si128(pixels0, mask);						// R0..R3
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels0 = _mm_srli_si128(pixels0, 1);
			tempI = _mm_and_si128(pixels0, mask);						// G0..G3
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_srli_si128(pixels0, 1);
			tempI = _mm_and_si128(pixels0, mask);						// B0..B3
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_cvttps_epi32(Y);

			// For pixels 4..7
			tempI = _mm_and_si128(pixels1, mask);						// R4..R7
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels1 = _mm_srli_si128(pixels1, 1);
			tempI = _mm_and_si128(pixels1, mask);						// G4..G7
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_srli_si128(pixels1, 1);
			tempI = _mm_and_si128(pixels1, mask);						// B4..B7
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);
			pixels0 = _mm_packus_epi32(pixels0, pixels1);

			// For pixels 8..11
			tempI = _mm_and_si128(pixels2, mask);						// R8..R11
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_srli_si128(pixels2, 1);
			tempI = _mm_and_si128(pixels2, mask);						// G8..G11
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_srli_si128(pixels2, 1);
			tempI = _mm_and_si128(pixels2, mask);						// B8..B11
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvttps_epi32(Y);

			// For pixels 12..15
			tempI = _mm_and_si128(pixels3, mask);						// R12..R15
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels3 = _mm_srli_si128(pixels3, 1);
			tempI = _mm_and_si128(pixels3, mask);						// G12..G15
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels3 = _mm_srli_si128(pixels3, 1);
			tempI = _mm_and_si128(pixels3, mask);						// B12..B15
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels3 = _mm_cvttps_epi32(Y);
			pixels1 = _mm_packus_epi32(pixels2, pixels3);

			pixels0 = _mm_packus_epi16(pixels0, pixels1);
			_mm_storeu_si128((__m128i *)pLocalDst, pixels0);

			pLocalSrc += 64;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			float R = (float)*pLocalSrc++;
			float G = (float)*pLocalSrc++;
			float B = (float)*pLocalSrc++;
			pLocalSrc++;

			*pLocalDst++ = (vx_uint8)((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722));
		}

		pSrcImage += srcImageStrideInBytes;
		pDstYImage += dstYImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_U_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstUImage,
		vx_uint32     dstUImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i pixels0, pixels1, pixels2, pixels3, tempI;
	__m128i mask = _mm_set_epi8((char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF);
	__m128i offset = _mm_set1_epi32((int)128);
	__m128 weights_R = _mm_set_ps1((float) -0.1146);
	__m128 weights_G = _mm_set_ps1((float) -0.3854);
	__m128 weights_B = _mm_set_ps1((float) 0.5);
	__m128 temp, Y;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstUImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			pixels0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));
			pixels3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 48));

			// For pixels 0..3
			tempI = _mm_and_si128(pixels0, mask);						// R0..R3
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels0 = _mm_srli_si128(pixels0, 1);
			tempI = _mm_and_si128(pixels0, mask);						// G0..G3
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_srli_si128(pixels0, 1);
			tempI = _mm_and_si128(pixels0, mask);						// B0..B3
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_cvttps_epi32(Y);
			pixels0 = _mm_add_epi32(pixels0, offset);

			// For pixels 4..7
			tempI = _mm_and_si128(pixels1, mask);						// R4..R7
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels1 = _mm_srli_si128(pixels1, 1);
			tempI = _mm_and_si128(pixels1, mask);						// G4..G7
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_srli_si128(pixels1, 1);
			tempI = _mm_and_si128(pixels1, mask);						// B4..B7
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);
			pixels1 = _mm_add_epi32(pixels1, offset);
			pixels0 = _mm_packus_epi32(pixels0, pixels1);
			
			// For pixels 8..11
			tempI = _mm_and_si128(pixels2, mask);						// R8..R11
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_srli_si128(pixels2, 1);
			tempI = _mm_and_si128(pixels2, mask);						// G8..G11
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_srli_si128(pixels2, 1);
			tempI = _mm_and_si128(pixels2, mask);						// B8..B11
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvttps_epi32(Y);
			pixels2 = _mm_add_epi32(pixels2, offset);

			// For pixels 12..15
			tempI = _mm_and_si128(pixels3, mask);						// R12..R15
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels3 = _mm_srli_si128(pixels3, 1);
			tempI = _mm_and_si128(pixels3, mask);						// G12..G15
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels3 = _mm_srli_si128(pixels3, 1);
			tempI = _mm_and_si128(pixels3, mask);						// B12..B15
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels3 = _mm_cvttps_epi32(Y);
			pixels3 = _mm_add_epi32(pixels3, offset);
			pixels1 = _mm_packus_epi32(pixels2, pixels3);
			
			pixels0 = _mm_packus_epi16(pixels0, pixels1);
			_mm_store_si128((__m128i *)pLocalDst, pixels0);

			pLocalSrc += 64;
			pLocalDst += 16;
		}
		
		for (int width = 0; width < postfixWidth; width++)
		{
			float R = (float)*pLocalSrc++;
			float G = (float)*pLocalSrc++;
			float B = (float)*pLocalSrc++;
			pLocalSrc++;

			*pLocalDst++ = (vx_uint8)((R * -0.1146f) + (G * -0.3854) + (B * 0.5f) + 128.0f);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstUImage += dstUImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorConvert_V_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstVImage,
		vx_uint32     dstVImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i pixels0, pixels1, pixels2, pixels3, tempI;
	__m128i mask = _mm_set_epi8((char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF);
	__m128i offset = _mm_set1_epi32((int)128);
	__m128 weights_R = _mm_set_ps1((float) 0.5);
	__m128 weights_G = _mm_set_ps1((float)-0.4542);
	__m128 weights_B = _mm_set_ps1((float)-0.0458);
	__m128 temp, Y;

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstVImage;

		for (int width = 0; width < (alignedWidth>>4); width++)
		{
			pixels0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));
			pixels3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 48));

			// For pixels 0..3
			tempI = _mm_and_si128(pixels0, mask);						// R0..R3
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels0 = _mm_srli_si128(pixels0, 1);
			tempI = _mm_and_si128(pixels0, mask);						// G0..G3
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_srli_si128(pixels0, 1);
			tempI = _mm_and_si128(pixels0, mask);						// B0..B3
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels0 = _mm_cvttps_epi32(Y);
			pixels0 = _mm_add_epi32(pixels0, offset);

			// For pixels 4..7
			tempI = _mm_and_si128(pixels1, mask);						// R4..R7
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels1 = _mm_srli_si128(pixels1, 1);
			tempI = _mm_and_si128(pixels1, mask);						// G4..G7
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_srli_si128(pixels1, 1);
			tempI = _mm_and_si128(pixels1, mask);						// B4..B7
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels1 = _mm_cvttps_epi32(Y);
			pixels1 = _mm_add_epi32(pixels1, offset);
			pixels0 = _mm_packus_epi32(pixels0, pixels1);

			// For pixels 8..11
			tempI = _mm_and_si128(pixels2, mask);						// R8..R11
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels2 = _mm_srli_si128(pixels2, 1);
			tempI = _mm_and_si128(pixels2, mask);						// G8..G11
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_srli_si128(pixels2, 1);
			tempI = _mm_and_si128(pixels2, mask);						// B8..B11
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels2 = _mm_cvttps_epi32(Y);
			pixels2 = _mm_add_epi32(pixels2, offset);

			// For pixels 12..15
			tempI = _mm_and_si128(pixels3, mask);						// R12..R15
			temp = _mm_cvtepi32_ps(tempI);
			Y = _mm_mul_ps(temp, weights_R);
			pixels3 = _mm_srli_si128(pixels3, 1);
			tempI = _mm_and_si128(pixels3, mask);						// G12..G15
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_G);
			Y = _mm_add_ps(Y, temp);
			pixels3 = _mm_srli_si128(pixels3, 1);
			tempI = _mm_and_si128(pixels3, mask);						// B12..B15
			temp = _mm_cvtepi32_ps(tempI);
			temp = _mm_mul_ps(temp, weights_B);
			Y = _mm_add_ps(Y, temp);
			pixels3 = _mm_cvttps_epi32(Y);
			pixels3 = _mm_add_epi32(pixels3, offset);
			pixels1 = _mm_packus_epi32(pixels2, pixels3);

			pixels0 = _mm_packus_epi16(pixels0, pixels1);
			_mm_store_si128((__m128i *)pLocalDst, pixels0);

			pLocalSrc += 64;
			pLocalDst += 16;
		}
		
		for (int width = 0; width < postfixWidth; width++)
		{
			float R = (float)*pLocalSrc++;
			float G = (float)*pLocalSrc++;
			float B = (float)*pLocalSrc++;
			pLocalSrc++;

			*pLocalDst++ = (vx_uint8)((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstVImage += dstVImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i pixels, tempI;
	__m128 Y, U, V, weights_toY, weights_toU, weights_toV, temp;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDstY = pDstYImage;
		vx_uint8 * pLocalDstU = pDstUImage;
		vx_uint8 * pLocalDstV = pDstVImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			__m128i Yout = _mm_setzero_si128();
			__m128i Uout = _mm_setzero_si128();
			__m128i Vout = _mm_setzero_si128();

			for (int i = 0; i < 4; i++)
			{
				pixels = _mm_loadu_si128((__m128i *) pLocalSrc);

				weights_toY = _mm_set_ps1(0.2126f);
				weights_toU = _mm_set_ps1(-0.1146f);
				weights_toV = _mm_set_ps1(0.5f);
				tempI = _mm_and_si128(pixels, _mm_set1_epi32((int)0x000000FF));						// R0..R3
				temp = _mm_cvtepi32_ps(tempI);
				Y = _mm_mul_ps(temp, weights_toY);
				U = _mm_mul_ps(temp, weights_toU);
				V = _mm_mul_ps(temp, weights_toV);

				weights_toY = _mm_set_ps1(0.7152f);
				weights_toU = _mm_set_ps1(-0.3854f);
				weights_toV = _mm_set_ps1(-0.4542f);
				pixels = _mm_srli_si128(pixels, 1);
				tempI = _mm_and_si128(pixels, _mm_set1_epi32((int)0x000000FF));						// G0..G3
				temp = _mm_cvtepi32_ps(tempI);
				weights_toY = _mm_mul_ps(temp, weights_toY);
				Y = _mm_add_ps(Y, weights_toY);
				weights_toY = _mm_mul_ps(temp, weights_toU);
				U = _mm_add_ps(U, weights_toY);
				weights_toY = _mm_mul_ps(temp, weights_toV);
				V = _mm_add_ps(V, weights_toY);

				weights_toY = _mm_set_ps1(0.0722f);
				weights_toU = _mm_set_ps1(0.5f);
				weights_toV = _mm_set_ps1(-0.0458f);
				pixels = _mm_srli_si128(pixels, 1);
				tempI = _mm_and_si128(pixels, _mm_set1_epi32((int)0x000000FF));						// B0..B3
				temp = _mm_cvtepi32_ps(tempI);
				weights_toY = _mm_mul_ps(temp, weights_toY);
				Y = _mm_add_ps(Y, weights_toY);
				weights_toY = _mm_mul_ps(temp, weights_toU);
				U = _mm_add_ps(U, weights_toY);
				weights_toY = _mm_mul_ps(temp, weights_toV);
				V = _mm_add_ps(V, weights_toY);

				tempI = _mm_cvtps_epi32(Y);
				tempI = _mm_packus_epi32(tempI, tempI);
				tempI = _mm_packus_epi16(tempI, tempI);
				tempI = _mm_and_si128(tempI, _mm_set_epi32((int)0xFFFFFFFF, 0, 0, 0));
				Yout = _mm_srli_si128(Yout, 4);
				Yout = _mm_or_si128(Yout, tempI);
				
				tempI = _mm_cvtps_epi32(U);
				tempI = _mm_add_epi32(tempI, _mm_set1_epi32((int)128));
				tempI = _mm_packus_epi32(tempI, tempI);
				tempI = _mm_packus_epi16(tempI, tempI);
				tempI = _mm_and_si128(tempI, _mm_set_epi32((int)0xFFFFFFFF, 0, 0, 0));
				Uout = _mm_srli_si128(Uout, 4);
				Uout = _mm_or_si128(Uout, tempI);
				
				tempI = _mm_cvtps_epi32(V);
				tempI = _mm_add_epi32(tempI, _mm_set1_epi32((int)128));
				tempI = _mm_packus_epi32(tempI, tempI);
				tempI = _mm_packus_epi16(tempI, tempI);
				tempI = _mm_and_si128(tempI, _mm_set_epi32((int)0xFFFFFFFF, 0, 0, 0));
				Vout = _mm_srli_si128(Vout, 4);
				Vout = _mm_or_si128(Vout, tempI);
								
				pixels = _mm_srli_si128(pixels, 1);
				pLocalSrc += 16;
			}
			
			_mm_storeu_si128((__m128i *) pLocalDstY, Yout);
			_mm_storeu_si128((__m128i *) pLocalDstU, Uout);
			_mm_storeu_si128((__m128i *) pLocalDstV, Vout);

			pLocalDstY += 16;
			pLocalDstU += 16;
			pLocalDstV += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			float R = (float)*pLocalSrc++;
			float G = (float)*pLocalSrc++;
			float B = (float)*pLocalSrc++;
			pLocalSrc++;

			*pLocalDstY++ = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			*pLocalDstU++ = (vx_uint8)(round((R * -0.1146f) + (G * -0.3854) + (B * 0.5f) + 128.0f));
			*pLocalDstV++ = (vx_uint8)(round((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f));
		}

		pSrcImage += srcImageStrideInBytes;
		pDstYImage += dstYImageStrideInBytes;
		pDstUImage += dstUImageStrideInBytes;
		pDstVImage += dstVImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~3;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i row0, row1, tempI;
	__m128i addToChroma = _mm_set1_epi32((int)128);
	__m128i mask = _mm_set_epi8((char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF);
	__m128 Y0, U0, V0, Y1, U1, V1, weights_toY, weights_toU, weights_toV, temp, temp2;

	DECL_ALIGN(16) unsigned int Ybuf[8] ATTR_ALIGN(16);
	DECL_ALIGN(16) unsigned short Ubuf[8] ATTR_ALIGN(16);
	DECL_ALIGN(16) unsigned short Vbuf[8] ATTR_ALIGN(16);

	for (int height = 0; height < (int) dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDstY = pDstYImage;
		vx_uint8 * pLocalDstU = pDstUImage;
		vx_uint8 * pLocalDstV = pDstVImage;

		for (int width = 0; width < (alignedWidth >> 2); width++)
		{
			row0 = _mm_load_si128((__m128i *)pLocalSrc);
			row1 = _mm_load_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes));

			// R0..R3
			weights_toY = _mm_set_ps1(0.2126f);
			weights_toU = _mm_set_ps1(-0.1146f);
			weights_toV = _mm_set_ps1(0.5f);
			tempI = _mm_and_si128(row0, mask);
			temp = _mm_cvtepi32_ps(tempI);
			Y0 = _mm_mul_ps(temp, weights_toY);
			U0 = _mm_mul_ps(temp, weights_toU);
			V0 = _mm_mul_ps(temp, weights_toV);

			tempI = _mm_and_si128(row1, mask);
			temp = _mm_cvtepi32_ps(tempI);
			Y1 = _mm_mul_ps(temp, weights_toY);
			U1 = _mm_mul_ps(temp, weights_toU);
			V1 = _mm_mul_ps(temp, weights_toV);

			// G0..G3
			weights_toY = _mm_set_ps1(0.7152f);
			weights_toU = _mm_set_ps1(-0.3854f);
			weights_toV = _mm_set_ps1(-0.4542f);
			row0 = _mm_srli_si128(row0, 1);
			tempI = _mm_and_si128(row0, mask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y0 = _mm_add_ps(Y0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U0 = _mm_add_ps(U0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V0 = _mm_add_ps(V0, temp2);

			row1 = _mm_srli_si128(row1, 1);
			tempI = _mm_and_si128(row1, mask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y1 = _mm_add_ps(Y1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U1 = _mm_add_ps(U1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V1 = _mm_add_ps(V1, temp2);

			// G0..G3
			weights_toY = _mm_set_ps1(0.0722f);
			weights_toU = _mm_set_ps1(0.5f);
			weights_toV = _mm_set_ps1(-0.0458f);
			row0 = _mm_srli_si128(row0, 1);
			tempI = _mm_and_si128(row0, mask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y0 = _mm_add_ps(Y0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U0 = _mm_add_ps(U0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V0 = _mm_add_ps(V0, temp2);

			row1 = _mm_srli_si128(row1, 1);
			tempI = _mm_and_si128(row1, mask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y1 = _mm_add_ps(Y1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U1 = _mm_add_ps(U1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V1 = _mm_add_ps(V1, temp2);

			tempI = _mm_cvtps_epi32(Y0);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			row1 = _mm_cvtps_epi32(Y1);
			row1 = _mm_packus_epi32(row1, row1);
			row1 = _mm_packus_epi16(row1, row1);
			_mm_store_si128((__m128i *)Ybuf, tempI);
			_mm_store_si128((__m128i *)(Ybuf + 4), row1);

			// u00 u01 u02 u03
			// u10 u11 u12 u13
			tempI = _mm_cvtps_epi32(U0);
			tempI = _mm_add_epi32(tempI, addToChroma);
			tempI = _mm_packus_epi32(tempI, tempI);
			row1 = _mm_cvtps_epi32(U1);
			row1 = _mm_add_epi32(row1, addToChroma);
			row1 = _mm_packus_epi32(row1, row1);
			tempI = _mm_avg_epu16(tempI, row1);			// Average u00, u10; u01, u11 ...
			//tempI = _mm_haddd_epu16(tempI);					// TBD: XOP instruction - not supported on all platforms
			tempI = _mm_hadd_epi16(tempI, tempI);				// Average horizontally
			tempI = _mm_cvtepi16_epi32(tempI);
			row0 = _mm_set1_epi32(0);
			tempI = _mm_add_epi32(tempI, row0);
			tempI = _mm_srli_epi32(tempI, 1);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			_mm_store_si128((__m128i *)Ubuf, tempI);

			// v00 v01 v02 v03
			// v10 v11 v12 v13
			tempI = _mm_cvtps_epi32(V0);
			tempI = _mm_add_epi32(tempI, addToChroma);
			tempI = _mm_packus_epi32(tempI, tempI);
			row1 = _mm_cvtps_epi32(V1);
			row1 = _mm_add_epi32(row1, addToChroma);
			row1 = _mm_packus_epi32(row1, row1);
			tempI = _mm_avg_epu16(tempI, row1);			// Average u00, u10; u01, u11 ...
			//tempI = _mm_haddd_epu16(tempI);					// TBD: XOP instruction - not supported on all platforms
			tempI = _mm_hadd_epi16(tempI, tempI);				// Average horizontally
			tempI = _mm_cvtepi16_epi32(tempI);
			tempI = _mm_add_epi32(tempI, row0);
			tempI = _mm_srli_epi32(tempI, 1);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			_mm_store_si128((__m128i *)Vbuf, tempI);

			*(unsigned int *)(pLocalDstY) = Ybuf[0];
			*(unsigned int *)(pLocalDstY + dstYImageStrideInBytes) = Ybuf[4];
			*(unsigned short *)(pLocalDstU) = Ubuf[0];
			*(unsigned short *)(pLocalDstV) = Vbuf[0];

			pLocalSrc += 16;
			pLocalDstY += 4;
			pLocalDstU += 2;
			pLocalDstV += 2;
		}

		for (int width = 0; width < postfixWidth; width += 2)
		{
			float R = (float)*(pLocalSrc);
			float G = (float)*(pLocalSrc + 1);
			float B = (float)*(pLocalSrc + 2);

			*pLocalDstY = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			float U = (R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f;
			float V = (R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f;

			R = (float)*(pLocalSrc + 4);
			G = (float)*(pLocalSrc + 5);
			B = (float)*(pLocalSrc + 6);

			*(pLocalDstY + 1) = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			R = (float)*(pLocalSrc + srcImageStrideInBytes);
			G = (float)*(pLocalSrc + srcImageStrideInBytes + 1);
			B = (float)*(pLocalSrc + srcImageStrideInBytes + 2);

			*(pLocalDstY + dstYImageStrideInBytes) = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			R = (float)*(pLocalSrc + srcImageStrideInBytes + 4);
			G = (float)*(pLocalSrc + srcImageStrideInBytes + 5);
			B = (float)*(pLocalSrc + srcImageStrideInBytes + 6);

			*(pLocalDstY + dstYImageStrideInBytes + 1) = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			U /= 4.0f;	V /= 4.0f;

			*pLocalDstU++ = (vx_uint8)(round(U));
			*pLocalDstV++ = (vx_uint8)(round(V));

			pLocalSrc += 8;
			pLocalDstY += 2;
		}

		pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);
		pDstYImage += (dstYImageStrideInBytes + dstYImageStrideInBytes);
		pDstUImage += dstUImageStrideInBytes;
		pDstVImage += dstVImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~3;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i row0, row1, tempI;
	__m128i addToChroma = _mm_set1_epi32((int)128);
	__m128i mask = _mm_set_epi8((char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF, (char)0, (char)0, (char)0, (char)0xFF);
	__m128 Y0, U0, V0, Y1, U1, V1, weights_toY, weights_toU, weights_toV, temp, temp2;

	DECL_ALIGN(16) unsigned int Ybuf[8] ATTR_ALIGN(16);
	DECL_ALIGN(16) unsigned char Ubuf[16] ATTR_ALIGN(16);
	DECL_ALIGN(16) unsigned char Vbuf[16] ATTR_ALIGN(16);

	for (int height = 0; height < (int) dstHeight; height += 2)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDstLuma = pDstLumaImage;
		vx_uint8 * pLocalDstChroma = pDstChromaImage;

		for (int width = 0; width < (alignedWidth >> 2); width++)
		{
			row0 = _mm_load_si128((__m128i *)pLocalSrc);
			row1 = _mm_load_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes));

			// R0..R3
			weights_toY = _mm_set_ps1(0.2126f);
			weights_toU = _mm_set_ps1(-0.1146f);
			weights_toV = _mm_set_ps1(0.5f);
			tempI = _mm_and_si128(row0, mask);
			temp = _mm_cvtepi32_ps(tempI);
			Y0 = _mm_mul_ps(temp, weights_toY);
			U0 = _mm_mul_ps(temp, weights_toU);
			V0 = _mm_mul_ps(temp, weights_toV);

			tempI = _mm_and_si128(row1, mask);
			temp = _mm_cvtepi32_ps(tempI);
			Y1 = _mm_mul_ps(temp, weights_toY);
			U1 = _mm_mul_ps(temp, weights_toU);
			V1 = _mm_mul_ps(temp, weights_toV);

			// G0..G3
			weights_toY = _mm_set_ps1(0.7152f);
			weights_toU = _mm_set_ps1(-0.3854f);
			weights_toV = _mm_set_ps1(-0.4542f);
			row0 = _mm_srli_si128(row0, 1);
			tempI = _mm_and_si128(row0, mask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y0 = _mm_add_ps(Y0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U0 = _mm_add_ps(U0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V0 = _mm_add_ps(V0, temp2);

			row1 = _mm_srli_si128(row1, 1);
			tempI = _mm_and_si128(row1, mask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y1 = _mm_add_ps(Y1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U1 = _mm_add_ps(U1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V1 = _mm_add_ps(V1, temp2);

			// G0..G3
			weights_toY = _mm_set_ps1(0.0722f);
			weights_toU = _mm_set_ps1(0.5f);
			weights_toV = _mm_set_ps1(-0.0458f);
			row0 = _mm_srli_si128(row0, 1);
			tempI = _mm_and_si128(row0, mask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y0 = _mm_add_ps(Y0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U0 = _mm_add_ps(U0, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V0 = _mm_add_ps(V0, temp2);

			row1 = _mm_srli_si128(row1, 1);
			tempI = _mm_and_si128(row1, mask);
			temp = _mm_cvtepi32_ps(tempI);
			temp2 = _mm_mul_ps(temp, weights_toY);
			Y1 = _mm_add_ps(Y1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toU);
			U1 = _mm_add_ps(U1, temp2);
			temp2 = _mm_mul_ps(temp, weights_toV);
			V1 = _mm_add_ps(V1, temp2);

			tempI = _mm_cvtps_epi32(Y0);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			row1 = _mm_cvtps_epi32(Y1);
			row1 = _mm_packus_epi32(row1, row1);
			row1 = _mm_packus_epi16(row1, row1);
			_mm_store_si128((__m128i *)Ybuf, tempI);
			_mm_store_si128((__m128i *)(Ybuf + 4), row1);

			// u00 u01 u02 u03
			// u10 u11 u12 u13
			tempI = _mm_cvtps_epi32(U0);
			tempI = _mm_add_epi32(tempI, addToChroma);
			tempI = _mm_packus_epi32(tempI, tempI);
			row1 = _mm_cvtps_epi32(U1);
			row1 = _mm_add_epi32(row1, addToChroma);
			row1 = _mm_packus_epi32(row1, row1);
			tempI = _mm_avg_epu16(tempI, row1);			// Average u00, u10; u01, u11 ...
			//tempI = _mm_haddd_epu16(tempI);					// TBD: XOP instruction - not supported on all platforms
			tempI = _mm_hadd_epi16(tempI, tempI);				// Average horizontally
			tempI = _mm_cvtepi16_epi32(tempI);
			row0 = _mm_set1_epi16(0);
			tempI = _mm_add_epi16(tempI, row0);
			tempI = _mm_srli_epi16(tempI, 1);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			_mm_store_si128((__m128i *)Ubuf, tempI);

			// v00 v01 v02 v03
			// v10 v11 v12 v13
			tempI = _mm_cvtps_epi32(V0);
			tempI = _mm_add_epi32(tempI, addToChroma);
			tempI = _mm_packus_epi32(tempI, tempI);
			row1 = _mm_cvtps_epi32(V1);
			row1 = _mm_add_epi32(row1, addToChroma);
			row1 = _mm_packus_epi32(row1, row1);
			tempI = _mm_avg_epu16(tempI, row1);			// Average u00, u10; u01, u11 ...
			//tempI = _mm_haddd_epu16(tempI);					// TBD: XOP instruction - not supported on all platforms
			tempI = _mm_hadd_epi16(tempI, tempI);				// Average horizontally
			tempI = _mm_cvtepi16_epi32(tempI);
			tempI = _mm_add_epi16(tempI, row0);
			tempI = _mm_srli_epi16(tempI, 1);
			tempI = _mm_packus_epi32(tempI, tempI);
			tempI = _mm_packus_epi16(tempI, tempI);
			_mm_store_si128((__m128i *)Vbuf, tempI);

			*(unsigned int *) (pLocalDstLuma) = Ybuf[0];
			*(unsigned int *)(pLocalDstLuma + dstLumaImageStrideInBytes) = Ybuf[4];
			*(unsigned int *) (pLocalDstChroma) = Ubuf[0] | (Vbuf[0] << 8) | (Ubuf[1] << 16) | (Vbuf[1] << 24);

			pLocalSrc += 16;
			pLocalDstLuma += 4;
			pLocalDstChroma += 4;
		}

		for (int width = 0; width < postfixWidth; width += 2)
		{
			float R = (float)*(pLocalSrc);
			float G = (float)*(pLocalSrc + 1);
			float B = (float)*(pLocalSrc + 2);

			*pLocalDstLuma = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			float U = (R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f;
			float V = (R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f;

			R = (float)*(pLocalSrc + 4);
			G = (float)*(pLocalSrc + 5);
			B = (float)*(pLocalSrc + 6);

			*(pLocalDstLuma + 1) = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			R = (float)*(pLocalSrc + srcImageStrideInBytes);
			G = (float)*(pLocalSrc + srcImageStrideInBytes + 1);
			B = (float)*(pLocalSrc + srcImageStrideInBytes + 2);

			*(pLocalDstLuma + dstLumaImageStrideInBytes) = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			R = (float)*(pLocalSrc + srcImageStrideInBytes + 4);
			G = (float)*(pLocalSrc + srcImageStrideInBytes + 5);
			B = (float)*(pLocalSrc + srcImageStrideInBytes + 6);

			*(pLocalDstLuma + dstLumaImageStrideInBytes + 1) = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			U += ((R * -0.1146f) + (G * -0.3854f) + (B * 0.5f) + 128.0f);
			V += ((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f);

			U /= 4.0f;	V /= 4.0f;

			*pLocalDstChroma++ = (vx_uint8)(round(U));
			*pLocalDstChroma++ = (vx_uint8)(round(V));

			pLocalSrc += 8;
			pLocalDstLuma += 2;
		}

		pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);
		pDstLumaImage += (dstLumaImageStrideInBytes + dstLumaImageStrideInBytes);
		pDstChromaImage += dstChromaImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataColorConvert;

	__m128i pixels0, pixels1, pixels2, R, G, B;
	__m128i addToChroma = _mm_set1_epi32((int)128);
	__m128i mask1 = _mm_load_si128(tbl + 21);
	__m128i mask2 = _mm_load_si128(tbl + 22);
	__m128i mask3 = _mm_load_si128(tbl + 23);
	__m128 weights_R2Y = _mm_set_ps1((float) 0.2126);
	__m128 weights_G2Y = _mm_set_ps1((float) 0.7152);
	__m128 weights_B2Y = _mm_set_ps1((float) 0.0722);
	__m128 weights_R2U = _mm_set_ps1((float) -0.1146);
	__m128 weights_G2U = _mm_set_ps1((float) -0.3854);
	__m128 weights_B2U = _mm_set_ps1((float) 0.5);
	__m128 weights_R2V = _mm_set_ps1((float) 0.5);
	__m128 weights_G2V = _mm_set_ps1((float) -0.4542);
	__m128 weights_B2V = _mm_set_ps1((float) -0.0458);
	__m128 temp0, temp1, Y, U, V;

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc =  pSrcImage;
		vx_uint8 * pLocalDstY = pDstYImage;
		vx_uint8 * pLocalDstU = pDstUImage;
		vx_uint8 * pLocalDstV = pDstVImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			pixels0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));

			R = _mm_shuffle_epi8(pixels0, mask1);						// 0 0 0 0 0 0 0 0 0 0 R5 R4 R3 R2 R1 R0
			G = _mm_shuffle_epi8(pixels0, mask3);						// 0 0 0 0 0 0 0 0 0 0 0 G4 G3 G2 G1 G0
			B = _mm_shuffle_epi8(pixels0, mask2);						// 0 0 0 0 0 0 0 0 0 0 0 B4 B3 B2 B1 B0

			pixels0 = _mm_shuffle_epi8(pixels1, mask2);					// 0 0 0 0 0 0 0 0 0 0 0 0 R10 R9 R8 R7 R6
			pixels0 = _mm_slli_si128(pixels0, 6);
			R = _mm_or_si128(R, pixels0);								// 0 0 0 0 0 R10 R9 R8 R7 R6 R5 R4 R3 R2 R1 R0
			pixels0 = _mm_shuffle_epi8(pixels1, mask1);					// 0 0 0 0 0 0 0 0 0 0 G10 G9 G8 G7 G6 G5
			pixels0 = _mm_slli_si128(pixels0, 5);
			G = _mm_or_si128(G, pixels0);								// 0 0 0 0 0 G10 G9 G8 G7 G6 G5 G4 G3 G2 G1 G0
			pixels0 = _mm_shuffle_epi8(pixels1, mask3);					// 0 0 0 0 0 0 0 0 0 0 0 B9 B8 B7 B6 B5
			pixels0 = _mm_slli_si128(pixels0, 5);
			B = _mm_or_si128(B, pixels0);								// 0 0 0 0 0 0 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0

			pixels0 = _mm_shuffle_epi8(pixels2, mask3);					// 0 0 0 0 0 0 0 0 0 0 0 R15 R14 R13 R12 R11
			pixels0 = _mm_slli_si128(pixels0, 11);
			R = _mm_or_si128(R, pixels0);								// R15 R14 R13 R12 R11 R10 R9 R8 R7 R6 R5 R4 R3 R2 R1 R0
			pixels0 = _mm_shuffle_epi8(pixels2, mask2);					// 0 0 0 0 0 0 0 0 0 0 0 G15 G14 G13 G12 G11
			pixels0 = _mm_slli_si128(pixels0, 11);
			G = _mm_or_si128(G, pixels0);								// G15 G14 G13 G12 G11 G10 G9 G8 G7 G6 G5 G4 G3 G2 G1 G0
			pixels0 = _mm_shuffle_epi8(pixels2, mask1);					// 0 0 0 0 0 0 0 0 0 0 B15 B14 B13 B12 B11 B10
			pixels0 = _mm_slli_si128(pixels0, 10);
			B = _mm_or_si128(B, pixels0);								// B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0

			// For pixels 0..3
			pixels2 = _mm_cvtepu8_epi32(R);
			temp0 = _mm_cvtepi32_ps(pixels2);							// R0..R3
			Y = _mm_mul_ps(temp0, weights_R2Y);
			U = _mm_mul_ps(temp0, weights_R2U);
			V = _mm_mul_ps(temp0, weights_R2V);
			
			pixels2 = _mm_cvtepu8_epi32(G);
			temp0 = _mm_cvtepi32_ps(pixels2);
			temp1 = _mm_mul_ps(temp0, weights_G2Y);						// G0..G3
			Y = _mm_add_ps(Y, temp1);
			temp1 = _mm_mul_ps(temp0, weights_G2U);
			U = _mm_add_ps(U, temp1);
			temp1 = _mm_mul_ps(temp0, weights_G2V);
			V = _mm_add_ps(V, temp1);

			pixels2 = _mm_cvtepu8_epi32(B);
			temp0 = _mm_cvtepi32_ps(pixels2);
			temp1 = _mm_mul_ps(temp0, weights_B2Y);						// B0..B3
			Y = _mm_add_ps(Y, temp1);
			temp1 = _mm_mul_ps(temp0, weights_B2U);
			U = _mm_add_ps(U, temp1);
			temp1 = _mm_mul_ps(temp0, weights_B2V);
			V = _mm_add_ps(V, temp1);

			__m128i tempI0 = _mm_cvtps_epi32(Y);
			__m128i tempI1 = _mm_cvtps_epi32(U);
			tempI1 = _mm_add_epi32(tempI1, addToChroma);
			__m128i tempI2 = _mm_cvtps_epi32(V);
			tempI2 = _mm_add_epi32(tempI2, addToChroma);

			// For pixels 4..7
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);

			pixels2 = _mm_cvtepu8_epi32(R);
			temp0 = _mm_cvtepi32_ps(pixels2);							// R4..R7
			Y = _mm_mul_ps(temp0, weights_R2Y);
			U = _mm_mul_ps(temp0, weights_R2U);
			V = _mm_mul_ps(temp0, weights_R2V);

			pixels2 = _mm_cvtepu8_epi32(G);
			temp0 = _mm_cvtepi32_ps(pixels2);
			temp1 = _mm_mul_ps(temp0, weights_G2Y);						// G4..G7
			Y = _mm_add_ps(Y, temp1);
			temp1 = _mm_mul_ps(temp0, weights_G2U);
			U = _mm_add_ps(U, temp1);
			temp1 = _mm_mul_ps(temp0, weights_G2V);
			V = _mm_add_ps(V, temp1);

			pixels2 = _mm_cvtepu8_epi32(B);
			temp0 = _mm_cvtepi32_ps(pixels2);
			temp1 = _mm_mul_ps(temp0, weights_B2Y);						// B4..B7
			Y = _mm_add_ps(Y, temp1);
			temp1 = _mm_mul_ps(temp0, weights_B2U);
			U = _mm_add_ps(U, temp1);
			temp1 = _mm_mul_ps(temp0, weights_B2V);
			V = _mm_add_ps(V, temp1);

			pixels1 = _mm_cvtps_epi32(Y);
			tempI0 = _mm_packus_epi32(tempI0, pixels1);
			pixels1 = _mm_cvtps_epi32(U);
			pixels1 = _mm_add_epi32(pixels1, addToChroma);
			tempI1 = _mm_packus_epi32(tempI1, pixels1);
			pixels1 = _mm_cvtps_epi32(V);
			pixels1 = _mm_add_epi32(pixels1, addToChroma);
			tempI2 = _mm_packus_epi32(tempI2, pixels1);

			// For pixels 8..11
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);

			pixels2 = _mm_cvtepu8_epi32(R);
			temp0 = _mm_cvtepi32_ps(pixels2);							// R8..R11
			Y = _mm_mul_ps(temp0, weights_R2Y);
			U = _mm_mul_ps(temp0, weights_R2U);
			V = _mm_mul_ps(temp0, weights_R2V);

			pixels2 = _mm_cvtepu8_epi32(G);
			temp0 = _mm_cvtepi32_ps(pixels2);
			temp1 = _mm_mul_ps(temp0, weights_G2Y);						// G8..G11
			Y = _mm_add_ps(Y, temp1);
			temp1 = _mm_mul_ps(temp0, weights_G2U);
			U = _mm_add_ps(U, temp1);
			temp1 = _mm_mul_ps(temp0, weights_G2V);
			V = _mm_add_ps(V, temp1);

			pixels2 = _mm_cvtepu8_epi32(B);
			temp0 = _mm_cvtepi32_ps(pixels2);
			temp1 = _mm_mul_ps(temp0, weights_B2Y);						// B8..B11
			Y = _mm_add_ps(Y, temp1);
			temp1 = _mm_mul_ps(temp0, weights_B2U);
			U = _mm_add_ps(U, temp1);
			temp1 = _mm_mul_ps(temp0, weights_B2V);
			V = _mm_add_ps(V, temp1);

			pixels0 = _mm_cvtps_epi32(Y);
			pixels1 = _mm_cvtps_epi32(U);
			pixels1 = _mm_add_epi32(pixels1, addToChroma);
			pixels2 = _mm_cvtps_epi32(V);
			pixels2 = _mm_add_epi32(pixels2, addToChroma);

			// For pixels 12..15
			R = _mm_srli_si128(R, 4);
			G = _mm_srli_si128(G, 4);
			B = _mm_srli_si128(B, 4);

			R = _mm_cvtepu8_epi32(R);
			temp0 = _mm_cvtepi32_ps(R);									// R12..R15
			Y = _mm_mul_ps(temp0, weights_R2Y);
			U = _mm_mul_ps(temp0, weights_R2U);
			V = _mm_mul_ps(temp0, weights_R2V);

			G = _mm_cvtepu8_epi32(G);
			temp0 = _mm_cvtepi32_ps(G);
			temp1 = _mm_mul_ps(temp0, weights_G2Y);						// G12..G15
			Y = _mm_add_ps(Y, temp1);
			temp1 = _mm_mul_ps(temp0, weights_G2U);
			U = _mm_add_ps(U, temp1);
			temp1 = _mm_mul_ps(temp0, weights_G2V);
			V = _mm_add_ps(V, temp1);

			B = _mm_cvtepu8_epi32(B);
			temp0 = _mm_cvtepi32_ps(B);
			temp1 = _mm_mul_ps(temp0, weights_B2Y);						// B12..B15
			Y = _mm_add_ps(Y, temp1);
			temp1 = _mm_mul_ps(temp0, weights_B2U);
			U = _mm_add_ps(U, temp1);
			temp1 = _mm_mul_ps(temp0, weights_B2V);
			V = _mm_add_ps(V, temp1);

			R = _mm_cvtps_epi32(Y);
			pixels0 = _mm_packus_epi32(pixels0, R);
			G = _mm_cvtps_epi32(U);
			G = _mm_add_epi32(G, addToChroma);
			pixels1 = _mm_packus_epi32(pixels1, G);
			B = _mm_cvtps_epi32(V);
			B = _mm_add_epi32(B, addToChroma);
			pixels2 = _mm_packus_epi32(pixels2, B);

			tempI0 = _mm_packus_epi16(tempI0, pixels0);
			tempI1 = _mm_packus_epi16(tempI1, pixels1);
			tempI2 = _mm_packus_epi16(tempI2, pixels2);

			_mm_storeu_si128((__m128i *)pLocalDstY, tempI0);
			_mm_storeu_si128((__m128i *)pLocalDstU, tempI1);
			_mm_storeu_si128((__m128i *)pLocalDstV, tempI2);

			pLocalSrc += 48;
			pLocalDstY += 16;
			pLocalDstU += 16;
			pLocalDstV += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			float R = (float)*pLocalSrc++;
			float G = (float)*pLocalSrc++;
			float B = (float)*pLocalSrc++;

			*pLocalDstY++ = (vx_uint8)(round((R * 0.2126f) + (G * 0.7152f) + (B * 0.0722)));
			*pLocalDstU++ = (vx_uint8)(round((R * -0.1146f) + (G * -0.3854) + (B * 0.5f) + 128.0f));
			*pLocalDstV++ = (vx_uint8)(round((R * 0.5f) + (G * -0.4542f) + (B * -0.0458f) + 128.0f));
		}

		pSrcImage += srcImageStrideInBytes;
		pDstYImage += dstYImageStrideInBytes;
		pDstUImage += dstUImageStrideInBytes;
		pDstVImage += dstVImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i * tbl = (__m128i *) dataColorConvert;
	
	bool isAligned = ((intptr_t(pDstUImage) & intptr_t(pDstVImage) & 15) == ((intptr_t(pDstUImage) | intptr_t(pDstVImage)) & 15));

	unsigned char *pLocalSrc, *pLocalDstU, *pLocalDstV;
	__m128i *pLocalSrc_xmm, *pLocalDstU_xmm, *pLocalDstV_xmm;

	__m128i pixels0, pixels1, U, temp;
	__m128i mask_UV12ToIUV_Ulow = _mm_load_si128(tbl + 3);
	__m128i mask_UV12ToIUV_Uhi  = _mm_load_si128(tbl + 25);
	__m128i mask_UV12ToIUV_Vlow = _mm_load_si128(tbl + 0);
	__m128i mask_UV12ToIUV_Vhi  = _mm_load_si128(tbl + 24);

	if (isAligned)
	{
		int prefixWidth = intptr_t(pDstUImage) & 15;
		prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
		int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
		int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcChromaImage;
			pLocalDstU = (unsigned char *)pDstUImage;
			pLocalDstV = (unsigned char *)pDstVImage;

			for (int x = 0; x < prefixWidth; x++)
			{
				*pLocalDstU++ = *pLocalSrc++;
				*pLocalDstV++ = *pLocalSrc++;
			}

			pLocalSrc_xmm = (__m128i *) pLocalSrc;
			pLocalDstU_xmm = (__m128i *) pLocalDstU;
			pLocalDstV_xmm = (__m128i *) pLocalDstV;

			int width = (int)(alignedWidth >> 4);								// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels1 = _mm_loadu_si128(pLocalSrc_xmm++);

				U = _mm_shuffle_epi8(pixels0, mask_UV12ToIUV_Ulow);
				temp = _mm_shuffle_epi8(pixels1, mask_UV12ToIUV_Uhi);
				U = _mm_or_si128(U, temp);

				pixels0 = _mm_shuffle_epi8(pixels0, mask_UV12ToIUV_Vlow);
				temp = _mm_shuffle_epi8(pixels1, mask_UV12ToIUV_Vhi);
				pixels0 = _mm_or_si128(pixels0, temp);

				_mm_store_si128(pLocalDstU_xmm++, U);
				_mm_store_si128(pLocalDstV_xmm++, pixels0);

				width--;
			}

			pLocalSrc = (unsigned char *) pLocalSrc_xmm;
			pLocalDstU = (unsigned char *) pLocalDstU_xmm;
			pLocalDstV = (unsigned char *) pLocalDstV_xmm;
			for (int x = 0; x < postfixWidth; x++)
			{
				*pLocalDstU++ = *pLocalSrc++;
				*pLocalDstV++ = *pLocalSrc++;
			}

			pSrcChromaImage += srcChromaImageStrideInBytes;
			pDstUImage += dstUImageStrideInBytes;
			pDstVImage += dstVImageStrideInBytes;
			height--;
		}
	}
	else
	{
		int postfixWidth = (int)dstWidth & 15;					// 16 pixels processed at a time in SSE loop
		int alignedWidth = (int)dstWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc_xmm = (__m128i *) pSrcChromaImage;
			pLocalDstU_xmm = (__m128i *) pDstUImage;
			pLocalDstV_xmm = (__m128i *) pDstVImage;

			int width = (int)(alignedWidth >> 4);								// 16 pixels processed at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels1 = _mm_loadu_si128(pLocalSrc_xmm++);

				U = _mm_shuffle_epi8(pixels0, mask_UV12ToIUV_Ulow);
				temp = _mm_shuffle_epi8(pixels1, mask_UV12ToIUV_Uhi);
				U = _mm_or_si128(U, temp);

				pixels0 = _mm_shuffle_epi8(pixels0, mask_UV12ToIUV_Vlow);
				temp = _mm_shuffle_epi8(pixels1, mask_UV12ToIUV_Vhi);
				pixels0 = _mm_or_si128(pixels0, temp);

				_mm_storeu_si128(pLocalDstU_xmm++, U);
				_mm_storeu_si128(pLocalDstV_xmm++, pixels0);

				width--;
			}

			pLocalSrc = (unsigned char *)pLocalSrc_xmm;
			pLocalDstU = (unsigned char *)pLocalDstU_xmm;
			pLocalDstV = (unsigned char *)pLocalDstV_xmm;
			for (int x = 0; x < postfixWidth; x++)
			{
				*pLocalDstU++ = *pLocalSrc++;
				*pLocalDstV++ = *pLocalSrc++;
			}

			pSrcChromaImage += srcChromaImageStrideInBytes;
			pDstUImage += dstUImageStrideInBytes;
			pDstVImage += dstVImageStrideInBytes;
			height--;
		}
	}
	return AGO_SUCCESS;
}

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
	)
{
	int prefixWidth = intptr_t(pDstChromaImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	unsigned char *pLocalSrcU, *pLocalSrcV, *pLocalDst;
	__m128i *pLocalSrcU_xmm, *pLocalSrcV_xmm, *pLocalDst_xmm;
	__m128i pixels_U, pixels_V, pixels_out;

	int height = (int) dstHeight;

	while (height)
	{
		pLocalSrcU = (unsigned char *) pSrcUImage;
		pLocalSrcV = (unsigned char *) pSrcVImage;
		pLocalDst = (unsigned char *) pDstChromaImage;

		for (int x = 0; x < prefixWidth; x++)
		{
			*pLocalDst++ = *pLocalSrcU++;
			*pLocalDst++ = *pLocalSrcV++;
		}

		pLocalSrcU_xmm = (__m128i *) pLocalSrcU;
		pLocalSrcV_xmm = (__m128i *) pLocalSrcV;
		pLocalDst_xmm = (__m128i *) pLocalDst;

		int width = (int) (dstWidth >> 4);									// Each inner loop writes 16 pixels of each chroma plane in destination buffer
		while (width)
		{
			pixels_U = _mm_loadu_si128(pLocalSrcU_xmm++);
			pixels_V = _mm_loadu_si128(pLocalSrcV_xmm++);
			pixels_out = _mm_unpacklo_epi8(pixels_U, pixels_V);
			pixels_U = _mm_unpackhi_epi8(pixels_U, pixels_V);

			_mm_store_si128(pLocalDst_xmm++, pixels_out);
			_mm_store_si128(pLocalDst_xmm++, pixels_U);
			width--;
		}

		pLocalSrcU = (unsigned char *) pLocalSrcU_xmm;
		pLocalSrcV = (unsigned char *) pLocalSrcV_xmm;
		pLocalDst = (unsigned char *) pLocalDst_xmm;
		for (int x = 0; x < postfixWidth; x++)
		{
			*pLocalDst++ = *pLocalSrcU++;
			*pLocalDst++ = *pLocalSrcV++;
		}

		pSrcUImage += srcUImageStrideInBytes;
		pSrcVImage += srcVImageStrideInBytes;
		pDstChromaImage += dstChromaImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i *) dataColorConvert;
	vx_uint8 *pLocalSrc, *pLocalDstUCurrentRow, *pLocalDstUNextRow, *pLocalDstVCurrentRow, *pLocalDstVNextRow;
	__m128i pixels, U;
	__m128i maskU = _mm_load_si128(tbl + 6);
	__m128i maskV = _mm_load_si128(tbl + 7);

	int height = (int) (dstHeight >> 1);				// Each inner loop writes out two rows of dst buffer
	while (height)
	{
		pLocalSrc = pSrcChromaImage;
		pLocalDstUCurrentRow = pDstUImage;
		pLocalDstUNextRow = pDstUImage + dstUImageStrideInBytes;
		pLocalDstVCurrentRow = pDstVImage;
		pLocalDstVNextRow = pDstVImage + dstVImageStrideInBytes;

		int width = (int) (alignedWidth >> 4);				// Each inner loop iteration processess 16 output pixels
		while (width)
		{
			pixels = _mm_loadu_si128((__m128i*) pLocalSrc);
			U = _mm_shuffle_epi8(pixels, maskU);
			pixels = _mm_shuffle_epi8(pixels, maskV);

			_mm_storeu_si128((__m128i*) pLocalDstUCurrentRow, U);
			_mm_storeu_si128((__m128i*) pLocalDstUNextRow, U);
			_mm_storeu_si128((__m128i*) pLocalDstVCurrentRow, pixels);
			_mm_storeu_si128((__m128i*) pLocalDstVNextRow, pixels);

			pLocalSrc += 16;
			pLocalDstUCurrentRow += 16;
			pLocalDstUNextRow += 16;
			pLocalDstVCurrentRow += 16;
			pLocalDstVNextRow += 16;
			width--;
		}

		for (int w = 0; w < postfixWidth; w += 2)
		{
			*pLocalDstUCurrentRow++ = *pLocalSrc;
			*pLocalDstUCurrentRow++ = *pLocalSrc;
			*pLocalDstUNextRow++ = *pLocalSrc;
			*pLocalDstUNextRow++ = *pLocalSrc++;

			*pLocalDstVCurrentRow++ = *pLocalSrc;
			*pLocalDstVCurrentRow++ = *pLocalSrc;
			*pLocalDstVNextRow++ = *pLocalSrc;
			*pLocalDstVNextRow++ = *pLocalSrc++;
		}

		pSrcChromaImage += srcChromaImageStrideInBytes;
		pDstUImage += (dstUImageStrideInBytes + dstUImageStrideInBytes);
		pDstVImage += (dstVImageStrideInBytes + dstVImageStrideInBytes);
		height--;
	}
	return AGO_SUCCESS;
}
