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

static inline unsigned short Horizontal5x5GaussianFilter_C(unsigned char * srcImage)
{
	return((unsigned short)srcImage[-2] + 4 * (unsigned short)srcImage[-1] + 6 * (unsigned short)srcImage[0] + 4 * (unsigned short)srcImage[1] + (unsigned short)srcImage[2]);
}

static inline __m128i Horizontal3x3GaussianFilter_SampleFirstPixel_SSE(unsigned char * srcImage)
{
	__m128i shiftedL2, shiftedL1, row, shiftedR2, shiftedR1;
	__m128i resultH, resultL;
	__m128i zeromask = _mm_setzero_si128();
	__m128i mask = _mm_set1_epi32((int)0x0000FFFF);

	shiftedL2 = _mm_loadu_si128((__m128i *) (srcImage - 2));						// -2
	shiftedR2 = _mm_loadu_si128((__m128i *) (srcImage + 2));						// +2

	resultH = _mm_unpackhi_epi8(shiftedL2, zeromask);								// r[-2]
	resultL = _mm_cvtepu8_epi16(shiftedL2);											// r[-2]
	shiftedL2 = _mm_unpackhi_epi8(shiftedR2, zeromask);								// r[2]
	shiftedR2 = _mm_cvtepu8_epi16(shiftedR2);										// r[2]

	shiftedL1 = _mm_loadu_si128((__m128i *) (srcImage - 1));						// -1

	resultH = _mm_add_epi16(resultH, shiftedL2);									// r[-2] + r[2]
	resultL = _mm_add_epi16(resultL, shiftedR2);									// r[-2] + r[2]

	shiftedR1 = _mm_loadu_si128((__m128i *) (srcImage + 1));						// +1

	shiftedL2 = _mm_unpackhi_epi8(shiftedL1, zeromask);								// r[-1]
	shiftedL1 = _mm_cvtepu8_epi16(shiftedL1);										// r[-1]

	row = _mm_loadu_si128((__m128i *) srcImage);									// 0

	shiftedR2 = _mm_unpackhi_epi8(shiftedR1, zeromask);								// r[+1]
	shiftedR1 = _mm_cvtepu8_epi16(shiftedR1);										// r[+1]

	shiftedL2 = _mm_add_epi16(shiftedL2, shiftedR2);								// r[-1] + r[1]
	shiftedL1 = _mm_add_epi16(shiftedL1, shiftedR1);								// r[-1] + r[1]

	shiftedR1 = _mm_unpackhi_epi8(row, zeromask);									// r[0]
	row = _mm_cvtepu8_epi16(row);													// r[0]

	shiftedL2 = _mm_add_epi16(shiftedL2, shiftedR1);								// r[-1] + r[0] + r[1]
	shiftedL1 = _mm_add_epi16(shiftedL1, row);										// r[-1] + r[0] + r[1]
	shiftedL2 = _mm_slli_epi16(shiftedL2, 2);										// 4*r[-1] + 4*r[0] + 4*r[1]
	shiftedL1 = _mm_slli_epi16(shiftedL1, 2);										// 4*r[-1] + 4*r[0] + 4*r[1]

	shiftedR1 = _mm_slli_epi16(shiftedR1, 1);										// 2*r[0]
	row = _mm_slli_epi16(row, 1);													// 2*r[0]
	shiftedL2 = _mm_add_epi16(shiftedL2, shiftedR1);								// 4*r[-1] + 6*r[0] + 4*r[1]
	shiftedL1 = _mm_add_epi16(shiftedL1, row);										// 4*r[-1] + 6*r[0] + 4*r[1]

	resultH = _mm_add_epi16(resultH, shiftedL2);									// r[-2] + 4*r[-1] + 6*r[0] + 4*r[1] + r[2]
	resultL = _mm_add_epi16(resultL, shiftedL1);									// r[-2] + 4*r[-1] + 6*r[0] + 4*r[1] + r[2]

	resultH = _mm_and_si128(resultH, mask);											// Select words: 0, 2, 4, 6
	resultL = _mm_and_si128(resultL, mask);											// Select words: 0, 2, 4, 6

	resultL = _mm_packus_epi32(resultL, resultH);

	return(resultL);
}

/* Kernel			1   4   6   4   1			1		1   4   6   4   1
				    4  16  24  16   4			4
			1/256	6  24  36  24   6    =		6									>> 8				
					4  16  24  16   4			4
					1   4   6   4   1			1
*/
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
	)
{
	int alignedDstStride = (dstImageStrideInBytes + 15) & ~15;
	alignedDstStride <<= 2;				// Each row stores two short values (Gx,Gy) for each pixel
	unsigned short * r0 = (unsigned short *)pScratch;
	unsigned short * r1 = (unsigned short *)(pScratch + alignedDstStride);
	unsigned short * r2 = (unsigned short *)(pScratch + 2 * alignedDstStride);
	unsigned short * r3 = (unsigned short *)(pScratch + 3 * alignedDstStride);
	unsigned short * r4 = (unsigned short *)(pScratch + 4 * alignedDstStride);

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;
	int srcRowOffset = sampleFirstRow ? 0 : srcImageStrideInBytes;
	int srcColOffset = sampleFirstColumn ? 0 : 1;

	pSrcImage += srcRowOffset;																	// Offset for odd/even row sampling
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	unsigned char *pLocalDst = (unsigned char *)pDstImage;

	unsigned short * pRowMinus2 = r0;
	unsigned short * pRowMinus1 = r1;
	unsigned short * pRowCurr = r2;
	unsigned short * pRowPlus1 = r3;
	unsigned short * pRowPlus2 = r4;

	__m128i temp0, temp1, temp2, temp3, pixels_plus1H, pixels_plus1L, pixels_plus2H, pixels_plus2L;

	unsigned short * pLocalRowMinus2 = pRowMinus2;
	unsigned short * pLocalRowMinus1 = pRowMinus1;
	unsigned short * pLocalRowCurr = pRowCurr;
	unsigned short * pLocalRowPlus1 = pRowPlus1;
	unsigned short * pLocalRowPlus2 = pRowPlus2;
	unsigned short * pTemp0, *pTemp1;

	int srcStride = (int)srcImageStrideInBytes;
	pLocalSrc += srcColOffset;

	// Process first three rows - Horizontal filtering
	for (int x = 0; x < (prefixWidth << 1); x++, pLocalSrc += 2)
	{
		*pLocalRowMinus2++ = Horizontal5x5GaussianFilter_C(pLocalSrc - (srcStride + srcStride));
		*pLocalRowMinus1++ = Horizontal5x5GaussianFilter_C(pLocalSrc - srcStride);
		*pLocalRowCurr++ = Horizontal5x5GaussianFilter_C(pLocalSrc);
	}

	for (int x = 0; x < (alignedWidth >> 3); x++)
	{
		__m128i temp0, temp1;

		temp0 = Horizontal3x3GaussianFilter_SampleFirstPixel_SSE(pLocalSrc - (srcStride + srcStride));
		_mm_storeu_si128((__m128i *)pLocalRowMinus2, temp0);

		temp1 = Horizontal3x3GaussianFilter_SampleFirstPixel_SSE(pLocalSrc - srcStride);
		_mm_storeu_si128((__m128i *)pLocalRowMinus1, temp1);

		temp0 = Horizontal3x3GaussianFilter_SampleFirstPixel_SSE(pLocalSrc);
		_mm_storeu_si128((__m128i *)pLocalRowCurr, temp0);

		pLocalSrc += 16;
		pLocalRowMinus2 += 8;
		pLocalRowMinus1 += 8;
		pLocalRowCurr += 8;
	}

	for (int x = 0; x < (postfixWidth << 1); x++, pLocalSrc += 2)
	{
		*pLocalRowMinus2++ = Horizontal5x5GaussianFilter_C(pLocalSrc - (srcStride + srcStride));
		*pLocalRowMinus1++ = Horizontal5x5GaussianFilter_C(pLocalSrc - srcStride);
		*pLocalRowCurr++ = Horizontal5x5GaussianFilter_C(pLocalSrc);
	}

	pLocalRowMinus2 = pRowMinus2;
	pLocalRowMinus1 = pRowMinus1;
	pLocalRowCurr = pRowCurr;

	// Process rows 4 till the end
	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcStride + srcColOffset);			// Pointing to the row below
		unsigned char * pLocalSrc_NextRow = pLocalSrc + srcStride;
		pLocalDst = (unsigned char *)pDstImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc += 2)
		{
			short temp_plus1 = Horizontal5x5GaussianFilter_C(pLocalSrc);				// row + 1
			*pLocalRowPlus1++ = temp_plus1;
			short temp_plus2 = Horizontal5x5GaussianFilter_C(pLocalSrc_NextRow);		// row + 2
			*pLocalRowPlus2++ = temp_plus2;

			*pLocalDst++ = (unsigned char)((*pLocalRowMinus2++ + 4 * (*pLocalRowMinus1++) + 6 * (*pLocalRowCurr++) + 4 * temp_plus1 + temp_plus2) >> 8);
		}

		int width = (int)(alignedWidth >> 4);															// 16 dst pixels processed in one go
		while (width)
		{
			temp0 = _mm_loadu_si128((__m128i *) pLocalRowCurr);											// c[0]
			temp1 = _mm_loadu_si128((__m128i *) (pLocalRowCurr + 8));									// c[0]

			pixels_plus1L = Horizontal3x3GaussianFilter_SampleFirstPixel_SSE(pLocalSrc);				// Horizontal filtering	- c[1]
			_mm_storeu_si128((__m128i *)pLocalRowPlus1, pixels_plus1L);
			pixels_plus1H = Horizontal3x3GaussianFilter_SampleFirstPixel_SSE(pLocalSrc + 16);			// Horizontal filtering	- c[1]
			_mm_storeu_si128((__m128i *)(pLocalRowPlus1 + 8), pixels_plus1H);

			pixels_plus1H = _mm_add_epi16(pixels_plus1H, temp1);										// c[0] + c[1]
			pixels_plus1L = _mm_add_epi16(pixels_plus1L, temp0);										// c[0] + c[1]

			pixels_plus2L = Horizontal3x3GaussianFilter_SampleFirstPixel_SSE(pLocalSrc_NextRow);		// Horizontal filtering	- c[2]
			_mm_storeu_si128((__m128i *)pLocalRowPlus2, pixels_plus2L);
			pixels_plus2H = Horizontal3x3GaussianFilter_SampleFirstPixel_SSE(pLocalSrc_NextRow + 16);	// Horizontal filtering	- c[2]
			_mm_storeu_si128((__m128i *)(pLocalRowPlus2 + 8), pixels_plus2H);

			temp2 = _mm_loadu_si128((__m128i *) pLocalRowMinus1);										// c[-1]
			temp3 = _mm_loadu_si128((__m128i *) (pLocalRowMinus1 + 8));									// c[-1]

			temp1 = _mm_slli_epi16(temp1, 1);															// 2*c[0]
			temp0 = _mm_slli_epi16(temp0, 1);															// 2*c[0]

			pixels_plus1H = _mm_add_epi16(pixels_plus1H, temp3);										// c[-1] + c[0] + c[1]
			pixels_plus1L = _mm_add_epi16(pixels_plus1L, temp2);										// c[-1] + c[0] + c[1]

			temp2 = _mm_loadu_si128((__m128i *) pLocalRowMinus2);										// c[-2]
			temp3 = _mm_loadu_si128((__m128i *) (pLocalRowMinus2 + 8));									// c[-2]

			pixels_plus1H = _mm_slli_epi16(pixels_plus1H, 2);											// 4*c[-1] + 4*c[0] + 4*c[1]
			pixels_plus1L = _mm_slli_epi16(pixels_plus1L, 2);											// 4*c[-1] + 4*c[0] + 4*c[1]

			pixels_plus1H = _mm_add_epi16(pixels_plus1H, temp1);										// 4*c[-1] + 6*c[0] + 4*c[1]
			pixels_plus1L = _mm_add_epi16(pixels_plus1L, temp0);										// 4*c[-1] + 6*c[0] + 4*c[1]

			pixels_plus2H = _mm_add_epi16(pixels_plus2H, temp3);										// c[-2] + c[2]
			pixels_plus2L = _mm_add_epi16(pixels_plus2L, temp2);										// c[-2] + c[2]

			pixels_plus1H = _mm_add_epi16(pixels_plus1H, pixels_plus2H);								// c[-2] + 4*c[-1] + 4*c[0] + 4*c[1] + c[2]
			pixels_plus1L = _mm_add_epi16(pixels_plus1L, pixels_plus2L);								// c[-2] + 4*c[-1] + 4*c[0] + 4*c[1] + c[2]

			pixels_plus1H = _mm_srli_epi16(pixels_plus1H, 8);											// Divide by 256
			pixels_plus1L = _mm_srli_epi16(pixels_plus1L, 8);											// Divide by 256

			pixels_plus1L = _mm_packus_epi16(pixels_plus1L, pixels_plus1H);
			_mm_store_si128((__m128i *)pLocalDst, pixels_plus1L);

			pLocalSrc += 32;
			pLocalSrc_NextRow += 32;
			pLocalDst += 16;
			pLocalRowMinus2 += 16;
			pLocalRowMinus1 += 16;
			pLocalRowCurr += 16;
			pLocalRowPlus1 += 16;
			pLocalRowPlus2 += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc += 2, pLocalSrc_NextRow += 2)
		{
			short temp_plus1 = Horizontal5x5GaussianFilter_C(pLocalSrc);				// row + 1
			*pLocalRowPlus1++ = temp_plus1;
			short temp_plus2 = Horizontal5x5GaussianFilter_C(pLocalSrc_NextRow);		// row + 2
			*pLocalRowPlus2++ = temp_plus2;

			*pLocalDst++ = (unsigned char)((*pLocalRowMinus2++ + 4 * (*pLocalRowMinus1++) + 6 * (*pLocalRowCurr++) + 4 * temp_plus1 + temp_plus2) >> 8);
		}

		// Move two rows ahead
		pTemp0 = pRowMinus2;
		pTemp1 = pRowMinus1;
		pRowMinus2 = pRowCurr;
		pRowMinus1 = pRowPlus1;
		pRowCurr = pRowPlus2;
		pRowPlus1 = pTemp1;
		pRowPlus2 = pTemp0;

		pLocalRowMinus2 = pRowMinus2;
		pLocalRowMinus1 = pRowMinus1;
		pLocalRowCurr = pRowCurr;
		pLocalRowPlus1 = pRowPlus1;
		pLocalRowPlus2 = pRowPlus2;

		pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);
		pDstImage += dstImageStrideInBytes;
		height--;
	}

	return AGO_SUCCESS;
}

#define FP_BITS		16
#define FP_MUL		(1<<FP_BITS)
#define FP_ROUND    (1<<15)

// The kernel does a gaussian blur followed by ORB scaling

/* Gaussian Kernel			1   4   6   4   1			1		1   4   6   4   1
							4  16  24  16   4			4
					1/256	6  24  36  24   6    =		6									>> 8
							4  16  24  16   4			4
							1   4   6   4   1			1
*/

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
	)
{
	int xpos, ypos, x;
	// need to recalculate scale_factor because they might differ from global scale_factor for different pyramid levels.
	float xcale = (float)srcWidth / dstWidth;
	float yscale = (float)srcHeight / (dstHeight + 4);
	int xinc = (int)(FP_MUL * xcale);		// to convert to fixed point
	int yinc = (int)(FP_MUL * yscale);
	unsigned short *Xmap = (unsigned short *)pLocalData;
	unsigned short *r0 = (Xmap + ((dstWidth+15)&~15));
	vx_uint8 *r1 = (vx_uint8 *)(r0 + ((srcWidth&15)&~15));
	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);

	// generate xmap for orbit scaling
	// generate xmap;
	xpos = (int)(0.5f*xinc);
	for (x = 0; x < (int)dstWidth; x++, xpos += xinc)
	{
		int xmap;
		xmap = (xpos >> FP_BITS);
		Xmap[x] = (unsigned short)xmap;
	}

	ypos = (int)((2.5f) * yinc);  //starting from row 2 of dstimage
	// do gaussian verical filter for ypos
	for (int y = 0; y < (int)dstHeight; y++, ypos += yinc)
	{
		unsigned int x;
		unsigned int *pdst = (unsigned int *)pDstImage;
		const vx_uint8* pSrc = pSrcImage + (ypos >> FP_BITS)*srcImageStrideInBytes;
		const vx_uint8* srow0 = pSrc - 2 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrc - srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrc + srcImageStrideInBytes;
		const vx_uint8* srow3 = pSrc + 2*srcImageStrideInBytes;
		// do vertical convolution
		for (x = 0; x < srcWidth; x += 16)
		{
			__m128i s0 = _mm_load_si128((const __m128i*)(srow0 + x));
			__m128i s1 = _mm_load_si128((const __m128i*)(srow1 + x));
			__m128i s2 = _mm_load_si128((const __m128i*)(pSrc + x));
			__m128i s3 = _mm_load_si128((const __m128i*)(srow2 + x));
			__m128i s4 = _mm_load_si128((const __m128i*)(srow3 + x));
			__m128i s0_L = _mm_unpacklo_epi8(s0, z);
			__m128i s4_L = _mm_unpacklo_epi8(s4, z);
			s0 = _mm_unpackhi_epi8(s0, z);
			s4 = _mm_unpackhi_epi8(s4, z);
			s0_L = _mm_add_epi16(s0_L, s4_L);
			s0 = _mm_add_epi16(s0, s4);
			__m128i s1_L = _mm_add_epi16(_mm_unpacklo_epi8(s1, z), _mm_unpacklo_epi8(s3, z));
			s1 = _mm_add_epi16(_mm_unpackhi_epi8(s1, z), _mm_unpackhi_epi8(s3, z));
			s4_L = _mm_unpacklo_epi8(s2, z);
			s2  = _mm_unpackhi_epi8(s2, z);
			s0_L = _mm_add_epi16(s0_L, _mm_slli_epi16(s1_L, 2));
			s0 = _mm_add_epi16(s0, _mm_slli_epi16(s1, 2));
			s0_L = _mm_add_epi16(s0_L, _mm_mullo_epi16(s4_L, c6));		// low 8 filtered
			s0 = _mm_add_epi16(s0, _mm_mullo_epi16(s2, c6));			// Hi 8 filtered.
			// copy to temp
			_mm_store_si128((__m128i*)(r0 + x), s0_L);
			_mm_store_si128((__m128i*)(r0 + x+ 8), s0);
		}

		// do horizontal convolution and copy to r1
		for (x = 0; x <srcWidth; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));
			s0 = _mm_add_epi16(s0, s4);
			s1 = _mm_add_epi16(s1, s3);
			s0 = _mm_add_epi16(s0, _mm_slli_epi16(s1, 2));
			s0 = _mm_add_epi16(s0, _mm_mullo_epi16(s2, c6));			// filtered.
			s0 = _mm_srli_epi16(s0, 8);				// /256
			s0 = _mm_packus_epi16(s0, s0);
			_mm_storel_epi64((__m128i*)(r1 + x), s0);
		}
		// do NN scaling and copy to dst
		for (x = 0; x <= dstWidth-4; x += 4){
			const unsigned short *xm = &Xmap[x];
			*pdst++ = r1[xm[0]] | (r1[xm[1]] << 8) |
				(r1[xm[2]] << 16) | (r1[xm[3]] << 24);
		}
		for (; x < dstWidth; x++)
			pDstImage[x] = r1[Xmap[x]];

		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}