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

static const int n_offset[][2][2] = {
	{ { -1, 0 }, { 1, 0 } },
	{ { 1, -1 }, { -1, 1 } },
	{ { 0, -1 }, { 0, 1 } },
	{ { -1, -1 }, { 1, 1 } },
};
static const ago_coord2d_short_t dir_offsets[8] = {
	{ -1, -1 },
	{ 0, -1 },
	{ +1, -1 },
	{ -1, 0 },
	{ +1, 0 },
	{ -1, +1 },
	{ 0, +1 },
	{ +1, +1 },
};


int HafCpu_CannySobel_U16_U8_3x3_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	pSrcImage += srcImageStrideInBytes;
	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += dstride;		// don't care about border. start processing from row2
	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 1; y < (int)dstHeight - 1; y++)
	{
		const vx_uint8* srow0 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage;
		const vx_uint8* srow2 = pSrcImage + srcImageStrideInBytes;
		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 1] - (vx_int16)srow0[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow2[x - 1] + 2 * ((vx_int16)srow1[x + 1] - (vx_int16)srow1[x - 1]);
			vx_int16 Gy = (vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow0[x - 1] - (vx_int16)srow0[x + 1] + 2 * ((vx_int16)srow2[x] - (vx_int16)srow0[x]);
			Gy = ~Gy + 1;
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i t0 = _mm_add_epi16(_mm_add_epi16(s0, s2), _mm_slli_epi16(s1, 1));
			__m128i t1 = _mm_sub_epi16(s2, s0);
			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r1 + x));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));

			__m128i t0 = _mm_sub_epi16(s1, s0);
			__m128i t1 = _mm_add_epi16(_mm_add_epi16(s2, s4), _mm_slli_epi16(s3, 1));
			t1 = _mm_sub_epi16(z, t1);
			for (int i = 0; i < 8; i++){
				M128I(s1).m128i_i16[i] = HafCpu_FastAtan2_Canny(M128I(t0).m128i_i16[i], M128I(t1).m128i_i16[i]);
			}
			t0 = _mm_add_epi16(_mm_abs_epi16(t0), _mm_abs_epi16(t1));
			// pack with signed saturation
			t0 = _mm_or_si128(_mm_slli_epi16(t0, 2), s1);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), t0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 1] - (vx_int16)srow0[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow2[x - 1] + 2 * ((vx_int16)srow1[x + 1] - (vx_int16)srow1[x - 1]);
			vx_int16 Gy = (vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow0[x - 1] - (vx_int16)srow0[x + 1] + 2 * ((vx_int16)srow2[x] - (vx_int16)srow0[x]);
			Gy = ~Gy + 1;
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}

// Using separable filter
//			-1	-2	0	2	1			1
//										4
//  Gx =								6
//										4
//										1

int HafCpu_CannySobel_U16_U8_5x5_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);
	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += 2 * dstride;		// don't care about border. start processing from row2
	pSrcImage += 2 * srcImageStrideInBytes;
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 2; y < (int)dstHeight - 2; y++)
	{
		const vx_uint8* srow0 = pSrcImage - 2 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrcImage;
		const vx_uint8* srow3 = pSrcImage + srcImageStrideInBytes;
		const vx_uint8* srow4 = pSrcImage + 2 * srcImageStrideInBytes;

		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 2] + (2 * ((vx_int16)srow0[x + 1])) - (2 * ((vx_int16)srow0[x - 1])) - (vx_int16)srow0[x - 2]
				+ 4 * ((vx_int16)srow1[x + 2] + (2 * ((vx_int16)srow1[x + 1])) - (2 * ((vx_int16)srow1[x - 1])) - (vx_int16)srow1[x - 2])
				+ 6 * ((vx_int16)srow2[x + 2] + (2 * ((vx_int16)srow2[x + 1])) - (2 * ((vx_int16)srow2[x - 1])) - (vx_int16)srow2[x - 2])
				+ 4 * ((vx_int16)srow3[x + 2] + (2 * ((vx_int16)srow3[x + 1])) - (2 * ((vx_int16)srow3[x - 1])) - (vx_int16)srow3[x - 2])
				+ (vx_int16)srow4[x + 2] + (2 * ((vx_int16)srow4[x + 1])) - (2 * ((vx_int16)srow4[x - 1])) - (vx_int16)srow4[x - 2];
			vx_int16 Gy = (vx_int16)srow4[x - 2] + (4 * (vx_int16)srow4[x - 1]) + (6 * (vx_int16)srow4[x]) + (4 * (vx_int16)srow4[x + 1]) + (vx_int16)srow4[x + 2]
				+ 2 * ((vx_int16)srow3[x - 2] + (4 * (vx_int16)srow3[x - 1]) + (6 * (vx_int16)srow3[x]) + (4 * (vx_int16)srow3[x + 1]) + (vx_int16)srow3[x + 2])
				- 2 * ((vx_int16)srow1[x - 2] + (4 * (vx_int16)srow1[x - 1]) + (6 * (vx_int16)srow1[x]) + (4 * (vx_int16)srow1[x + 1]) + (vx_int16)srow1[x + 2])
				- ((vx_int16)srow0[x - 2] + (4 * (vx_int16)srow0[x - 1]) + (6 * (vx_int16)srow0[x]) + (4 * (vx_int16)srow0[x + 1]) + (vx_int16)srow0[x + 2]);
			Gy = ~Gy + 1;
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution - SSE
		for (x = prefixWidth; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow3 + x)), z);
			__m128i s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow4 + x)), z);

			__m128i t0 = _mm_add_epi16(_mm_slli_epi16(_mm_add_epi16(s1, s3), 2), _mm_mullo_epi16(s2, c6));
			t0 = _mm_add_epi16(t0, _mm_add_epi16(s0, s4));

			__m128i t1 = _mm_slli_epi16(_mm_sub_epi16(s3, s1), 1);
			t1 = _mm_add_epi16(t1, _mm_sub_epi16(s4, s0));
			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));

			__m128i s4 = _mm_loadu_si128((const __m128i*)(r1 + x - 2));
			__m128i s5 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			__m128i s6 = _mm_loadu_si128((const __m128i*)(r1 + x));
			__m128i s7 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));
			__m128i s8 = _mm_loadu_si128((const __m128i*)(r1 + x + 2));

			__m128i t0 = _mm_slli_epi16(_mm_sub_epi16(s2, s1), 1);
			t0 = _mm_adds_epi16(t0, _mm_sub_epi16(s3, s0));
			__m128i t1 = _mm_slli_epi16(_mm_add_epi16(s5, s7), 2);
			s0 = _mm_mullo_epi16(s6, c6);
			t1 = _mm_add_epi16(t1, _mm_add_epi16(s4, s8));
			t1 = _mm_add_epi16(t1, s0);
			t1 = _mm_sub_epi16(z, t1);
			// find magnitude
			s0 = _mm_add_epi16(_mm_abs_epi16(t0), _mm_abs_epi16(t1));
			//s0 = _mm_min_epi16(s0, clamp);
			for (int i = 0; i < 8; i++){
				M128I(t0).m128i_i16[i] = HafCpu_FastAtan2_Canny(M128I(t0).m128i_i16[i], M128I(t1).m128i_i16[i]);
			}
			s0 = _mm_or_si128(_mm_slli_epi16(s0, 2), t0);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), s0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 2] + (2 * ((vx_int16)srow0[x + 1])) - (2 * ((vx_int16)srow0[x - 1])) - (vx_int16)srow0[x - 2]
				+ 4 * ((vx_int16)srow1[x + 2] + (2 * ((vx_int16)srow1[x + 1])) - (2 * ((vx_int16)srow1[x - 1])) - (vx_int16)srow1[x - 2])
				+ 6 * ((vx_int16)srow2[x + 2] + (2 * ((vx_int16)srow2[x + 1])) - (2 * ((vx_int16)srow2[x - 1])) - (vx_int16)srow2[x - 2])
				+ 4 * ((vx_int16)srow3[x + 2] + (2 * ((vx_int16)srow3[x + 1])) - (2 * ((vx_int16)srow3[x - 1])) - (vx_int16)srow3[x - 2])
				+ (vx_int16)srow4[x + 2] + (2 * ((vx_int16)srow4[x + 1])) - (2 * ((vx_int16)srow4[x - 1])) - (vx_int16)srow4[x - 2];
			vx_int16 Gy = (vx_int16)srow4[x - 2] + (4 * (vx_int16)srow4[x - 1]) + (6 * (vx_int16)srow4[x]) + (4 * (vx_int16)srow4[x + 1]) + (vx_int16)srow4[x + 2]
				+ 2 * ((vx_int16)srow3[x - 2] + (4 * (vx_int16)srow3[x - 1]) + (6 * (vx_int16)srow3[x]) + (4 * (vx_int16)srow3[x + 1]) + (vx_int16)srow3[x + 2])
				- 2 * ((vx_int16)srow1[x - 2] + (4 * (vx_int16)srow1[x - 1]) + (6 * (vx_int16)srow1[x]) + (4 * (vx_int16)srow1[x + 1]) + (vx_int16)srow1[x + 2])
				- ((vx_int16)srow0[x - 2] + (4 * (vx_int16)srow0[x - 1]) + (6 * (vx_int16)srow0[x]) + (4 * (vx_int16)srow0[x + 1]) + (vx_int16)srow0[x + 2]);
			Gy = ~Gy + 1;
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}

	return AGO_SUCCESS;
}

int HafCpu_CannySobel_U16_U8_7x7_L1NORM
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint16   * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i z = _mm_setzero_si128(), c5 = _mm_set1_epi16(5), c6 = _mm_set1_epi16(6);
	__m128i c15 = _mm_set1_epi16(15), c20 = _mm_set1_epi16(20);
	__m128i clamp = _mm_set1_epi16(0x3FFF);

	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += 3 * dstride;		// don't care about border. start processing from row2
	pSrcImage += 3 * srcImageStrideInBytes;
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 3; y < (int)dstHeight - 3; y++)
	{
		const vx_uint8* srow0 = pSrcImage - 3 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage - 2 * srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow3 = pSrcImage;
		const vx_uint8* srow4 = pSrcImage + srcImageStrideInBytes;
		const vx_uint8* srow5 = pSrcImage + 2 * srcImageStrideInBytes;
		const vx_uint8* srow6 = pSrcImage + 3 * srcImageStrideInBytes;

		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 3] + (4 * (vx_int16)srow0[x + 2]) + (5 * (vx_int16)srow0[x + 1]) - (5 * (vx_int16)srow0[x - 1]) - (4 * (vx_int16)srow0[x - 2]) - (vx_int16)srow0[x - 3]
				+ 6 * ((vx_int16)srow1[x + 3] + (4 * (vx_int16)srow1[x + 2]) + (5 * (vx_int16)srow1[x + 1]) - (5 * (vx_int16)srow1[x - 1]) - (4 * (vx_int16)srow1[x - 2]) - (vx_int16)srow1[x - 3])
				+ 15 * ((vx_int16)srow2[x + 3] + (4 * (vx_int16)srow2[x + 2]) + (5 * (vx_int16)srow2[x + 1]) - (5 * (vx_int16)srow2[x - 1]) - (4 * (vx_int16)srow2[x - 2]) - (vx_int16)srow2[x - 3])
				+ 20 * ((vx_int16)srow3[x + 3] + (4 * (vx_int16)srow3[x + 2]) + (5 * (vx_int16)srow3[x + 1]) - (5 * (vx_int16)srow3[x - 1]) - (4 * (vx_int16)srow3[x - 2]) - (vx_int16)srow3[x - 3])
				+ 15 * ((vx_int16)srow4[x + 3] + (4 * (vx_int16)srow4[x + 2]) + (5 * (vx_int16)srow4[x + 1]) - (5 * (vx_int16)srow4[x - 1]) - (4 * (vx_int16)srow4[x - 2]) - (vx_int16)srow4[x - 3])
				+ 6 * ((vx_int16)srow5[x + 3] + (4 * (vx_int16)srow5[x + 2]) + (5 * (vx_int16)srow5[x + 1]) - (5 * (vx_int16)srow5[x - 1]) - (4 * (vx_int16)srow5[x - 2]) - (vx_int16)srow5[x - 3])
				+ (vx_int16)srow6[x + 3] + (4 * (vx_int16)srow6[x + 2]) + (5 * (vx_int16)srow6[x + 1]) - (5 * (vx_int16)srow6[x - 1]) - (4 * (vx_int16)srow6[x - 2]) - (vx_int16)srow6[x - 3];
			vx_int16 Gy = (vx_int16)srow6[x - 3] + (vx_int16)srow6[x + 3] + (6 * ((vx_int16)srow6[x - 2] + (vx_int16)srow6[x + 2])) + (15 * ((vx_int16)srow6[x - 1] + (vx_int16)srow6[x + 1])) + (20 * (vx_int16)srow6[x])
				+ 4 * ((vx_int16)srow5[x - 3] + (vx_int16)srow5[x + 3] + (6 * ((vx_int16)srow5[x - 2] + (vx_int16)srow5[x + 2])) + (15 * ((vx_int16)srow5[x - 1] + (vx_int16)srow5[x + 1])) + (20 * (vx_int16)srow5[x]))
				+ 5 * ((vx_int16)srow4[x - 3] + (vx_int16)srow4[x + 3] + (6 * ((vx_int16)srow4[x - 2] + (vx_int16)srow4[x + 2])) + (15 * ((vx_int16)srow4[x - 1] + (vx_int16)srow4[x + 1])) + (20 * (vx_int16)srow4[x]))
				- 5 * ((vx_int16)srow2[x - 3] + (vx_int16)srow2[x + 3] + (6 * ((vx_int16)srow2[x - 2] + (vx_int16)srow2[x + 2])) + (15 * ((vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1])) + (20 * (vx_int16)srow2[x]))
				- 4 * ((vx_int16)srow1[x - 3] + (vx_int16)srow1[x + 3] + (6 * ((vx_int16)srow1[x - 2] + (vx_int16)srow1[x + 2])) + (15 * ((vx_int16)srow1[x - 1] + (vx_int16)srow1[x + 1])) + (20 * (vx_int16)srow1[x]))
				- ((vx_int16)srow0[x - 3] + (vx_int16)srow0[x + 3] + (6 * ((vx_int16)srow0[x - 2] + (vx_int16)srow0[x + 2])) + (15 * ((vx_int16)srow0[x - 1] + (vx_int16)srow0[x + 1])) + (20 * (vx_int16)srow0[x]));
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution - SSE
		for (x = prefixWidth; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow3 + x)), z);
			__m128i s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow4 + x)), z);
			__m128i s5 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow5 + x)), z);
			__m128i s6 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow6 + x)), z);

			__m128i t0 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s1, s5), c6), _mm_mullo_epi16(s3, c20));
			__m128i t2 = _mm_mullo_epi16(_mm_add_epi16(s2, s4), c15);
			t0 = _mm_add_epi16(t0, _mm_add_epi16(s0, s6));
			__m128i t1 = _mm_slli_epi16(_mm_sub_epi16(s5, s1), 2);
			t0 = _mm_add_epi16(t0, t2);

			t2 = _mm_mullo_epi16(_mm_sub_epi16(s4, s2), c5);
			t0 = _mm_srai_epi16(t0, 2);
			t1 = _mm_add_epi16(t1, _mm_sub_epi16(s6, s0));
			t1 = _mm_add_epi16(t1, t2);
			t1 = _mm_srai_epi16(t1, 2);

			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 3));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));
			__m128i s5 = _mm_loadu_si128((const __m128i*)(r0 + x + 3));


			__m128i t0 = _mm_slli_epi16(_mm_subs_epi16(s4, s1), 2);
			__m128i t1 = _mm_mullo_epi16(_mm_subs_epi16(s3, s2), c5);
			t0 = _mm_adds_epi16(t0, _mm_subs_epi16(s5, s0));
			t0 = _mm_adds_epi16(t0, t1);

			s0 = _mm_loadu_si128((const __m128i*)(r1 + x - 3));
			s1 = _mm_loadu_si128((const __m128i*)(r1 + x - 2));
			s2 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			s3 = _mm_loadu_si128((const __m128i*)(r1 + x));
			s4 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));
			s5 = _mm_loadu_si128((const __m128i*)(r1 + x + 2));
			__m128i s6 = _mm_loadu_si128((const __m128i*)(r1 + x + 3));


			t1 = _mm_adds_epi16(_mm_mullo_epi16(_mm_add_epi16(s1, s5), c6), _mm_mullo_epi16(s3, c20));
			__m128i t2 = _mm_mullo_epi16(_mm_add_epi16(s2, s4), c15);
			t1 = _mm_adds_epi16(t1, _mm_adds_epi16(s0, s6));
			t1 = _mm_adds_epi16(t1, t2);
			t1 = _mm_subs_epi16(z, t1);
			// find magnitude
			s0 = _mm_add_epi16(_mm_abs_epi16(t0), _mm_abs_epi16(t1));
			s0 = _mm_min_epi16(s0, clamp);
			for (int i = 0; i < 8; i++){
				M128I(t0).m128i_i16[i] = HafCpu_FastAtan2_Canny(M128I(t0).m128i_i16[i], M128I(t1).m128i_i16[i]);
			}
			s0 = _mm_or_si128(_mm_slli_epi16(s0, 2), t0);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), s0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 3] + (4 * (vx_int16)srow0[x + 2]) + (5 * (vx_int16)srow0[x + 1]) - (5 * (vx_int16)srow0[x - 1]) - (4 * (vx_int16)srow0[x - 2]) - (vx_int16)srow0[x - 3]
				+ 6 * ((vx_int16)srow1[x + 3] + (4 * (vx_int16)srow1[x + 2]) + (5 * (vx_int16)srow1[x + 1]) - (5 * (vx_int16)srow1[x - 1]) - (4 * (vx_int16)srow1[x - 2]) - (vx_int16)srow1[x - 3])
				+ 15 * ((vx_int16)srow2[x + 3] + (4 * (vx_int16)srow2[x + 2]) + (5 * (vx_int16)srow2[x + 1]) - (5 * (vx_int16)srow2[x - 1]) - (4 * (vx_int16)srow2[x - 2]) - (vx_int16)srow2[x - 3])
				+ 20 * ((vx_int16)srow3[x + 3] + (4 * (vx_int16)srow3[x + 2]) + (5 * (vx_int16)srow3[x + 1]) - (5 * (vx_int16)srow3[x - 1]) - (4 * (vx_int16)srow3[x - 2]) - (vx_int16)srow3[x - 3])
				+ 15 * ((vx_int16)srow4[x + 3] + (4 * (vx_int16)srow4[x + 2]) + (5 * (vx_int16)srow4[x + 1]) - (5 * (vx_int16)srow4[x - 1]) - (4 * (vx_int16)srow4[x - 2]) - (vx_int16)srow4[x - 3])
				+ 6 * ((vx_int16)srow5[x + 3] + (4 * (vx_int16)srow5[x + 2]) + (5 * (vx_int16)srow5[x + 1]) - (5 * (vx_int16)srow5[x - 1]) - (4 * (vx_int16)srow5[x - 2]) - (vx_int16)srow5[x - 3])
				+ (vx_int16)srow6[x + 3] + (4 * (vx_int16)srow6[x + 2]) + (5 * (vx_int16)srow6[x + 1]) - (5 * (vx_int16)srow6[x - 1]) - (4 * (vx_int16)srow6[x - 2]) - (vx_int16)srow6[x - 3];
			vx_int16 Gy = (vx_int16)srow6[x - 3] + (vx_int16)srow6[x + 3] + (6 * ((vx_int16)srow6[x - 2] + (vx_int16)srow6[x + 2])) + (15 * ((vx_int16)srow6[x - 1] + (vx_int16)srow6[x + 1])) + (20 * (vx_int16)srow6[x])
				+ 4 * ((vx_int16)srow5[x - 3] + (vx_int16)srow5[x + 3] + (6 * ((vx_int16)srow5[x - 2] + (vx_int16)srow5[x + 2])) + (15 * ((vx_int16)srow5[x - 1] + (vx_int16)srow5[x + 1])) + (20 * (vx_int16)srow5[x]))
				+ 5 * ((vx_int16)srow4[x - 3] + (vx_int16)srow4[x + 3] + (6 * ((vx_int16)srow4[x - 2] + (vx_int16)srow4[x + 2])) + (15 * ((vx_int16)srow4[x - 1] + (vx_int16)srow4[x + 1])) + (20 * (vx_int16)srow4[x]))
				- 5 * ((vx_int16)srow2[x - 3] + (vx_int16)srow2[x + 3] + (6 * ((vx_int16)srow2[x - 2] + (vx_int16)srow2[x + 2])) + (15 * ((vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1])) + (20 * (vx_int16)srow2[x]))
				- 4 * ((vx_int16)srow1[x - 3] + (vx_int16)srow1[x + 3] + (6 * ((vx_int16)srow1[x - 2] + (vx_int16)srow1[x + 2])) + (15 * ((vx_int16)srow1[x - 1] + (vx_int16)srow1[x + 1])) + (20 * (vx_int16)srow1[x]))
				- ((vx_int16)srow0[x - 3] + (vx_int16)srow0[x + 3] + (6 * ((vx_int16)srow0[x - 2] + (vx_int16)srow0[x + 2])) + (15 * ((vx_int16)srow0[x - 1] + (vx_int16)srow0[x + 1])) + (20 * (vx_int16)srow0[x]));
			vx_int16 tmp = abs(Gx) + abs(Gy);
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}


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
	)
{
	vx_int16 *Gx, *Gy;
	vx_uint8 * pTemp;
	vx_uint32 dstride = ((dstWidth + 15)&~15);
	Gx = (vx_int16 *)pScratch;
	Gy = (vx_int16 *)(pScratch + dstride*sizeof(vx_int16));
	pTemp = pScratch + 2*dstride*sizeof(vx_int16);
	// compute Sobel gradients
	HafCpu_Sobel_S16S16_U8_3x3_GXY(dstWidth, dstHeight - 2, Gx + dstride, dstride * 2, Gy + dstride, dstride * 2, pSrcImage + srcImageStrideInBytes, srcImageStrideInBytes, pTemp);
	
	// compute L1 norm and phase
	unsigned int y = 1;
	vx_int16 *pGx = Gx + dstride;
	vx_int16 *pGy = Gy + dstride;
	vx_int16 *pMag = Gx;
	while (y < dstHeight)
	{
		vx_uint16 *pdst = (vx_uint16*)pMag;		// to store the result

		for (unsigned int x = 1; x < dstWidth; x++)
		{
			vx_uint8 orn;	// orientation

			float scale = (float)128 / 180.f;
			float arct = HafCpu_FastAtan2_deg(pGx[x], pGy[x]);
			// normalize and convert to degrees 0-180
			orn = (((int)(arct*scale) + 16) >> 5)&7;		// quantize to 8 (22.5 degrees)
			if (orn >= 4)orn -= 4;
			vx_int16 val = (vx_int16)(abs(pGx[x]) + abs(pGy[x]));
			pdst[x] = (vx_uint16)((val << 2) | orn);				// store both mag and orientation
		}
		pGx += dstride;
		pGy += dstride;
		pMag += dstride;
		y++;
	}

	// do minmax suppression: from Gx
	ago_coord2d_ushort_t *pxyStack = xyStack;
	for (y = 1; y < dstHeight - 1; y++)
	{
		vx_uint8* pOut = pDst + y*dstStrideInBytes;
		vx_int16 *pSrc = (vx_int16 *)(Gx + y * dstride);	// we are processing from 2nd row
		for (unsigned int x = 1; x < dstWidth - 1; x++, pSrc++)
		{
			vx_int32 edge;
			// get the Mag and angle
			int mag = (pSrc[0] >> 2);
			int ang = pSrc[0] & 3;
			int offset0 = n_offset[ang][0][1] * dstride + n_offset[ang][0][0];
			int offset1 = n_offset[ang][1][1] * dstride + n_offset[ang][1][0];
			edge = ((mag >(pSrc[offset0] >> 2)) && (mag >(pSrc[offset1] >> 2))) ? mag : 0;
			if (edge > hyst_upper){
				pOut[x] = (vx_int8)255;
				// add the cordinates to stacktop
				pxyStack->x = x;	// store x and y co-ordinates
				pxyStack->y = y;	// store x and y co-ordinates
				pxyStack++;
			}
			else if (edge <= hyst_lower){
				pOut[x] = 0;
			}
			else pOut[x] = 127;
		}
	}
	*pxyStackTop = (vx_uint32)(pxyStack - xyStack);

	return AGO_SUCCESS;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	return AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;
}

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
	)
{
	// do minmax suppression: from Gx
	vx_uint32 sstride = srcStrideInBytes>>1;
	ago_coord2d_ushort_t *pxyStack = xyStack;
	for (unsigned int y = 1; y < dstHeight - 1; y++)
	{
		vx_uint8* pOut = pDst + y*dstStrideInBytes;
		vx_uint16 *pLocSrc = pSrc + y * sstride + 1;	// we are processing from 2nd row
		for (unsigned int x = 1; x < dstWidth - 1; x++, pLocSrc++)
		{
			vx_int32 edge;
			// get the Mag and angle
			int mag = (pLocSrc[0] >> 2);
			int ang = pLocSrc[0] & 3;
			int offset0 = n_offset[ang][0][1] * sstride + n_offset[ang][0][0];
			int offset1 = n_offset[ang][1][1] * sstride + n_offset[ang][1][0];
			edge = ((mag >(pLocSrc[offset0] >> 2)) && (mag >(pLocSrc[offset1] >> 2))) ? mag : 0;
			if (edge > hyst_upper){
				pOut[x] = (vx_int8)255;
				// add the cordinates to stacktop
				pxyStack->x = x;	// store x and y co-ordinates
				pxyStack->y = y;	// store x and y co-ordinates
				pxyStack++;

			}
			else if (edge <= hyst_lower){
				pOut[x] = 0;
			}
			else pOut[x] = 127;
		}
	}
	*pxyStackTop = (vx_uint32)(pxyStack - xyStack);
	return AGO_SUCCESS;
}

int HafCpu_CannyEdgeTrace_U8_U8XY
	(
		vx_uint32              dstWidth,
		vx_uint32              dstHeight,
		vx_uint8             * pDstImage,
		vx_uint32              dstImageStrideInBytes,
		vx_uint32              capacityOfXY,
		ago_coord2d_ushort_t   xyStack[],
		vx_uint32              xyStackTop
	)
{
	ago_coord2d_ushort_t *pxyStack = xyStack + xyStackTop;
	while (pxyStack != xyStack){
			pxyStack--;
			vx_uint16 x = pxyStack->x;
			vx_uint16 y = pxyStack->y;
			// look at all the neighbors for strong edge value
		for (int i = 0; i < 8; i++){
			const ago_coord2d_short_t offs = dir_offsets[i];
			vx_int16 x1 = x + offs.x;
			vx_int16 y1 = y + offs.y;
			vx_uint8 *pDst = pDstImage + y1*dstImageStrideInBytes + x1;
			if (*pDst == 127)
			{
				*pDst |= 0x80;		// *pDst = 255
				*((unsigned *)pxyStack) = (y1<<16)|x1;
				pxyStack++;
			}
		}
	}
	// go through the entire destination and convert all 127 to 0
	const __m128i mm127 = _mm_set1_epi8((char)127);
	for (unsigned int y = 0; y < dstHeight; y++) {
		__m128i * src = (__m128i *)pDstImage;
		vx_uint32 width = (dstWidth + 15) >> 4;

		for (unsigned int x = 0; x < width; x++) {
			__m128i mask;
			__m128i pixels = _mm_load_si128(src);
			mask = _mm_cmpeq_epi8(pixels, mm127);
			pixels = _mm_andnot_si128(mask, pixels);
			_mm_store_si128(src++, pixels);
		}
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_CannySobel_U16_U8_3x3_L2NORM
(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint16   * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_uint8    * pLocalData
)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	pSrcImage += srcImageStrideInBytes;
	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += dstride;		// don't care about border. start processing from row2
	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 1; y < (int)dstHeight - 1; y++)
	{
		const vx_uint8* srow0 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage;
		const vx_uint8* srow2 = pSrcImage + srcImageStrideInBytes;
		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 1] - (vx_int16)srow0[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow2[x - 1] + 2 * ((vx_int16)srow1[x + 1] - (vx_int16)srow1[x - 1]);
			vx_int16 Gy = (vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow0[x - 1] - (vx_int16)srow0[x + 1] + 2 * ((vx_int16)srow2[x] - (vx_int16)srow0[x]);
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i t0 = _mm_add_epi16(_mm_add_epi16(s0, s2), _mm_slli_epi16(s1, 1));
			__m128i t1 = _mm_sub_epi16(s2, s0);
			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst - SSE
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r1 + x));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));

			__m128i t0 = _mm_sub_epi16(s1, s0);
			__m128i t1 = _mm_add_epi16(_mm_add_epi16(s2, s4), _mm_slli_epi16(s3, 1));
			t1 = _mm_sub_epi16(z, t1);
			s0 = _mm_mullo_epi16(t0, t0);
			s1 = _mm_mullo_epi16(t1, t1);
			// unpack to dwords for multiplication
			s2 = _mm_unpackhi_epi16(s0, z);
			s0 = _mm_unpacklo_epi16(s0, z);
			s3 = _mm_unpackhi_epi16(s1, z);
			s1 = _mm_unpacklo_epi16(s1, z);
			__m128 f0 = _mm_cvtepi32_ps(s0);
			__m128 f1 = _mm_cvtepi32_ps(s2);
			__m128 f2 = _mm_cvtepi32_ps(s1);
			__m128 f3 = _mm_cvtepi32_ps(s3);
			f0 = _mm_add_ps(f0, f2);
			f1 = _mm_add_ps(f1, f3);
			f0 = _mm_sqrt_ps(f0);
			f1 = _mm_sqrt_ps(f1);

			for (int i = 0; i < 8; i++){
				M128I(s1).m128i_i16[i] = HafCpu_FastAtan2_Canny(M128I(t0).m128i_i16[i], M128I(t1).m128i_i16[i]);
			}
			t0 = _mm_cvtps_epi32(f0);
			t1 = _mm_cvtps_epi32(f1);
			// pack with signed saturation
			t0 = _mm_packus_epi32(t0, t1);
			t0 = _mm_or_si128(_mm_slli_epi16(t0, 2), s1);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), t0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 1] - (vx_int16)srow0[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow2[x - 1] + 2 * ((vx_int16)srow1[x + 1] - (vx_int16)srow1[x - 1]);
			vx_int16 Gy = (vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1] - (vx_int16)srow0[x - 1] - (vx_int16)srow0[x + 1] + 2 * ((vx_int16)srow2[x] - (vx_int16)srow0[x]);
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}

int HafCpu_CannySobel_U16_U8_5x5_L2NORM
	(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint16   * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i z = _mm_setzero_si128(), c6 = _mm_set1_epi16(6);
	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += 2 * dstride;		// don't care about border. start processing from row2
	pSrcImage += 2 * srcImageStrideInBytes;
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 2; y < (int)dstHeight - 2; y++)
	{
		const vx_uint8* srow0 = pSrcImage - 2 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrcImage;
		const vx_uint8* srow3 = pSrcImage + srcImageStrideInBytes;
		const vx_uint8* srow4 = pSrcImage + 2 * srcImageStrideInBytes;

		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 2] + (2 * ((vx_int16)srow0[x + 1])) - (2 * ((vx_int16)srow0[x - 1])) - (vx_int16)srow0[x - 2]
				+ 4 * ((vx_int16)srow1[x + 2] + (2 * ((vx_int16)srow1[x + 1])) - (2 * ((vx_int16)srow1[x - 1])) - (vx_int16)srow1[x - 2])
				+ 6 * ((vx_int16)srow2[x + 2] + (2 * ((vx_int16)srow2[x + 1])) - (2 * ((vx_int16)srow2[x - 1])) - (vx_int16)srow2[x - 2])
				+ 4 * ((vx_int16)srow3[x + 2] + (2 * ((vx_int16)srow3[x + 1])) - (2 * ((vx_int16)srow3[x - 1])) - (vx_int16)srow3[x - 2])
				+ (vx_int16)srow4[x + 2] + (2 * ((vx_int16)srow4[x + 1])) - (2 * ((vx_int16)srow4[x - 1])) - (vx_int16)srow4[x - 2];
			vx_int16 Gy = (vx_int16)srow4[x - 2] + (4 * (vx_int16)srow4[x - 1]) + (6 * (vx_int16)srow4[x]) + (4 * (vx_int16)srow4[x + 1]) + (vx_int16)srow4[x + 2]
				+ 2 * ((vx_int16)srow3[x - 2] + (4 * (vx_int16)srow3[x - 1]) + (6 * (vx_int16)srow3[x]) + (4 * (vx_int16)srow3[x + 1]) + (vx_int16)srow3[x + 2])
				- 2 * ((vx_int16)srow1[x - 2] + (4 * (vx_int16)srow1[x - 1]) + (6 * (vx_int16)srow1[x]) + (4 * (vx_int16)srow1[x + 1]) + (vx_int16)srow1[x + 2])
				- (vx_int16)srow0[x - 2] + (4 * (vx_int16)srow0[x - 1]) + (6 * (vx_int16)srow0[x]) + (4 * (vx_int16)srow0[x + 1]) + (vx_int16)srow0[x + 2];
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution
		for (x = prefixWidth; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow3 + x)), z);
			__m128i s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow4 + x)), z);

			__m128i t0 = _mm_add_epi16(_mm_slli_epi16(_mm_add_epi16(s1, s3), 2), _mm_mullo_epi16(s2, c6));
			t0 = _mm_add_epi16(t0, _mm_add_epi16(s0, s4));

			__m128i t1 = _mm_slli_epi16(_mm_sub_epi16(s3, s1), 1);
			t1 = _mm_add_epi16(t1, _mm_sub_epi16(s4, s0));
			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));

			__m128i s4 = _mm_loadu_si128((const __m128i*)(r1 + x - 2));
			__m128i s5 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			__m128i s6 = _mm_loadu_si128((const __m128i*)(r1 + x));
			__m128i s7 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));
			__m128i s8 = _mm_loadu_si128((const __m128i*)(r1 + x + 2));

			__m128i t0 = _mm_slli_epi16(_mm_sub_epi16(s2, s1), 1);
			t0 = _mm_adds_epi16(t0, _mm_sub_epi16(s3, s0));
			__m128i t1 = _mm_slli_epi16(_mm_add_epi16(s5, s7), 2);
			s0 = _mm_mullo_epi16(s6, c6);
			t1 = _mm_add_epi16(t1, _mm_add_epi16(s4, s8));
			t1 = _mm_adds_epi16(t1, s0);
			t1 = _mm_sub_epi16(z, t1);
			// unpack for multiplication
			s0 = _mm_unpacklo_epi16(t0, t1);
			s2 = _mm_unpackhi_epi16(t0, t1);
			s0 = _mm_madd_epi16(s0, s0);
			s2 = _mm_madd_epi16(s2, s2);

			__m128 f0 = _mm_cvtepi32_ps(s0);
			__m128 f1 = _mm_cvtepi32_ps(s2);
			f0 = _mm_sqrt_ps(f0);
			f1 = _mm_sqrt_ps(f1);

			for (int i = 0; i < 8; i++){
				M128I(s1).m128i_i16[i] = HafCpu_FastAtan2_Canny(M128I(t0).m128i_i16[i], M128I(t1).m128i_i16[i]);
			}
			t0 = _mm_cvtps_epi32(f0);
			t1 = _mm_cvtps_epi32(f1);
			// pack with signed saturation
			t0 = _mm_packus_epi32(t0, t1);
			t0 = _mm_or_si128(_mm_slli_epi16(t0, 2), s1);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), t0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 2] + (2 * ((vx_int16)srow0[x + 1])) - (2 * ((vx_int16)srow0[x - 1])) - (vx_int16)srow0[x - 2]
				+ 4 * ((vx_int16)srow1[x + 2] + (2 * ((vx_int16)srow1[x + 1])) - (2 * ((vx_int16)srow1[x - 1])) - (vx_int16)srow1[x - 2])
				+ 6 * ((vx_int16)srow2[x + 2] + (2 * ((vx_int16)srow2[x + 1])) - (2 * ((vx_int16)srow2[x - 1])) - (vx_int16)srow2[x - 2])
				+ 4 * ((vx_int16)srow3[x + 2] + (2 * ((vx_int16)srow3[x + 1])) - (2 * ((vx_int16)srow3[x - 1])) - (vx_int16)srow3[x - 2])
				+ (vx_int16)srow4[x + 2] + (2 * ((vx_int16)srow4[x + 1])) - (2 * ((vx_int16)srow4[x - 1])) - (vx_int16)srow4[x - 2];
			vx_int16 Gy = (vx_int16)srow4[x - 2] + (4 * (vx_int16)srow4[x - 1]) + (6 * (vx_int16)srow4[x]) + (4 * (vx_int16)srow4[x + 1]) + (vx_int16)srow4[x + 2]
				+ 2 * ((vx_int16)srow3[x - 2] + (4 * (vx_int16)srow3[x - 1]) + (6 * (vx_int16)srow3[x]) + (4 * (vx_int16)srow3[x + 1]) + (vx_int16)srow3[x + 2])
				- 2 * ((vx_int16)srow1[x - 2] + (4 * (vx_int16)srow1[x - 1]) + (6 * (vx_int16)srow1[x]) + (4 * (vx_int16)srow1[x + 1]) + (vx_int16)srow1[x + 2])
				- (vx_int16)srow0[x - 2] + (4 * (vx_int16)srow0[x - 1]) + (6 * (vx_int16)srow0[x]) + (4 * (vx_int16)srow0[x + 1]) + (vx_int16)srow0[x + 2];
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}

int HafCpu_CannySobel_U16_U8_7x7_L2NORM
	(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint16   * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_uint8    * pLocalData
	)
{
	int x, y;
	int prefixWidth = ((intptr_t)(pDstImage)) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i z = _mm_setzero_si128(), c5 = _mm_set1_epi16(5), c6 = _mm_set1_epi16(6);
	__m128i c15 = _mm_set1_epi16(15), c20 = _mm_set1_epi16(20);
	__m128i clamp = _mm_set1_epi16(0x3FFF);

	vx_uint32 dstride = dstImageStrideInBytes >> 1;
	pDstImage += 3 * dstride;		// don't care about border. start processing from row2
	pSrcImage += 3 * srcImageStrideInBytes;
	vx_int16 *r0 = (vx_int16*)(pLocalData + 16);
	vx_int16 *r1 = r0 + ((dstWidth + 15) & ~15);

	for (y = 3; y < (int)dstHeight - 3; y++)
	{
		const vx_uint8* srow0 = pSrcImage - 3 * srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage - 2 * srcImageStrideInBytes;
		const vx_uint8* srow2 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow3 = pSrcImage;
		const vx_uint8* srow4 = pSrcImage + srcImageStrideInBytes;
		const vx_uint8* srow5 = pSrcImage + 2 * srcImageStrideInBytes;
		const vx_uint8* srow6 = pSrcImage + 3 * srcImageStrideInBytes;

		vx_uint16* drow = (vx_uint16*)pDstImage;

		for (x = 0; x < prefixWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 3] + (4 * (vx_int16)srow0[x + 2]) + (5 * (vx_int16)srow0[x + 1]) - (5 * (vx_int16)srow0[x - 1]) - (4 * (vx_int16)srow0[x - 2]) - (vx_int16)srow0[x - 3]
				+ 6 * ((vx_int16)srow1[x + 3] + (4 * (vx_int16)srow1[x + 2]) + (5 * (vx_int16)srow1[x + 1]) - (5 * (vx_int16)srow1[x - 1]) - (4 * (vx_int16)srow1[x - 2]) - (vx_int16)srow1[x - 3])
				+ 15 * ((vx_int16)srow2[x + 3] + (4 * (vx_int16)srow2[x + 2]) + (5 * (vx_int16)srow2[x + 1]) - (5 * (vx_int16)srow2[x - 1]) - (4 * (vx_int16)srow2[x - 2]) - (vx_int16)srow2[x - 3])
				+ 20 * ((vx_int16)srow3[x + 3] + (4 * (vx_int16)srow3[x + 2]) + (5 * (vx_int16)srow3[x + 1]) - (5 * (vx_int16)srow3[x - 1]) - (4 * (vx_int16)srow3[x - 2]) - (vx_int16)srow3[x - 3])
				+ 15 * ((vx_int16)srow4[x + 3] + (4 * (vx_int16)srow4[x + 2]) + (5 * (vx_int16)srow4[x + 1]) - (5 * (vx_int16)srow4[x - 1]) - (4 * (vx_int16)srow4[x - 2]) - (vx_int16)srow4[x - 3])
				+ 6 * ((vx_int16)srow5[x + 3] + (4 * (vx_int16)srow5[x + 2]) + (5 * (vx_int16)srow5[x + 1]) - (5 * (vx_int16)srow5[x - 1]) - (4 * (vx_int16)srow5[x - 2]) - (vx_int16)srow5[x - 3])
				+ (vx_int16)srow6[x + 3] + (4 * (vx_int16)srow6[x + 2]) + (5 * (vx_int16)srow6[x + 1]) - (5 * (vx_int16)srow6[x - 1]) - (4 * (vx_int16)srow6[x - 2]) - (vx_int16)srow6[x - 3];
			vx_int16 Gy = (vx_int16)srow6[x - 3] + (vx_int16)srow6[x + 3] + (6 * ((vx_int16)srow6[x - 2] + (vx_int16)srow6[x + 2])) + (15 * ((vx_int16)srow6[x - 1] + (vx_int16)srow6[x + 1])) + (20 * (vx_int16)srow6[x])
				+ 4 * ((vx_int16)srow5[x - 3] + (vx_int16)srow5[x + 3] + (6 * ((vx_int16)srow5[x - 2] + (vx_int16)srow5[x + 2])) + (15 * ((vx_int16)srow5[x - 1] + (vx_int16)srow5[x + 1])) + (20 * (vx_int16)srow5[x]))
				+ 5 * ((vx_int16)srow4[x - 3] + (vx_int16)srow4[x + 3] + (6 * ((vx_int16)srow4[x - 2] + (vx_int16)srow4[x + 2])) + (15 * ((vx_int16)srow4[x - 1] + (vx_int16)srow4[x + 1])) + (20 * (vx_int16)srow4[x]))
				- 5 * ((vx_int16)srow2[x - 3] + (vx_int16)srow2[x + 3] + (6 * ((vx_int16)srow2[x - 2] + (vx_int16)srow2[x + 2])) + (15 * ((vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1])) + (20 * (vx_int16)srow2[x]))
				- 4 * ((vx_int16)srow1[x - 3] + (vx_int16)srow1[x + 3] + (6 * ((vx_int16)srow1[x - 2] + (vx_int16)srow1[x + 2])) + (15 * ((vx_int16)srow1[x - 1] + (vx_int16)srow1[x + 1])) + (20 * (vx_int16)srow1[x]))
				- ((vx_int16)srow0[x - 3] + (vx_int16)srow0[x + 3] + (6 * ((vx_int16)srow0[x - 2] + (vx_int16)srow0[x + 2])) + (15 * ((vx_int16)srow0[x - 1] + (vx_int16)srow0[x + 1])) + (20 * (vx_int16)srow0[x]));
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}

		// do vertical convolution
		for (x = prefixWidth; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i s3 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow3 + x)), z);
			__m128i s4 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow4 + x)), z);
			__m128i s5 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow5 + x)), z);
			__m128i s6 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow6 + x)), z);

			__m128i t0 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s1, s5), c6), _mm_mullo_epi16(s3, c20));
			__m128i t2 = _mm_mullo_epi16(_mm_add_epi16(s2, s4), c15);
			t0 = _mm_add_epi16(t0, _mm_add_epi16(s0, s6));
			__m128i t1 = _mm_slli_epi16(_mm_sub_epi16(s5, s1), 2);
			t0 = _mm_add_epi16(t0, t2);

			t2 = _mm_mullo_epi16(_mm_sub_epi16(s4, s2), c5);
			t1 = _mm_add_epi16(t1, _mm_sub_epi16(s6, s0));
			t0 = _mm_srai_epi16(t0, 2);
			t1 = _mm_add_epi16(t1, t2);
			t1 = _mm_srai_epi16(t1, 2);

			_mm_store_si128((__m128i*)(r0 + x), t0);
			_mm_store_si128((__m128i*)(r1 + x), t1);
		}

		// do horizontal convolution, interleave the results and store them to dst
		x = prefixWidth;
		for (; x <= alignedWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 3));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x - 2));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(r0 + x + 2));
			__m128i s5 = _mm_loadu_si128((const __m128i*)(r0 + x + 3));


			__m128i t0 = _mm_slli_epi16(_mm_subs_epi16(s4, s1), 2);
			__m128i t1 = _mm_mullo_epi16(_mm_subs_epi16(s3, s2), c5);
			t0 = _mm_adds_epi16(t0, _mm_subs_epi16(s5, s0));
			t0 = _mm_adds_epi16(t0, t1);

			s0 = _mm_loadu_si128((const __m128i*)(r1 + x - 3));
			s1 = _mm_loadu_si128((const __m128i*)(r1 + x - 2));
			s2 = _mm_loadu_si128((const __m128i*)(r1 + x - 1));
			s3 = _mm_loadu_si128((const __m128i*)(r1 + x));
			s4 = _mm_loadu_si128((const __m128i*)(r1 + x + 1));
			s5 = _mm_loadu_si128((const __m128i*)(r1 + x + 2));
			__m128i s6 = _mm_loadu_si128((const __m128i*)(r1 + x + 3));


			t1 = _mm_adds_epi16(_mm_mullo_epi16(_mm_add_epi16(s1, s5), c6), _mm_mullo_epi16(s3, c20));
			__m128i t2 = _mm_mullo_epi16(_mm_add_epi16(s2, s4), c15);
			t1 = _mm_adds_epi16(t1, _mm_adds_epi16(s0, s6));
			t1 = _mm_adds_epi16(t1, t2);
			t1 = _mm_subs_epi16(z, t1);
			// unpack for multiplication
			s0 = _mm_unpacklo_epi16(t0, t1);
			s2 = _mm_unpackhi_epi16(t0, t1);
			s0 = _mm_madd_epi16(s0, s0);
			s2 = _mm_madd_epi16(s2, s2);

			__m128 f0 = _mm_cvtepi32_ps(s0);
			__m128 f1 = _mm_cvtepi32_ps(s2);
			f0 = _mm_sqrt_ps(f0);
			f1 = _mm_sqrt_ps(f1);
			for (int i = 0; i < 8; i++){
				M128I(s1).m128i_i16[i] = HafCpu_FastAtan2_Canny(M128I(t0).m128i_i16[i], M128I(t1).m128i_i16[i]);
			}
			t0 = _mm_cvtps_epi32(f0);
			t1 = _mm_cvtps_epi32(f1);
			// pack with signed saturation
			t0 = _mm_packus_epi32(t0, t1);
			t0 = _mm_or_si128(_mm_slli_epi16(t0, 2), s1);
			// store magnitude and angle to destination
			_mm_store_si128((__m128i*)(drow + x), t0);
		}

		for (x = alignedWidth + prefixWidth - 1; x < (int)dstWidth; x++)
		{
			vx_int16 Gx = (vx_int16)srow0[x + 3] + (4 * (vx_int16)srow0[x + 2]) + (5 * (vx_int16)srow0[x + 1]) - (5 * (vx_int16)srow0[x - 1]) - (4 * (vx_int16)srow0[x - 2]) - (vx_int16)srow0[x - 3]
				+ 6 * ((vx_int16)srow1[x + 3] + (4 * (vx_int16)srow1[x + 2]) + (5 * (vx_int16)srow1[x + 1]) - (5 * (vx_int16)srow1[x - 1]) - (4 * (vx_int16)srow1[x - 2]) - (vx_int16)srow1[x - 3])
				+ 15 * ((vx_int16)srow2[x + 3] + (4 * (vx_int16)srow2[x + 2]) + (5 * (vx_int16)srow2[x + 1]) - (5 * (vx_int16)srow2[x - 1]) - (4 * (vx_int16)srow2[x - 2]) - (vx_int16)srow2[x - 3])
				+ 20 * ((vx_int16)srow3[x + 3] + (4 * (vx_int16)srow3[x + 2]) + (5 * (vx_int16)srow3[x + 1]) - (5 * (vx_int16)srow3[x - 1]) - (4 * (vx_int16)srow3[x - 2]) - (vx_int16)srow3[x - 3])
				+ 15 * ((vx_int16)srow4[x + 3] + (4 * (vx_int16)srow4[x + 2]) + (5 * (vx_int16)srow4[x + 1]) - (5 * (vx_int16)srow4[x - 1]) - (4 * (vx_int16)srow4[x - 2]) - (vx_int16)srow4[x - 3])
				+ 6 * ((vx_int16)srow5[x + 3] + (4 * (vx_int16)srow5[x + 2]) + (5 * (vx_int16)srow5[x + 1]) - (5 * (vx_int16)srow5[x - 1]) - (4 * (vx_int16)srow5[x - 2]) - (vx_int16)srow5[x - 3])
				+ (vx_int16)srow6[x + 3] + (4 * (vx_int16)srow6[x + 2]) + (5 * (vx_int16)srow6[x + 1]) - (5 * (vx_int16)srow6[x - 1]) - (4 * (vx_int16)srow6[x - 2]) - (vx_int16)srow6[x - 3];
			vx_int16 Gy = (vx_int16)srow6[x - 3] + (vx_int16)srow6[x + 3] + (6 * ((vx_int16)srow6[x - 2] + (vx_int16)srow6[x + 2])) + (15 * ((vx_int16)srow6[x - 1] + (vx_int16)srow6[x + 1])) + (20 * (vx_int16)srow6[x])
				+ 4 * ((vx_int16)srow5[x - 3] + (vx_int16)srow5[x + 3] + (6 * ((vx_int16)srow5[x - 2] + (vx_int16)srow5[x + 2])) + (15 * ((vx_int16)srow5[x - 1] + (vx_int16)srow5[x + 1])) + (20 * (vx_int16)srow5[x]))
				+ 5 * ((vx_int16)srow4[x - 3] + (vx_int16)srow4[x + 3] + (6 * ((vx_int16)srow4[x - 2] + (vx_int16)srow4[x + 2])) + (15 * ((vx_int16)srow4[x - 1] + (vx_int16)srow4[x + 1])) + (20 * (vx_int16)srow4[x]))
				- 5 * ((vx_int16)srow2[x - 3] + (vx_int16)srow2[x + 3] + (6 * ((vx_int16)srow2[x - 2] + (vx_int16)srow2[x + 2])) + (15 * ((vx_int16)srow2[x - 1] + (vx_int16)srow2[x + 1])) + (20 * (vx_int16)srow2[x]))
				- 4 * ((vx_int16)srow1[x - 3] + (vx_int16)srow1[x + 3] + (6 * ((vx_int16)srow1[x - 2] + (vx_int16)srow1[x + 2])) + (15 * ((vx_int16)srow1[x - 1] + (vx_int16)srow1[x + 1])) + (20 * (vx_int16)srow1[x]))
				- ((vx_int16)srow0[x - 3] + (vx_int16)srow0[x + 3] + (6 * ((vx_int16)srow0[x - 2] + (vx_int16)srow0[x + 2])) + (15 * ((vx_int16)srow0[x - 1] + (vx_int16)srow0[x + 1])) + (20 * (vx_int16)srow0[x]));
			vx_int16 tmp = (vx_int16)sqrt((Gx*Gx) + (Gy*Gy));
			tmp <<= 2;
			tmp |= (HafCpu_FastAtan2_Canny(Gx, Gy) & 3);
			drow[x] = tmp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstride;
	}
	return AGO_SUCCESS;
}