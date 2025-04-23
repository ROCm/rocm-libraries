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

// The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
// It processes the pixels in a width which is the next highest multiple of 16 after dstWidth
static int HafCpu_Histogram1Threshold_DATA_U8
	(
		vx_uint32     dstHist[],
		vx_uint8      distThreshold,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	// offset: to convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	// thresh: source threshold in -128..127 range
	__m128i offset = _mm_set1_epi8((char)0x80);
	__m128i thresh = _mm_set1_epi8((char)((distThreshold - 1) ^ 0x80));
	__m128i onemask = _mm_set1_epi8((char)1);
	// process one pixel row at a time that counts "pixel < srcThreshold"
	__m128i count = _mm_set1_epi8((char)0);
	vx_uint8 * srcRow = pSrcImage;
	vx_uint32 width = (srcWidth + 15) >> 4;
	for (unsigned int y = 0; y < srcHeight; y++) {
		__m128i * src = (__m128i *)srcRow;
		for (unsigned int x = 0; x < width; x++) {
			__m128i pixels = _mm_load_si128(src++);
			pixels = _mm_xor_si128(pixels, offset);
			pixels = _mm_cmpgt_epi8(pixels, thresh);
			pixels = _mm_and_si128(pixels, onemask);
			pixels = _mm_sad_epu8(pixels, onemask);
			count = _mm_add_epi32(count, pixels);
		}
		srcRow += srcImageStrideInBytes;
	}
	// extract histogram from count
	dstHist[0] = M128I(count).m128i_u32[0] + M128I(count).m128i_u32[2];
	dstHist[1] = srcWidth * srcHeight - dstHist[0];
	return AGO_SUCCESS;
}

static int HafCpu_Histogram3Thresholds_DATA_U8
	(
		vx_uint32     dstHist[],
		vx_uint8      distThreshold0,
		vx_uint8      distThreshold1,
		vx_uint8      distThreshold2,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	// offset: to convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	// thresh: source threshold in -128..127 range
	__m128i offset = _mm_set1_epi8((char)0x80);
	__m128i T0 = _mm_set1_epi8((char)((distThreshold0 - 1) ^ 0x80));
	__m128i T1 = _mm_set1_epi8((char)((distThreshold1 - 1) ^ 0x80));
	__m128i T2 = _mm_set1_epi8((char)((distThreshold2 - 1) ^ 0x80));
	__m128i onemask = _mm_set1_epi8((char)1);
	// process one pixel row at a time that counts "pixel < srcThreshold"
	__m128i count0 = _mm_set1_epi8((char)0);
	__m128i count1 = _mm_set1_epi8((char)0);
	__m128i count2 = _mm_set1_epi8((char)0);
	vx_uint8 * srcRow = pSrcImage;
	vx_uint32 width = (srcWidth + 15) >> 4;
	for (unsigned int y = 0; y < srcHeight; y++) {
		__m128i * src = (__m128i *)srcRow;
		for (unsigned int x = 0; x < width; x++) {
			__m128i pixels = _mm_load_si128(src++);
			pixels = _mm_xor_si128(pixels, offset);
			__m128i cmpout;
			cmpout = _mm_cmpgt_epi8(pixels, T0);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count0 = _mm_add_epi32(count0, cmpout);
			cmpout = _mm_cmpgt_epi8(pixels, T1);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count1 = _mm_add_epi32(count1, cmpout);
			cmpout = _mm_cmpgt_epi8(pixels, T2);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count2 = _mm_add_epi32(count2, cmpout);
		}
		srcRow += srcImageStrideInBytes;
	}
	// extract histogram from count: special case needed when T1 == T2
	dstHist[0] = M128I(count0).m128i_u32[0] + M128I(count0).m128i_u32[2];
	dstHist[1] = M128I(count1).m128i_u32[0] + M128I(count1).m128i_u32[2] - dstHist[0];
	dstHist[2] = M128I(count2).m128i_u32[0] + M128I(count2).m128i_u32[2] - dstHist[0] - dstHist[1];
	dstHist[3] = srcWidth * srcHeight - dstHist[0] - dstHist[1] - dstHist[2];
	if (M128I(T1).m128i_i8[0] == M128I(T2).m128i_i8[0]) {
		dstHist[2] = dstHist[3];
		dstHist[3] = 0;
	}
	return AGO_SUCCESS;
}

static int HafCpu_Histogram8Bins_DATA_U8
	(
		vx_uint32   * dstHist,
		vx_uint8      distOffset, 
		vx_uint8      distWindow,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	// offset: to convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	// thresh: source threshold in -128..127 range
	__m128i offset = _mm_set1_epi8((char)0x80);
	__m128i T0 = _mm_set1_epi8((char)(((distOffset ? distOffset : distWindow) - 1) ^ 0x80));
	__m128i dT = _mm_set1_epi8((char)distWindow);
	__m128i onemask = _mm_set1_epi8((char)1);
	// process one pixel row at a time that counts "pixel < srcThreshold"
	vx_uint32 count[9] = { 0 };
	vx_uint8 * srcRow = pSrcImage;
	vx_uint32 width = (srcWidth + 15) >> 4;
	for (unsigned int y = 0; y < srcHeight; y++) {
		__m128i * src = (__m128i *)srcRow;
		__m128i count0 = _mm_set1_epi8((char)0);
		__m128i count1 = _mm_set1_epi8((char)0);
		__m128i count2 = _mm_set1_epi8((char)0);
		for (unsigned int x = 0; x < width; x++) {
			__m128i pixels = _mm_load_si128(src++);
			pixels = _mm_xor_si128(pixels, offset);
			__m128i cmpout, Tnext = T0;
			// 0..3
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 32);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 48);
			count0 = _mm_add_epi32(count0, cmpout);
			// 4..7
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 32);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 48);
			count1 = _mm_add_epi32(count1, cmpout);
			// 8
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count2 = _mm_add_epi32(count2, cmpout);
		}
		srcRow += srcImageStrideInBytes;
		// move counts from count0..2 into count[]
		for (int i = 0; i < 4; i++) {
			count[ 0 + i] += M128I(count0).m128i_u16[i] + M128I(count0).m128i_u16[4 + i];
			count[ 4 + i] += M128I(count1).m128i_u16[i] + M128I(count1).m128i_u16[4 + i];
		}
		count[8 + 0] += M128I(count2).m128i_u16[0] + M128I(count2).m128i_u16[4 + 0];
	}
	// extract histogram from count
	if (distOffset == 0) {
		vx_uint32 last = (distWindow >= 32) ? srcWidth * srcHeight : count[7];
		for (int i = 6; i >= 0; i--) {
			count[i] = last - count[i];
			last -= count[i];
		}
		dstHist[0] = last;
		for (int i = 1; i < 8; i++)
			dstHist[i] = count[i - 1];
	}
	else {
		vx_uint32 last = (distOffset + distWindow * 8 - 1 > 255) ? srcWidth * srcHeight : count[8];
		for (int i = 7; i >= 0; i--) {
			count[i] = last - count[i];
			last -= count[i];
			dstHist[i] = count[i];
		}
	}
	return AGO_SUCCESS;
}

static int HafCpu_Histogram9Bins_DATA_U8
	(
		vx_uint32   * dstHist,
		vx_uint8      distOffset,
		vx_uint8      distWindow,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	// offset: to convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	// thresh: source threshold in -128..127 range
	__m128i offset = _mm_set1_epi8((char)0x80);
	__m128i T0 = _mm_set1_epi8((char)(((distOffset ? distOffset : distWindow) - 1) ^ 0x80));
	__m128i dT = _mm_set1_epi8((char)distWindow);
	__m128i onemask = _mm_set1_epi8((char)1);
	// process one pixel row at a time that counts "pixel < srcThreshold"
	vx_uint32 count[10] = { 0 };
	vx_uint8 * srcRow = pSrcImage;
	vx_uint32 width = (srcWidth + 15) >> 4;
	for (unsigned int y = 0; y < srcHeight; y++) {
		__m128i * src = (__m128i *)srcRow;
		__m128i count0 = _mm_set1_epi8((char)0);
		__m128i count1 = _mm_set1_epi8((char)0);
		__m128i count2 = _mm_set1_epi8((char)0);
		for (unsigned int x = 0; x < width; x++) {
			__m128i pixels = _mm_load_si128(src++);
			pixels = _mm_xor_si128(pixels, offset);
			__m128i cmpout, Tnext = T0;
			// 0..3
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 32);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 48);
			count0 = _mm_add_epi32(count0, cmpout);
			// 4..7
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 32);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 48);
			count1 = _mm_add_epi32(count1, cmpout);
			// 8..9
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count2 = _mm_add_epi32(count2, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count2 = _mm_add_epi32(count2, cmpout);
		}
		srcRow += srcImageStrideInBytes;
		// move counts from count0..2 into count[]
		for (int i = 0; i < 4; i++) {
			count[0 + i] += M128I(count0).m128i_u16[i] + M128I(count0).m128i_u16[4 + i];
			count[4 + i] += M128I(count1).m128i_u16[i] + M128I(count1).m128i_u16[4 + i];
		}
		count[8 + 0] += M128I(count2).m128i_u16[0] + M128I(count2).m128i_u16[4 + 0];
		count[8 + 1] += M128I(count2).m128i_u16[1] + M128I(count2).m128i_u16[4 + 1];
	}
	// extract histogram from count
	if (distOffset == 0) {
		vx_uint32 last = (distWindow >= 29) ? srcWidth * srcHeight : count[8];
		for (int i = 7; i >= 0; i--) {
			count[i] = last - count[i];
			last -= count[i];
		}
		dstHist[0] = last;
		for (int i = 1; i < 9; i++)
			dstHist[i] = count[i - 1];
	}
	else {
		vx_uint32 last = (distOffset + distWindow * 9 - 1 > 255) ? srcWidth * srcHeight : count[9];
		for (int i = 8; i >= 0; i--) {
			count[i] = last - count[i];
			last -= count[i];
			dstHist[i] = count[i];
		}
	}
	return AGO_SUCCESS;
}

static int HafCpu_Histogram16Bins_DATA_U8
	(
		vx_uint32   * dstHist,
		vx_uint8      distOffset, 
		vx_uint8      distWindow,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	// offset: to convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	// thresh: source threshold in -128..127 range
	__m128i offset = _mm_set1_epi8((char)0x80);
	__m128i T0 = _mm_set1_epi8((char)(((distOffset ? distOffset : distWindow) - 1) ^ 0x80));
	__m128i dT = _mm_set1_epi8((char)distWindow);
	__m128i onemask = _mm_set1_epi8((char)1);
	// process one pixel row at a time that counts "pixel < srcThreshold"
	vx_uint32 count[16] = { 0 };
	vx_uint8 * srcRow = pSrcImage;
	vx_uint32 width = (srcWidth + 15) >> 4;
	for (unsigned int y = 0; y < srcHeight; y++) {
		__m128i * src = (__m128i *)srcRow;
		__m128i count0 = _mm_set1_epi8((char)0);
		__m128i count1 = _mm_set1_epi8((char)0);
		__m128i count2 = _mm_set1_epi8((char)0);
		__m128i count3 = _mm_set1_epi8((char)0);
		for (unsigned int x = 0; x < width; x++) {
			__m128i pixels = _mm_load_si128(src++);
			pixels = _mm_xor_si128(pixels, offset);
			__m128i cmpout, Tnext = T0;
			// 0..3
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 32);
			count0 = _mm_add_epi32(count0, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 48);
			count0 = _mm_add_epi32(count0, cmpout);
			// 4..7
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 32);
			count1 = _mm_add_epi32(count1, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 48);
			count1 = _mm_add_epi32(count1, cmpout);
			// 8..11
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count2 = _mm_add_epi32(count2, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count2 = _mm_add_epi32(count2, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 32);
			count2 = _mm_add_epi32(count2, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 48);
			count2 = _mm_add_epi32(count2, cmpout);
			// 12..15
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			count3 = _mm_add_epi32(count3, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 16);
			count3 = _mm_add_epi32(count3, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 32);
			count3 = _mm_add_epi32(count3, cmpout);
			Tnext = _mm_add_epi8(Tnext, dT);
			cmpout = _mm_cmpgt_epi8(pixels, Tnext);
			cmpout = _mm_and_si128(cmpout, onemask);
			cmpout = _mm_sad_epu8(cmpout, onemask);
			cmpout = _mm_slli_epi64(cmpout, 48);
			count3 = _mm_add_epi32(count3, cmpout);
		}
		srcRow += srcImageStrideInBytes;
		// move counts from count0..2 into count[]
		for (int i = 0; i < 4; i++) {
			count[ 0 + i] += M128I(count0).m128i_u16[i] + M128I(count0).m128i_u16[4 + i];
			count[ 4 + i] += M128I(count1).m128i_u16[i] + M128I(count1).m128i_u16[4 + i];
			count[ 8 + i] += M128I(count2).m128i_u16[i] + M128I(count2).m128i_u16[4 + i];
			count[12 + i] += M128I(count3).m128i_u16[i] + M128I(count3).m128i_u16[4 + i];
		}
	}
	// extract histogram from count
	if (distOffset == 0) {
		vx_uint32 last = (distWindow >= 16) ? srcWidth * srcHeight : count[15];
		for (int i = 14; i >= 0; i--) {
			count[i] = last - count[i];
			last -= count[i];
		}
		dstHist[0] = last;
		for (int i = 1; i < 16; i++)
			dstHist[i] = count[i - 1];
	}
	else {
		vx_uint32 last = srcWidth * srcHeight;
		for (int i = 15; i >= 0; i--) {
			count[i] = last - count[i];
			last -= count[i];
			dstHist[i] = count[i];
		}
	}
	return AGO_SUCCESS;
}

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
	)
{
	int status = AGO_ERROR_HAFCPU_NOT_IMPLEMENTED;

	// compute number of split points in [0..255] range to compute the histogram
	vx_int32 numSplits = (distBinCount - 1) + ((distOffset > 0) ? 1 : 0) + (((distOffset + distRange) < 256) ? 1 : 0);
	bool useGeneral = (srcWidth & 7) || (((intptr_t)pSrcImage) & 15);			// Use general code if width is not multiple of 8 or the buffer is unaligned
	if ((numSplits < 1 && distBinCount > 1) || (distBinCount == 0)) return status;

	if (numSplits <= 3 && !useGeneral) {
		if (numSplits == 0) {
			dstHist[0] = srcWidth * srcHeight;
			status = VX_SUCCESS;
		}
		else if (numSplits == 1) {
			vx_uint32 hist[2];
			status = HafCpu_Histogram1Threshold_DATA_U8(hist, distOffset ? distOffset : distWindow, srcWidth, srcHeight, pSrcImage, srcImageStrideInBytes);
			if (distBinCount == 1) {
				dstHist[0] = hist[distOffset > 0 ? 1 : 0];
			}
			else {
				dstHist[0] = hist[0];
				dstHist[1] = hist[1];
			}
		}
		else {
			// compute thresholds (split-points)
			vx_uint8 thresh[3], tlast = 0;
			vx_uint32 split = 0;
			if (distOffset > 0)
				tlast = thresh[split++] = distOffset;
			for (vx_uint32 bin = 1; bin < distBinCount; bin++)
				tlast = thresh[split++] = tlast + distWindow;
			if (split < 3) {
				if (((int)distOffset + distRange) < 256)
					tlast = thresh[split++] = tlast + distWindow;
				while (split < 3)
					thresh[split++] = tlast;
			}
			vx_uint32 count[4];
			status = HafCpu_Histogram3Thresholds_DATA_U8(count, thresh[0], thresh[1], thresh[2], srcWidth, srcHeight, pSrcImage, srcImageStrideInBytes);
			if (!status) {
				for (vx_uint32 i = 0; i < distBinCount; i++) {
					dstHist[i] = count[i + (distOffset ? 1 : 0)];
				}
			}
		}
	}
	// else if (distBinCount == 8 && !useGeneral) {
	// 	printf("distbin 8 method called\n");
	// 	status = HafCpu_Histogram8Bins_DATA_U8(dstHist, distOffset, distWindow, srcWidth, srcHeight, pSrcImage, srcImageStrideInBytes);
	// }
	// else if (distBinCount == 9 && !useGeneral) {
	// 	printf("distbin 9 method called\n");
	// 	status = HafCpu_Histogram9Bins_DATA_U8(dstHist, distOffset, distWindow, srcWidth, srcHeight, pSrcImage, srcImageStrideInBytes);
	// }
	else if (distBinCount == 16 && numSplits <= 16 && !useGeneral) {
		status = HafCpu_Histogram16Bins_DATA_U8(dstHist, distOffset, distWindow, srcWidth, srcHeight, pSrcImage, srcImageStrideInBytes);
	}
	else {
		// use general 256-bin histogram
		vx_uint32 histTmp[256];
		status = HafCpu_Histogram_DATA_U8(histTmp, srcWidth, srcHeight, pSrcImage, srcImageStrideInBytes);
		if (!status) {
			// convert [256] histogram into [numbins]
			if (distWindow == 1) {
				memcpy(dstHist, &histTmp[distOffset], distBinCount * sizeof(vx_uint32));
			}
			else {
				for (vx_uint32 i = distOffset; i < (distOffset + distRange); i++) {
					vx_size index = (i-distOffset) * distBinCount / distRange;
					dstHist[index] += histTmp[i];
				}
			}
		}
	}
	return status;
}
