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

DECL_ALIGN(16) unsigned char dataFastCornersPixelMask[7 * 16] ATTR_ALIGN(16) = {
	  1,   2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,
	255, 255,   4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 255,
	255, 255, 255,   6, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0, 255, 255,
	255, 255, 255, 255,   6, 255, 255, 255, 255, 255, 255, 255,   0, 255, 255, 255,
	255, 255, 255, 255, 255,   6, 255, 255, 255, 255, 255,   0, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255,   4, 255, 255, 255,   0, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255,   2,   1,   0, 255, 255, 255, 255, 255, 255
};

static inline void generateOffset(int srcStride, int * offsets)
{
	offsets[0] = -3 * srcStride;
	offsets[15] = offsets[0] - 1;
	offsets[1] = offsets[0] + 1;
	offsets[2] = -(srcStride << 1) + 2;
	offsets[14] = offsets[2] - 4;
	offsets[3] = -srcStride + 3;
	offsets[13] = offsets[3] - 6;
	offsets[4] = 3;
	offsets[12] = -3;
	offsets[5] = srcStride + 3;
	offsets[11] = offsets[5] - 6;
	offsets[6] = (srcStride << 1) + 2;
	offsets[10] = offsets[6] - 4;
	offsets[7] = 3 * srcStride + 1;
	offsets[8] = offsets[7] - 1;
	offsets[9] = offsets[8] - 1;

	return;
}

static inline void generateMasks_C(unsigned char * src, int srcStride, int* offsets, short t, int mask[2])
{
	mask[0] = 0;
	mask[1] = 0;
	int iterMask = 1;

	// Early exit conditions
	if ((abs((short)src[0] - (short)src[offsets[0]]) < t) && (abs((short)src[0] - (short)src[offsets[8]]) < t))					// Pixels 1 and 9 within t of the candidate
		return;
	if ((abs((short)src[0] - (short)src[offsets[4]]) < t) && (abs((short)src[0] - (short)src[offsets[12]]) < t))				// Pixels 5 and 13 within t of the candidate
		return;

	// Check for I_p + t
	short cand = (short)(*src) + t;
	for (int i = 0; i < 16; i++)
	{
		if ((short)src[offsets[i]] > cand)
			mask[0] |= iterMask;
		iterMask <<= 1;
	}

	// Check for I_p - t
	iterMask = 1;
	cand = (short)(*src) - t;
	for (int i = 0; i < 16; i++)
	{
		if ((short)src[offsets[i]] < cand)
			mask[1] |= iterMask;
		iterMask <<= 1;
	}
}

static inline bool isCorner(int mask[2])
{
	int cornerMask = 0x1FF;									// Nine 1's in the LSB

	if (mask[0] || mask[1])
	{
		mask[0] = mask[0] | (mask[0] << 16);
		mask[1] = mask[1] | (mask[1] << 16);

		for (int i = 0; i < 16; i++)
		{
			if (((mask[0] & cornerMask) == cornerMask) || ((mask[1] & cornerMask) == cornerMask))
				return true;
			mask[0] >>= 1;
			mask[1] >>= 1;
		}
	}
	
	return false;
}

static inline bool isCorner(int mask)
{
	int cornerMask = 0x1FF;									// Nine 1's in the LSB

	if (mask)
	{
		mask = mask | (mask << 16);
		for (int i = 0; i < 16; i++)
		{
			if ((mask & cornerMask) == cornerMask)
				return true;
			mask >>= 1;
		}
	}
	return false;
}

static inline bool isCornerPlus(short candidate, short * boundary, short t)
{
	// Early exit conditions
	if ((abs(candidate - boundary[0]) < t) && (abs(candidate - boundary[8]) < t))					// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(candidate - boundary[4]) < t) && (abs(candidate - boundary[12]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;

	candidate += t;
	int mask = 0;
	int iterMask = 1;
	for (int i = 0; i < 16; i++)
	{
		if (boundary[i] > candidate)
			mask |= iterMask;
		iterMask <<= 1;
	}

	return isCorner(mask);
}

static inline bool isCornerPlus_SSE(__m128i candidate, __m128i boundary, short t)
{
	__m128i boundaryH = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());								// Boundary 8..15 (words)
	__m128i boundaryL = _mm_cvtepu8_epi16(boundary);													// Boundary 0..7 (words)
	__m128i threshold = _mm_set1_epi16(t);

	short cand = M128I(candidate).m128i_i16[0];

	// Early exit conditions
	if ((abs(cand - M128I(boundaryL).m128i_i16[0]) < t) && (abs(cand - M128I(boundaryH).m128i_i16[0]) < t))			// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(cand - M128I(boundaryL).m128i_i16[4]) < t) && (abs(cand - M128I(boundaryH).m128i_i16[4]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;

	candidate = _mm_add_epi16(candidate, threshold);
	boundaryH = _mm_cmpgt_epi16(boundaryH, candidate);
	boundaryL = _mm_cmpgt_epi16(boundaryL, candidate);
	boundaryL = _mm_packs_epi16(boundaryL, boundaryH);											// 255 at ith byte if boundary[i] > pixel + t
	int mask = _mm_movemask_epi8(boundaryL);

	return isCorner(mask);
}

static inline bool isCornerMinus(short candidate, short * boundary, short t)
{
	// Early exit conditions
	if ((abs(candidate - boundary[0]) < t) && (abs(candidate - boundary[8]) < t))					// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(candidate - boundary[4]) < t) && (abs(candidate - boundary[12]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;

	candidate -= t;
	int mask = 0;
	int iterMask = 1;
	for (int i = 0; i < 16; i++)
	{
		if (boundary[i] < candidate)
			mask |= iterMask;
		iterMask <<= 1;
	}

	return isCorner(mask);
}

static inline bool isCornerMinus_SSE(__m128i candidate, __m128i boundary, short t)
{
	__m128i boundaryH = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());								// Boundary 8..15 (words)
	__m128i boundaryL = _mm_cvtepu8_epi16(boundary);													// Boundary 0..7 (words)
	__m128i threshold = _mm_set1_epi16(t);

	short cand = M128I(candidate).m128i_i16[0];

	// Early exit conditions
	if ((abs(cand - M128I(boundaryL).m128i_i16[0]) < t) && (abs(cand - M128I(boundaryH).m128i_i16[0]) < t))			// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs(cand - M128I(boundaryL).m128i_i16[4]) < t) && (abs(cand - M128I(boundaryH).m128i_i16[4]) < t))					// Pixels 5 and 13 within t of the candidate
		return false;

	candidate = _mm_sub_epi16(candidate, threshold);
	boundaryH = _mm_cmplt_epi16(boundaryH, candidate);
	boundaryL = _mm_cmplt_epi16(boundaryL, candidate);
	boundaryL = _mm_packs_epi16(boundaryL, boundaryH);											// 255 at ith byte if boundary[i] > pixel + t
	int mask = _mm_movemask_epi8(boundaryL);

	return isCorner(mask);
}

static inline bool checkForCornerAndGetStrength(unsigned char * src, int* offsets, short t, short * strength)
{
	// Early exit conditions
	if ((abs((short)src[0] - (short)src[offsets[0]]) < t) && (abs((short)src[0] - (short)src[offsets[8]]) < t))					// Pixels 1 and 9 within t of the candidate
		return false;
	if ((abs((short)src[0] - (short)src[offsets[4]]) < t) && (abs((short)src[0] - (short)src[offsets[12]]) < t))				// Pixels 5 and 13 within t of the candidate
		return false;

	// Get boundary
	short boundary[16];
	for (int i = 0; i < 16; i++)
		boundary[i] = (short)src[offsets[i]];

	// Check for I_p + t
	short cand = (short)(*src) + t;
	int maskP = 0;
	int iterMask = 1;
	for (int i = 0; i < 16; i++)
	{
		if (boundary[i] > cand)
			maskP |= iterMask;
		iterMask <<= 1;
	}

	// If it is a corner, then compute the threshold
	short strength_pos = 0;
	cand = src[0];
	if (isCorner(maskP))
	{
		short thresh_upper = 255;
		short thresh_lower = t;
		
		while (thresh_upper - thresh_lower > 1)						// Binary search
		{
			strength_pos = (thresh_upper + thresh_lower) >> 1;
			if (isCornerPlus(cand, boundary, strength_pos))
				thresh_lower = strength_pos;
			else
				thresh_upper = strength_pos;
		}
		strength_pos = thresh_lower;
	}

	// Check for I_p - t
	cand = (short)(*src) - t;
	int maskN = 0;
	iterMask = 1;
	for (int i = 0; i < 16; i++)
	{
		if (boundary[i] < cand)
			maskN |= iterMask;
		iterMask <<= 1;
	}

	// If it is a corner, then compute the threshold
	short strength_neg = 0;
	cand = src[0];
	if (isCorner(maskN))
	{
		short thresh_upper = 255;
		short thresh_lower = t;
		
		while (thresh_upper - thresh_lower > 1)						// Binary search
		{
			strength_neg = (thresh_upper + thresh_lower) >> 1;
			if (isCornerMinus(cand, boundary, strength_neg))
				thresh_lower = strength_neg;
			else
				thresh_upper = strength_neg;
		}
		strength_neg = thresh_lower;
	}

	if (maskP || maskN)
	{
		*strength = max(strength_pos, strength_neg);
		return true;
	}
	return false;
}

bool isCorner_SSE(unsigned char pixel, __m128i boundary, __m128i t)
{
	// Check for boundary > pixel + t
	__m128i cand = _mm_set1_epi16((short)pixel);									// The candidate pixel
	cand = _mm_add_epi16(cand, t);													// Pixel + t

	__m128i temp0 = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());				// Boundary 8..15 (words)
	__m128i temp1 = _mm_cvtepu8_epi16(boundary);									// Boundary 0..7 (words)

	temp0 = _mm_cmpgt_epi16(temp0, cand);
	temp1 = _mm_cmpgt_epi16(temp1, cand);
	temp1 = _mm_packs_epi16(temp1, temp0);											// 255 at ith byte if boundary[i] > pixel + t
	int mask = _mm_movemask_epi8(temp1);
	int plusMask = mask | (mask << 16);

	// Check for boundary > pixel - t
	cand = _mm_sub_epi16(cand, t);													// pixel + t - t = pixel
	cand = _mm_sub_epi16(cand, t);													// pixel - t

	temp0 = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());						// Boundary 8..15 (words)
	temp1 = _mm_cvtepu8_epi16(boundary);											// Boundary 0..7 (words)

	temp0 = _mm_cmplt_epi16(temp0, cand);
	temp1 = _mm_cmplt_epi16(temp1, cand);
	temp1 = _mm_packs_epi16(temp1, temp0);											// 255 at ith byte if boundary[i] > pixel + t
	mask = _mm_movemask_epi8(temp1);
	int minusMask = mask | (mask << 16);

	if (plusMask || minusMask)
	{
		int cornerMask = 0x1FF;														// Nine 1's in the LSB

		for (int i = 0; i < 16; i++)
		{
			if (((plusMask & cornerMask) == cornerMask) || ((minusMask & cornerMask) == cornerMask))
				return true;
			plusMask >>= 1;
			minusMask >>= 1;
		}
	}

	return false;
}

static inline bool checkForCornerAndGetStrength_SSE(unsigned char pixel, __m128i boundary, short threshold, short * strength)
{
	__m128i t = _mm_set1_epi16(threshold);

	// Check for boundary > pixel + t
	__m128i cand = _mm_set1_epi16((short)pixel);									// The candidate pixel
	cand = _mm_add_epi16(cand, t);													// Pixel + t

	__m128i temp0 = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());				// Boundary 8..15 (words)
	__m128i temp1 = _mm_cvtepu8_epi16(boundary);									// Boundary 0..7 (words)

	temp0 = _mm_cmpgt_epi16(temp0, cand);
	temp1 = _mm_cmpgt_epi16(temp1, cand);
	temp1 = _mm_packs_epi16(temp1, temp0);											// 255 at ith byte if boundary[i] > pixel + t
	int plusMask = _mm_movemask_epi8(temp1);

	// If it is a corner, then compute the threshold
	short strength_pos = 0;
	cand = _mm_sub_epi16(cand, t);
	if (isCorner(plusMask))
	{
		short thresh_upper = 255;
		short thresh_lower = threshold;

		while (thresh_upper - thresh_lower > 1)										// Binary search
		{
			strength_pos = (thresh_upper + thresh_lower) >> 1;
			if (isCornerPlus_SSE(cand, boundary, strength_pos))
				thresh_lower = strength_pos;
			else
				thresh_upper = strength_pos;
		}
		strength_pos = thresh_lower;
	}

	// Check for boundary > pixel - t
	cand = _mm_sub_epi16(cand, t);													// pixel - t

	temp0 = _mm_unpackhi_epi8(boundary, _mm_setzero_si128());						// Boundary 8..15 (words)
	temp1 = _mm_cvtepu8_epi16(boundary);											// Boundary 0..7 (words)

	temp0 = _mm_cmplt_epi16(temp0, cand);
	temp1 = _mm_cmplt_epi16(temp1, cand);
	temp1 = _mm_packs_epi16(temp1, temp0);											// 255 at ith byte if boundary[i] > pixel + t
	int minusMask = _mm_movemask_epi8(temp1);

	// If it is a corner, then compute the threshold
	short strength_neg = 0;
	cand = _mm_add_epi16(cand, t);
	if (isCorner(minusMask))
	{
		short thresh_upper = 255;
		short thresh_lower = threshold;

		while (thresh_upper - thresh_lower > 1)										// Binary search
		{
			strength_neg = (thresh_upper + thresh_lower) >> 1;
			if (isCornerMinus_SSE(cand, boundary, strength_neg))
				thresh_lower = strength_neg;
			else
				thresh_upper = strength_neg;
		}
		strength_neg = thresh_lower;
	}

	if (plusMask || minusMask)
	{
		*strength = max(strength_pos, strength_neg);
		return true;
	}
	return false;
}

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
	)
{
	unsigned char * pLocalSrc;
	int srcStride = (int)srcImageStrideInBytes;
	vx_uint32 cornerCount = 0;
	short t = (short)floorf(strength_threshold);
	
	pSrcImage += (srcStride * 3);														// Leave first three rows

	int alignedWidth = (int)srcWidth & ~7;
	int postfixWidth = (int)srcWidth & 7;

	// Generate offsets for C code if necessary
	int neighbor_offset[16] = { 0 };
	if (postfixWidth)
		generateOffset(srcStride, neighbor_offset);

	__m128i zeromask = _mm_setzero_si128();

	for (int height = 0; height < (int)(srcHeight - 6); height++)
	{
		pLocalSrc = (unsigned char *) pSrcImage;
		int width = 0;
		
		for (int x = 0; x < (alignedWidth >> 3); x++)
		{
			__m128i rowMinus3, rowMinus2, rowMinus1, row, rowPlus1, rowPlus2, rowPlus3;
			__m128i thresh = _mm_set1_epi16(t);

			// Check for early escape based on pixels 1 and 9 around the candidate
			rowMinus3 = _mm_loadu_si128((__m128i *)(pLocalSrc - 3 * srcStride - 1));
			rowMinus2 = _mm_srli_si128(rowMinus3, 1);									// row - 3: Pixels 0..7 in lower 7 bytes
			rowMinus2 = _mm_cvtepu8_epi16(rowMinus2);

			row = _mm_loadu_si128((__m128i *)(pLocalSrc - 3));
			rowMinus1 = _mm_srli_si128(row, 3);											// row: Pixels 0..7 in lower 7 bytes
			rowMinus1 = _mm_cvtepu8_epi16(rowMinus1);

			rowPlus3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 3 * srcStride - 1));
			rowPlus2 = _mm_srli_si128(rowPlus3, 1);										// row + 3: Pixels 0..7 in lower 7 bytes
			rowPlus2 = _mm_cvtepu8_epi16(rowPlus2);

			rowPlus1 = _mm_loadu_si128((__m128i *)(pLocalSrc + srcStride - 3));

			rowMinus2 = _mm_sub_epi16(rowMinus2, rowMinus1);
			rowMinus2 = _mm_abs_epi16(rowMinus2);
			rowPlus2 = _mm_sub_epi16(rowPlus2, rowMinus1);
			rowPlus2 = _mm_abs_epi16(rowPlus2);

			rowMinus2 = _mm_cmplt_epi16(rowMinus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate
			rowPlus2 = _mm_cmplt_epi16(rowPlus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate

			int maskSkip = _mm_movemask_epi8(rowMinus2);
			maskSkip &= _mm_movemask_epi8(rowPlus2);									// 1 if both 0 and 8 are within 't' of the candidate pixel

			// Check for early escape based on pixels 12 and 4 around the candidate
			rowMinus2 = _mm_cvtepu8_epi16(row);
			rowPlus2 = _mm_srli_si128(row, 6);
			rowPlus2 = _mm_cvtepu8_epi16(rowPlus2);

			rowMinus2 = _mm_sub_epi16(rowMinus2, rowMinus1);
			rowMinus2 = _mm_abs_epi16(rowMinus2);
			rowPlus2 = _mm_sub_epi16(rowPlus2, rowMinus1);
			rowPlus2 = _mm_abs_epi16(rowPlus2);

			rowMinus1 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcStride - 3));

			rowMinus2 = _mm_cmplt_epi16(rowMinus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate
			rowPlus2 = _mm_cmplt_epi16(rowPlus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate

			int maskSkip1 = _mm_movemask_epi8(rowMinus2);
			rowMinus2 = _mm_loadu_si128((__m128i *)(pLocalSrc - (srcStride + srcStride) - 2));

			maskSkip1 &= _mm_movemask_epi8(rowPlus2);									// 1 if both 0 and 8 are within 't' of the candidate pixel
			rowPlus2 = _mm_loadu_si128((__m128i *)(pLocalSrc + (srcStride + srcStride) - 2));

			maskSkip |= maskSkip1;

			// Check for corners in the eight pixels
			if (maskSkip != 0xFFFF)
			{
				for (int i = 0; i < 8; i++)
				{
					__m128i * tbl = (__m128i *) dataFastCornersPixelMask;

					if (!(maskSkip & 1))
					{
						// Get the boundary pixels in an XMM register
						__m128i mask = _mm_load_si128(tbl++);
						__m128i boundary = _mm_shuffle_epi8(rowMinus3, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowMinus2, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowMinus1, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(row, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowPlus1, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowPlus2, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowPlus3, mask);
						boundary = _mm_or_si128(boundary, mask);

						if (isCorner_SSE(M128I(row).m128i_u8[3], boundary, thresh))
						{
							if (cornerCount < capacityOfDstCorner)
							{
								dstCorner[cornerCount].y = height + 3;
								dstCorner[cornerCount].x = width + i;
								dstCorner[cornerCount].strength = strength_threshold;			// Undefined as per the 1.0.1 spec
								dstCorner[cornerCount].scale = 0;
								dstCorner[cornerCount].orientation = 0;
								dstCorner[cornerCount].error = 0;
								dstCorner[cornerCount++].tracking_status = 1;
							}
							else
								cornerCount++;
						}
					}
					maskSkip >>= 2;
					rowMinus3 = _mm_srli_si128(rowMinus3, 1);
					rowMinus2 = _mm_srli_si128(rowMinus2, 1);
					rowMinus1 = _mm_srli_si128(rowMinus1, 1);
					row = _mm_srli_si128(row, 1);
					rowPlus1 = _mm_srli_si128(rowPlus1, 1);
					rowPlus2 = _mm_srli_si128(rowPlus2, 1);
					rowPlus3 = _mm_srli_si128(rowPlus3, 1);
				}
			}

			width += 8;
			pLocalSrc += 8;
		}

		for (int x = 0; x < postfixWidth; x++)
		{
			int masks[2];
			generateMasks_C(pLocalSrc, srcStride, neighbor_offset, t, masks);
			if (isCorner(masks))
			{
				if (cornerCount < capacityOfDstCorner)
				{
					dstCorner[cornerCount].y = height + 3;
					dstCorner[cornerCount].x = width;
					dstCorner[cornerCount].strength = strength_threshold;			// Undefined as per the 1.0.1 spec
					dstCorner[cornerCount].scale = 0;
					dstCorner[cornerCount].orientation = 0;
					dstCorner[cornerCount].error = 0;
					dstCorner[cornerCount++].tracking_status = 1;
				}
				else
					cornerCount++;
			}
			width++;
			pLocalSrc++;
		}
		pSrcImage += srcStride;
	}

	*pDstCornerCount = cornerCount;
	return AGO_SUCCESS;
}

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
	)
{
	unsigned char * pLocalSrc;
	int srcStride = (int)srcImageStrideInBytes;
	vx_uint32 cornerCount = 0;
	short t = (short)floorf(strength_threshold);

	pSrcImage += (srcStride * 3) + 3;														// Leave first three rows and start from the third pixel

	int alignedWidth = (int)(srcWidth - 6) & ~7;
	int postfixWidth = (int)(srcWidth - 6) & 7;

	// Generate offsets for C code if necessary
	int neighbor_offset[16] = { 0 };
	if (postfixWidth)
		generateOffset(srcStride, neighbor_offset);

	memset(pScratch, 0, sizeof(vx_uint8) * srcWidth * srcHeight);

	for (int height = 0; height < (int)(srcHeight - 6); height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 3;

		for (int x = 0; x < (alignedWidth >> 3); x++)
		{
			__m128i rowMinus3, rowMinus2, rowMinus1, row, rowPlus1, rowPlus2, rowPlus3;
			__m128i thresh = _mm_set1_epi16(t);

			// Check for early escape based on pixels 1 and 9 around the candidate
			rowMinus3 = _mm_loadu_si128((__m128i *)(pLocalSrc - 3 * srcStride - 1));
			rowMinus2 = _mm_srli_si128(rowMinus3, 1);									// row - 3: Pixels 0..7 in lower 7 bytes
			rowMinus2 = _mm_cvtepu8_epi16(rowMinus2);

			row = _mm_loadu_si128((__m128i *)(pLocalSrc - 3));
			rowMinus1 = _mm_srli_si128(row, 3);											// row: Pixels 0..7 in lower 7 bytes
			rowMinus1 = _mm_cvtepu8_epi16(rowMinus1);

			rowPlus3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 3 * srcStride - 1));
			rowPlus2 = _mm_srli_si128(rowPlus3, 1);										// row + 3: Pixels 0..7 in lower 7 bytes
			rowPlus2 = _mm_cvtepu8_epi16(rowPlus2);

			rowPlus1 = _mm_loadu_si128((__m128i *)(pLocalSrc + srcStride - 3));

			rowMinus2 = _mm_sub_epi16(rowMinus2, rowMinus1);
			rowMinus2 = _mm_abs_epi16(rowMinus2);
			rowPlus2 = _mm_sub_epi16(rowPlus2, rowMinus1);
			rowPlus2 = _mm_abs_epi16(rowPlus2);

			rowMinus2 = _mm_cmplt_epi16(rowMinus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate
			rowPlus2 = _mm_cmplt_epi16(rowPlus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate

			int maskSkip = _mm_movemask_epi8(rowMinus2);
			maskSkip &= _mm_movemask_epi8(rowPlus2);									// 1 if both 0 and 8 are within 't' of the candidate pixel

			// Check for early escape based on pixels 12 and 4 around the candidate
			rowMinus2 = _mm_cvtepu8_epi16(row);
			rowPlus2 = _mm_srli_si128(row, 6);
			rowPlus2 = _mm_cvtepu8_epi16(rowPlus2);

			rowMinus2 = _mm_sub_epi16(rowMinus2, rowMinus1);
			rowMinus2 = _mm_abs_epi16(rowMinus2);
			rowPlus2 = _mm_sub_epi16(rowPlus2, rowMinus1);
			rowPlus2 = _mm_abs_epi16(rowPlus2);

			rowMinus1 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcStride - 3));

			rowMinus2 = _mm_cmplt_epi16(rowMinus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate
			rowPlus2 = _mm_cmplt_epi16(rowPlus2, thresh);								// Check if pixel 0 is less than 't' different from the candidate

			int maskSkip1 = _mm_movemask_epi8(rowMinus2);
			rowMinus2 = _mm_loadu_si128((__m128i *)(pLocalSrc - (srcStride + srcStride) - 2));

			maskSkip1 &= _mm_movemask_epi8(rowPlus2);									// 1 if both 0 and 8 are within 't' of the candidate pixel
			rowPlus2 = _mm_loadu_si128((__m128i *)(pLocalSrc + (srcStride + srcStride) - 2));

			maskSkip |= maskSkip1;

			// Check for corners in the eight pixels
			if (maskSkip != 0xFFFF)
			{
				for (int i = 0; i < 8; i++)
				{
					__m128i * tbl = (__m128i *) dataFastCornersPixelMask;

					if (!(maskSkip & 1))
					{
						// Get the boundary pixels in an XMM register
						__m128i mask = _mm_load_si128(tbl++);
						__m128i boundary = _mm_shuffle_epi8(rowMinus3, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowMinus2, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowMinus1, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(row, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowPlus1, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowPlus2, mask);
						boundary = _mm_or_si128(boundary, mask);

						mask = _mm_load_si128(tbl++);
						mask = _mm_shuffle_epi8(rowPlus3, mask);
						boundary = _mm_or_si128(boundary, mask);

						short strength = 0;
						if (checkForCornerAndGetStrength_SSE(M128I(row).m128i_u8[3], boundary, t, &strength))
							pScratch[(height + 3) * srcWidth + width + i] = (vx_uint8)strength;
					}
					maskSkip >>= 2;
					rowMinus3 = _mm_srli_si128(rowMinus3, 1);
					rowMinus2 = _mm_srli_si128(rowMinus2, 1);
					rowMinus1 = _mm_srli_si128(rowMinus1, 1);
					row = _mm_srli_si128(row, 1);
					rowPlus1 = _mm_srli_si128(rowPlus1, 1);
					rowPlus2 = _mm_srli_si128(rowPlus2, 1);
					rowPlus3 = _mm_srli_si128(rowPlus3, 1);
				}
			}

			width += 8;
			pLocalSrc += 8;
		}

		for (int x = 0; x < postfixWidth; x++)
		{
			short strength = 0;
			if (checkForCornerAndGetStrength(pLocalSrc, neighbor_offset, t, &strength))
				pScratch[(height + 3) * srcWidth + width] = (vx_uint8)strength;

			width++;
			pLocalSrc++;
		}
		pSrcImage += srcStride;
	}

	// Non-max supression
	pScratch += (3 * srcWidth + 3);
	cornerCount = 0;
	for (int height = 0; height < int(srcHeight - 6); height++)
	{
		for (int width = 0; width < int(srcWidth - 6); width++)
		{
			vx_uint8 * prev = pScratch - srcWidth;
			vx_uint8 * nxt = pScratch + srcWidth;
			vx_uint8 cand = *pScratch;
			if (cand && (cand >= *(prev - 1)) && (cand >= *prev) && (cand >= *(prev + 1))
				&& (cand >= *(pScratch - 1)) && (cand > *(pScratch + 1))
				&& (cand > *(nxt - 1)) && (cand > *nxt) && (cand > *(nxt + 1)))
			{
				if (cornerCount < capacityOfDstCorner)
				{
					dstCorner[cornerCount].x = (vx_int32)(width + 3);
					dstCorner[cornerCount].y = (vx_int32)(height + 3);
					dstCorner[cornerCount].strength = (vx_float32)cand;
					dstCorner[cornerCount].scale = 0;
					dstCorner[cornerCount].orientation = 0;
					dstCorner[cornerCount].error = 0;
					dstCorner[cornerCount++].tracking_status = 1;
				}
				else
					cornerCount++;
			}
			pScratch++;
		}
		pScratch += 6;
	}
	*pDstCornerCount = cornerCount;
	return AGO_SUCCESS;
}


int HafCpu_FastCornerMerge_XY_XY
	(
		vx_uint32       capacityOfDstCorner,
		vx_keypoint_t   dstCorner[],
		vx_uint32     * pDstCornerCount,
		vx_uint32		numSrcCornerBuffers,
		vx_keypoint_t * pSrcCorners[],
		vx_uint32       numSrcCorners[]
	)
{
	int dstCount = 0;
	int srcCount;
	vx_keypoint_t * srcList;

	for (int i = 0; i < (int)numSrcCornerBuffers; i++)
	{
		srcCount = numSrcCorners[i];
		srcList = pSrcCorners[i];

		while (srcCount)
		{
			*dstCorner++ = *srcList++;
			dstCount++;
			srcCount--;
			if (dstCount >(int) capacityOfDstCorner)
			{
				*pDstCornerCount = (vx_uint32)(dstCount - 1);
				return AGO_SUCCESS;
			}
		}
	}

	*pDstCornerCount = (vx_uint32)(dstCount - 1);
	return AGO_SUCCESS;
}
