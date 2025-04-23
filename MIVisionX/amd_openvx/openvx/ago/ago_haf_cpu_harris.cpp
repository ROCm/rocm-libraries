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

typedef struct {
	vx_float32 GxGx;
	vx_float32 GxGy;
	vx_float32 GyGy;
} ago_harris_Gxy_t;

void insertAtLocation(vx_uint32 listCapacity, vx_keypoint_t * pList, vx_uint32 * cornerCount, vx_keypoint_t itemToBeAdded, vx_uint32 loc)
{
	vx_keypoint_t incoming_keypt = itemToBeAdded;
	vx_keypoint_t temp;

	for (int i = (int)loc; i <= (int)*cornerCount; i++)
	{
		temp = pList[i];
		pList[i] = incoming_keypt;
		incoming_keypt = temp;
	}

	*cornerCount = *cornerCount + 1;
}

void AddToTheSortedKeypointList(vx_uint32 listCapacity, vx_keypoint_t * pList, vx_uint32 * cornerCount, vx_keypoint_t itemToBeAdded)
{
	if (*cornerCount == 0)									// Add the item to the head
	{
		pList[0] = itemToBeAdded;
		*cornerCount = 1;
	}
	else
	{
		if (itemToBeAdded.strength <= pList[*cornerCount - 1].strength)
		{
			if (*cornerCount == listCapacity)
				return;
			else
			{
				pList[*cornerCount] = itemToBeAdded;
				*cornerCount = *cornerCount + 1;
			}
		}
		else
		{
			int idx = 0;
			while (pList[idx].strength > itemToBeAdded.strength)
				idx++;
			insertAtLocation(listCapacity, pList, cornerCount, itemToBeAdded, idx);
		}
	}
}

// Using Separable filter:
// For Gx:
//	-1	0	1		-1	0	1		1
//	-2	0	2	=					2
//	-1	0	1						1
// For Gy:
//	-1	-2	-1		1	2	1		-1
//	 0	 0	 0	=					 0
//	 1	 2	 1						 1
int HafCpu_HarrisSobel_HG3_U8_3x3
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy_,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	)
{
	ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)((vx_uint8 *) pDstGxy_ + dstGxyStrideInBytes);
	
	int tmpWidth = (dstWidth + 15) & ~15;
	tmpWidth <<= 1;
	vx_int16 * pPrevRow = (vx_int16*)pScratch;
	vx_int16 * pCurrRow = ((vx_int16*)pScratch) + tmpWidth;
	vx_int16 * pNextRow = ((vx_int16*)pScratch) + (tmpWidth + tmpWidth);

	vx_int16 * pLocalPrevRow = pPrevRow;
	vx_int16 * pLocalCurrRow = pCurrRow;
	vx_int16 * pLocalNextRow = pNextRow;

	// Horizontal filtering for the first row - row 0
	vx_uint8 * pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];
	}

	// Horizontal filtering for the second row - row 1
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];
	}

	pSrcImage += srcImageStrideInBytes;
	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;

	vx_float32 div_factor = 1; // 4.0f * 255;

	// Process rows 2 until end
	for(int y = 0; y < (int) dstHeight - 2; y++)
	{
		pLocalSrc = pSrcImage;
		ago_harris_Gxy_t * pLocalDst = pDstGxy;
		for(int x = 0; x < (int) dstWidth; x++)
		{
			vx_int16 gx, gy;
			gx = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
			gy = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];

			*pLocalNextRow++ = gx;
			*pLocalNextRow++ = gy;

			gx += *pLocalPrevRow++ + (*pLocalCurrRow++ << 1);
			gy -= *pLocalPrevRow++;
			pLocalCurrRow++;

			pLocalDst->GxGx = ((vx_float32)gx * (vx_float32)gx) / div_factor;
			pLocalDst->GxGy = ((vx_float32)gx * (vx_float32)gy) / div_factor;
			pLocalDst->GyGy = ((vx_float32)gy * (vx_float32)gy) / div_factor;
			
			pLocalDst++;
			pLocalSrc++;
		}

		vx_int16 * pTemp = pPrevRow;
		pPrevRow = pCurrRow;
		pCurrRow = pNextRow;
		pNextRow = pTemp;

		pLocalPrevRow = pPrevRow;
		pLocalCurrRow = pCurrRow;
		pLocalNextRow = pNextRow;

		pSrcImage += srcImageStrideInBytes;
		pDstGxy += (dstGxyStrideInBytes / sizeof(ago_harris_Gxy_t));
	}


#if 0
	pSrcImage += srcImageStrideInBytes;										// First row not processed
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	__declspec(align(16)) short r0[3840 * 2], r1[3840 * 2], r2[3840 * 2];	// Intermideate buffers to store results between horizontally filtered rows - [GxL GxH GyL GyH]

	__m128i * pPrevRow = (__m128i *) r0;
	__m128i * pCurrRow = (__m128i *) r1;
	__m128i * pNextRow = (__m128i *) r2;

	__m128i row0, temp0, temp1, temp2, temp3, Gx, Gy;
	__m128i zeromask = _mm_setzero_si128();

	__m128i * pLocalPrevRow = pPrevRow;
	__m128i * pLocalCurrRow = pCurrRow;
	__m128i * pLocalNextRow = pNextRow;
	//__m128i * pTemp;

	int alignedWidth = dstWidth & ~15;								// Sixteen pixels processed in a go for first two rows
	int postfixWidth = dstWidth & 15;
	int srcStride = (int)srcImageStrideInBytes;

	// Process first two rows
	// Process first two rows - Horizontal filtering
	for (int x = 0; x < (int)(alignedWidth >> 4); x++)
	{
		__m128i shiftedR, shiftedL;

		// row above
		row0 = _mm_load_si128((__m128i *)(pLocalSrc - srcStride));
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcStride - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcStride + 1));

		temp0 = _mm_unpackhi_epi8(row0, zeromask);
		temp0 = _mm_slli_epi16(temp0, 1);								// GyH: 2 * (0,-1)
		Gy = _mm_cvtepu8_epi16(row0);
		Gy = _mm_slli_epi16(Gy, 1);										// GyL: 2 * (0,-1)

		Gx = _mm_cvtepu8_epi16(shiftedL);								// GxL: -1 * (-1,-1)	GyL: 1 * (-1,-1)
		temp1 = _mm_unpackhi_epi8(shiftedL, zeromask);					// GxH: -1 * (-1,-1)	GyH: 1 * (-1,-1)
		temp1 = _mm_add_epi16(temp0, temp1);
		Gy = _mm_add_epi16(Gy, Gx);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// GxH: 1 * (1,-1)		GyH: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// GxL: 1 * (1,-1)		GyL: 1 * (1,-1)
		temp1 = _mm_sub_epi16(shiftedL, temp1);
		Gx = _mm_sub_epi16(shiftedR, Gx);
		temp0 = _mm_add_epi16(temp0, shiftedL);
		Gy = _mm_add_epi16(Gy, shiftedR);

		_mm_store_si128(pLocalPrevRow++, Gx);
		_mm_store_si128(pLocalPrevRow++, temp1);
		_mm_store_si128(pLocalPrevRow++, Gy);
		_mm_store_si128(pLocalPrevRow++, temp0);

		// current row
		row0 = _mm_load_si128((__m128i *)pLocalSrc);
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

		temp0 = _mm_unpackhi_epi8(row0, zeromask);
		temp0 = _mm_slli_epi16(temp0, 1);								// GyH: 2 * (-1, 0)
		Gy = _mm_cvtepu8_epi16(row0);
		Gy = _mm_slli_epi16(Gy, 1);										// GyL: 2 * (-1, 0)

		Gx = _mm_cvtepu8_epi16(shiftedL);								// GxL: -1 * (-1,-1)	GyL: 1 * (-1,-1)
		temp1 = _mm_unpackhi_epi8(shiftedL, zeromask);					// GxH: -1 * (-1,-1)	GyH: 1 * (-1,-1)
		temp0 = _mm_add_epi16(temp0, temp1);
		Gy = _mm_add_epi16(Gy, Gx);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// GxH: 1 * (1,-1)		GyH: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// GxL: 1 * (1,-1)		GyL: 1 * (1,-1)
		temp1 = _mm_sub_epi16(shiftedL, temp1);
		Gx = _mm_sub_epi16(shiftedR, Gx);
		temp0 = _mm_add_epi16(temp0, shiftedL);
		Gy = _mm_add_epi16(Gy, shiftedR);

		_mm_store_si128(pLocalCurrRow++, Gx);
		_mm_store_si128(pLocalCurrRow++, temp1);
		_mm_store_si128(pLocalCurrRow++, Gy);
		_mm_store_si128(pLocalCurrRow++, temp0);

		pLocalSrc += 16;
	}

	short * pShort_Prev = (short *)pLocalPrevRow;
	short * pShort_Curr = (short *)pLocalCurrRow;
	for (int x = 0; x < postfixWidth; x++)
	{
		// Row above
		*pShort_Prev++ = (short)pLocalSrc[-srcStride + 1] - (short)pLocalSrc[-srcStride - 1];										// Gx
		*pShort_Prev++ = (short)pLocalSrc[-srcStride + 1] + (short)pLocalSrc[-srcStride] + (short)pLocalSrc[-srcStride - 1];		// Gy

		// Current row
		*pShort_Curr++ = (short)pLocalSrc[1] - (short)pLocalSrc[-1];										// Gx
		*pShort_Curr++ = (short)pLocalSrc[1] + (short)pLocalSrc[0] + (short)pLocalSrc[-1];					// Gy
	}
	
	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;
	pLocalNextRow = pNextRow;

	// Process rows 3 till the end
	int height = (int)(dstHeight - 2);
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcImageStrideInBytes);				// Pointing to the row below

		int width = (int)(alignedWidth >> 3);											// Eight pixels processed in a go
		while (width)
		{
			__m128i prevRowFiltered, currRowFiltered;

			// Horizontal filtering - next row
			row0 = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			Gx = _mm_cvtepu8_epi16(row0);												// 1 * (-1,1)
			Gy = _mm_add_epi16(Gx, zeromask);											// 1 * (-1,1)

			prevRowFiltered = _mm_load_si128(pLocalPrevRow++);

			row0 = _mm_srli_si128(row0, 1);
			temp0 = _mm_cvtepu8_epi16(row0);
			temp0 = _mm_slli_epi16(temp0, 1);											// 2 * (0,1)
			Gy = _mm_add_epi16(Gy, temp0);

			currRowFiltered = _mm_load_si128(pLocalCurrRow++);

			row0 = _mm_srli_si128(row0, 1);
			temp0 = _mm_cvtepu8_epi16(row0);											// 1 * (1,1)
			Gx = _mm_sub_epi16(temp0, Gx);
			Gy = _mm_add_epi16(Gy, temp0);

			currRowFiltered = _mm_slli_epi16(currRowFiltered, 1);						// 2 * filteredCurrRow
			Gx = _mm_add_epi16(Gx, currRowFiltered);

			Gx = _mm_add_epi16(Gx, prevRowFiltered);									// Gx0 Gx1 Gx2 Gx3 Gx4 Gx5 Gx6 Gx7
			Gy = _mm_subs_epi16(Gy, prevRowFiltered);									// Gy0 Gy1 Gy2 Gy3 Gy4 Gy5 Gy6 Gy7

			prevRowFiltered = _mm_cvtepi16_epi32(Gx);									// Gx0 Gx1 Gx2 Gx3
			currRowFiltered = _mm_cvtepi16_epi32(Gy);									// Gy0 Gy1 Gy2 Gy3

			temp0 = _mm_shuffle_epi32(prevRowFiltered, 64);								// Gx0 Gx0 Gx0 Gx1
			temp1 = _mm_shuffle_epi32(currRowFiltered, 64);								// Gy0 Gy0 Gy0 Gy1

			temp2 = _mm_blend_epi16(temp0, temp1, 0x10);								// Gx0 Gx0 Gy0 Gx1
			temp3 = _mm_blend_epi32(temp0, temp1, 0x14);								// Gx0 Gy0 Gy0 Gx1

			
			width--;

		}
		height--;
	}
#endif
	return AGO_SUCCESS;
}

// Using separable filter
//			-1	-2	0	2	1			1
//										4
//  Gx =								6
//										4
//										1
int HafCpu_HarrisSobel_HG3_U8_5x5
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy_,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	)
{
	ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)((vx_uint8 *)pDstGxy_ + 2*dstGxyStrideInBytes);

	int tmpWidth = (dstWidth + 15) & ~15;
	tmpWidth <<= 1;
	vx_int16 * pRowMinus2 = (vx_int16*)pScratch;
	vx_int16 * pRowMinus1 = ((vx_int16*)pScratch) + tmpWidth;
	vx_int16 * pRowCurr = ((vx_int16*)pScratch) + (2*tmpWidth);
	vx_int16 * pRowPlus1 = ((vx_int16*)pScratch) + (3*tmpWidth);
	vx_int16 * pRowPlus2 = ((vx_int16*)pScratch) + (4*tmpWidth);

	vx_int16 * pLocalRowMinus2 = pRowMinus2;
	vx_int16 * pLocalRowMinus1 = pRowMinus1;
	vx_int16 * pLocalRowCurr = pRowCurr;
	vx_int16 * pLocalRowPlus1 = pRowPlus1;
	vx_int16 * pLocalRowPlus2 = pRowPlus2;

	// Horizontal filtering for the first row - row 0
	vx_uint8 * pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus2++ = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
		*pLocalRowMinus2++ = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);
	}

	// Horizontal filtering for the second row - row 1
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus1++ = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
		*pLocalRowMinus1++ = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);
	}

	// Horizontal filtering for the second row - row 2
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowCurr++ = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
		*pLocalRowCurr++ = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);
	}

	// Horizontal filtering for the second row - row 3
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowPlus1++ = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
		*pLocalRowPlus1++ = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);
	}

	pSrcImage += srcImageStrideInBytes;
	
	pLocalRowMinus2 = pRowMinus2;
	pLocalRowMinus1 = pRowMinus1;
	pLocalRowCurr = pRowCurr;
	pLocalRowPlus1 = pRowPlus1;

	// Process rows 4 until end
	for (int y = 0; y < (int)dstHeight - 4; y++)
	{
		pLocalSrc = pSrcImage;
		for (int x = 0; x < (int)dstWidth; x++)
		{
			vx_int16 gx, gy;
			
			gx = (vx_int16)pLocalSrc[2] - (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1);
			gy = (vx_int16)pLocalSrc[2] + (vx_int16)pLocalSrc[-2] + (((vx_int16)pLocalSrc[1] + (vx_int16)pLocalSrc[0] + (vx_int16)pLocalSrc[-1]) << 2) + ((vx_int16)pLocalSrc[0] << 1);

			*pLocalRowPlus2++ = gx;
			*pLocalRowPlus2++ = gy;

			gx += *pLocalRowMinus2++ + ((*pLocalRowMinus1++ + *pLocalRowCurr + *pLocalRowPlus1++) << 2) + (*pLocalRowCurr << 1);
			gy += ((*pLocalRowPlus1++ - *pLocalRowMinus1++) << 1) - *pLocalRowMinus2++;
			pLocalRowCurr += 2;

			pDstGxy->GxGx = ((vx_float32)gx * (vx_float32)gx);	// / 16.0f;
			pDstGxy->GxGy = ((vx_float32)gx * (vx_float32)gy);	// / 16.0f;
			pDstGxy->GyGy = ((vx_float32)gy * (vx_float32)gy);	// / 16.0f;

			pDstGxy++;
			pLocalSrc++;
		}

		vx_int16 * pTemp = pRowMinus2;
		pRowMinus2 = pRowMinus1;
		pRowMinus1 = pRowCurr;
		pRowCurr = pRowPlus1;
		pRowPlus1 = pRowPlus2;
		pRowPlus2 = pTemp;

		pLocalRowMinus2 = pRowMinus2;
		pLocalRowMinus1 = pRowMinus1;
		pLocalRowCurr = pRowCurr;
		pLocalRowPlus1 = pRowPlus1;
		pLocalRowPlus2 = pRowPlus2;

		pSrcImage += srcImageStrideInBytes;
	}
	
	return AGO_SUCCESS;
}

// Using separable filter
//				-1	-4	-5	0	5	4	1			1
//													6
//													15
//		Gx =										20	
//													15
//													6
//													1
int HafCpu_HarrisSobel_HG3_U8_7x7
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstGxy_,
		vx_uint32          dstGxyStrideInBytes,
		vx_uint8         * pSrcImage,
		vx_uint32          srcImageStrideInBytes,
		vx_uint8		 * pScratch
	)
{
	ago_harris_Gxy_t * pDstGxy = (ago_harris_Gxy_t *)((vx_uint8 *)pDstGxy_ + 3*dstGxyStrideInBytes);

	int tmpWidth = (dstWidth + 15) & ~15;
	tmpWidth <<= 1;
	vx_int32 * pRowMinus3 = (vx_int32*)pScratch;
	vx_int32 * pRowMinus2 = ((vx_int32*)pScratch) + tmpWidth;
	vx_int32 * pRowMinus1 = ((vx_int32*)pScratch) + (2 * tmpWidth);
	vx_int32 * pRowCurr = ((vx_int32*)pScratch) + (3 * tmpWidth);
	vx_int32 * pRowPlus1 = ((vx_int32*)pScratch) + (4 * tmpWidth);
	vx_int32 * pRowPlus2 = ((vx_int32*)pScratch) + (5 * tmpWidth);
	vx_int32 * pRowPlus3 = ((vx_int32*)pScratch) + (6 * tmpWidth);

	vx_int32 * pLocalRowMinus3 = pRowMinus3;
	vx_int32 * pLocalRowMinus2 = pRowMinus2;
	vx_int32 * pLocalRowMinus1 = pRowMinus1;
	vx_int32 * pLocalRowCurr = pRowCurr;
	vx_int32 * pLocalRowPlus1 = pRowPlus1;
	vx_int32 * pLocalRowPlus2 = pRowPlus2;
	vx_int32 * pLocalRowPlus3 = pRowPlus3;

	// Horizontal filtering for the first row - row 0
	vx_uint8 * pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus3++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowMinus3++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 1
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus2++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowMinus2++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 2
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowMinus1++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowMinus1++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 3
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowCurr++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowCurr++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 4
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowPlus1++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowPlus1++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	// Horizontal filtering for the second row - row 5
	pSrcImage += srcImageStrideInBytes;
	pLocalSrc = pSrcImage;
	for (int x = 0; x < (int)dstWidth; x++, pLocalSrc++)
	{
		*pLocalRowPlus2++ = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
		*pLocalRowPlus2++ = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);
	}

	pSrcImage += srcImageStrideInBytes;

	pLocalRowMinus3 = pRowMinus3;
	pLocalRowMinus2 = pRowMinus2;
	pLocalRowMinus1 = pRowMinus1;
	pLocalRowCurr = pRowCurr;
	pLocalRowPlus1 = pRowPlus1;
	pLocalRowPlus2 = pRowPlus2;

	// Process rows 4 until end
	for (int y = 0; y < (int)dstHeight - 6; y++)
	{
		pLocalSrc = pSrcImage;
		for (int x = 0; x < (int)dstWidth; x++)
		{
			vx_int32 gx, gy;

			gx = (vx_int32)pLocalSrc[3] - (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] - (vx_int32)pLocalSrc[-2]) << 2) + (((vx_int32)pLocalSrc[1] - (vx_int32)pLocalSrc[-1]) * 5);
			gy = (vx_int32)pLocalSrc[3] + (vx_int32)pLocalSrc[-3] + (((vx_int32)pLocalSrc[2] + (vx_int32)pLocalSrc[-2]) * 6) + (((vx_int32)pLocalSrc[1] + (vx_int32)pLocalSrc[-1]) * 15) + ((vx_int32)pLocalSrc[0] * 20);

			*pLocalRowPlus3++ = gx;
			*pLocalRowPlus3++ = gy;

			gx += *pLocalRowMinus3++ + ((*pLocalRowMinus2++ + *pLocalRowPlus2++) * 6) + ((*pLocalRowMinus1++ + *pLocalRowPlus1++) * 15) + (*pLocalRowCurr++ * 20);
			gy += ((*pLocalRowPlus2++ - *pLocalRowMinus2++) << 2) + ((*pLocalRowPlus1++ - *pLocalRowMinus1++) * 5) - *pLocalRowMinus3++;
			pLocalRowCurr++;

			pDstGxy->GxGx = ((vx_float32)gx * (vx_float32)gx);	// / 64.0f;
			pDstGxy->GxGy = ((vx_float32)gx * (vx_float32)gy);	// / 64.0f;
			pDstGxy->GyGy = ((vx_float32)gy * (vx_float32)gy);	// / 64.0f;
			pDstGxy++;
			pLocalSrc++;
		}

		vx_int32 * pTemp = pRowMinus3;
		pRowMinus3 = pRowMinus2;
		pRowMinus2 = pRowMinus1;
		pRowMinus1 = pRowCurr;
		pRowCurr = pRowPlus1;
		pRowPlus1 = pRowPlus2;
		pRowPlus2 = pRowPlus3;
		pRowPlus3 = pTemp;

		pLocalRowMinus3 = pRowMinus3;
		pLocalRowMinus2 = pRowMinus2;
		pLocalRowMinus1 = pRowMinus1;
		pLocalRowCurr = pRowCurr;
		pLocalRowPlus1 = pRowPlus1;
		pLocalRowPlus2 = pRowPlus2;
		pLocalRowPlus3 = pRowPlus3;

		pSrcImage += srcImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

int HafCpu_HarrisScore_HVC_HG3_3x3
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy_,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	)
{
	ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
	vx_float32 Tc = strength_threshold;
	vx_int32 srcStride = srcGxyStrideInBytes / sizeof(ago_harris_Gxy_t);
	vx_int32 dstStride = dstVcStrideInBytes / sizeof(vx_float32);
	pSrcGxy += srcStride;															// Skip first row
	memset(pDstVc, 0, dstVcStrideInBytes);											// Zero the thresholds of first row
	pDstVc += dstStride;

	for (int y = 1; y < (int)dstHeight - 1; y++)
	{
		ago_harris_Gxy_t * pLocalSrc = pSrcGxy;
		vx_float32 * pLocalDst = pDstVc;

		*pLocalDst = 0;															// First column Vc = 0;
		pLocalDst++;
		pLocalSrc++;
		for (int x = 1; x < (int)dstWidth - 1; x++)
		{
			vx_float32 gx2 = 0;
			vx_float32 gy2 = 0;
			vx_float32 gxy2 = 0;

			// Windowing
			for (int j = -1; j <= 1; j++)
			{
				ago_harris_Gxy_t * pTemp = pLocalSrc + j * srcStride;
				for (int i = -1; i <= 1; i++)
				{
					gx2 += pTemp[i].GxGx;
					gxy2 += pTemp[i].GxGy;
					gy2 += pTemp[i].GyGy;
				}
			}

			vx_float32 traceA = gx2 + gy2;
			vx_float32 detA = (gx2 * gy2) - (gxy2 * gxy2);
			vx_float32 Mc = detA - (sensitivity * traceA * traceA);
			Mc /= normalization_factor;
			*pLocalDst = (Mc > Tc) ? Mc : 0;

			pLocalSrc++;
			pLocalDst++;
		}

		*pLocalDst = 0;															// Last column Vc = 0;
		pSrcGxy += srcStride;
		pDstVc += dstStride;
	}
	memset(pDstVc, 0, dstVcStrideInBytes);											// Zero the thresholds of last row
	return AGO_SUCCESS;
}

int HafCpu_HarrisScore_HVC_HG3_5x5
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy_,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	)
{
	ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
	vx_float32 Tc = strength_threshold;
	vx_int32 srcStride = srcGxyStrideInBytes / sizeof(ago_harris_Gxy_t);
	vx_int32 dstStride = dstVcStrideInBytes / sizeof(vx_float32);
	pSrcGxy += (srcStride + srcStride);									// Skip first two rows
	memset(pDstVc, 0, dstVcStrideInBytes + dstVcStrideInBytes);			// Zero the thresholds of first two rows
	pDstVc += (dstStride + dstStride);

	for (int y = 2; y < (int)dstHeight - 2; y++)
	{
		ago_harris_Gxy_t * pLocalSrc = pSrcGxy;
		vx_float32  * pLocalDst = pDstVc;

		*pLocalDst = 0;															// First column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// Second column Vc = 0;
		pLocalDst++;
		pLocalSrc += 2;

		for (int x = 2; x < (int)dstWidth - 2; x++)
		{
			vx_float32 gx2 = 0;
			vx_float32 gy2 = 0;
			vx_float32 gxy2 = 0;

			// Windowing
			for (int j = -2; j <= 2; j++)
			{
				for (int i = -2; i <= 2; i++)
				{
					gx2 += pLocalSrc[j * srcStride + i].GxGx;
					gxy2 += pLocalSrc[j * srcStride + i].GxGy;
					gy2 += pLocalSrc[j * srcStride + i].GyGy;
				}
			}

			vx_float32 traceA = gx2 + gy2;
			vx_float32 detA = (gx2 * gy2) - (gxy2 * gxy2);
			vx_float32 Mc = detA - (sensitivity * traceA * traceA);
			Mc /= normalization_factor;
			*pLocalDst = (Mc > Tc) ? Mc : 0;

			pLocalSrc++;
			pLocalDst++;
		}

		*pLocalDst = 0;															// second to last column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// last column Vc = 0;

		pSrcGxy += srcStride;
		pDstVc += dstStride;
	}
	memset(pDstVc, 0, dstVcStrideInBytes + dstVcStrideInBytes);					// Zero the thresholds of last rows
	return AGO_SUCCESS;
}

int HafCpu_HarrisScore_HVC_HG3_7x7
	(
		vx_uint32          dstWidth,
		vx_uint32          dstHeight,
		vx_float32       * pDstVc,
		vx_uint32          dstVcStrideInBytes,
		vx_float32       * pSrcGxy_,
		vx_uint32          srcGxyStrideInBytes,
		vx_float32         sensitivity,
		vx_float32         strength_threshold,
		vx_float32		   normalization_factor
	)
{
	ago_harris_Gxy_t * pSrcGxy = (ago_harris_Gxy_t *)pSrcGxy_;
	vx_float32 Tc = strength_threshold;
	vx_int32 srcStride = srcGxyStrideInBytes / sizeof(ago_harris_Gxy_t);
	vx_int32 dstStride = dstVcStrideInBytes / sizeof(vx_float32);
	pSrcGxy += (srcStride * 3);																// Skip first three rows
	memset(pDstVc, 0, dstVcStrideInBytes * 3);												// Zero the thresholds of first three rows
	pDstVc += (dstStride * 3);

	for (int y = 3; y < (int)dstHeight - 3; y++)
	{
		ago_harris_Gxy_t * pLocalSrc = pSrcGxy;
		vx_float32 * pLocalDst = pDstVc;

		*pLocalDst = 0;															// First column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// Second column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// Third column Vc = 0;
		pLocalSrc += 3;

		for (int x = 3; x < (int)dstWidth - 3; x++)
		{
			vx_float32 gx2 = 0;
			vx_float32 gy2 = 0;
			vx_float32 gxy2 = 0;

			// Windowing
			for (int j = -3; j <= 3; j++)
			{
				for (int i = -3; i <= 3; i++)
				{
					gx2 += pLocalSrc[j * srcStride + i].GxGx;
					gxy2 += pLocalSrc[j * srcStride + i].GxGy;
					gy2 += pLocalSrc[j * srcStride + i].GyGy;
				}
			}

			vx_float32 traceA = gx2 + gy2;
			vx_float32 detA = (gx2 * gy2) - (gxy2 * gxy2);
			vx_float32 Mc = detA - (sensitivity * traceA * traceA);
			Mc /= normalization_factor;
			*pLocalDst = (Mc > Tc) ? Mc : 0;

			pLocalSrc++;
			pLocalDst++;
		}

		*pLocalDst = 0;															// third to last column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// second to last column Vc = 0;
		pLocalDst++;
		*pLocalDst = 0;															// last column Vc = 0;

		pSrcGxy += srcStride;
		pDstVc += dstStride;
	}
	memset(pDstVc, 0, dstVcStrideInBytes * 3);											// Zero the thresholds of last rows
	return AGO_SUCCESS;
}

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
	)
{
	vx_float32      * pLocalSrc;
	vx_float32      * pSrcVc_NMS = pSrcVc;
	vx_int32 radius = (vx_int32) min_distance;

	// Non max supression
	for (vx_int32 y = 0; y < (vx_int32)srcHeight; y++)
	{
		pLocalSrc = pSrcVc_NMS;
		for (vx_int32 x = 0; x < (vx_int32)srcWidth; x++)
		{
			vx_float32 Vc = *pLocalSrc;
			if (Vc)
			{
				
				for (vx_int32 i = max(y - radius, 0); i <= min(y + radius, (vx_int32) srcHeight - 1); i++)
				{
					for (vx_int32 j = max(x - radius, 0); j <= min(x + radius, (vx_int32) srcWidth - 1); j++)
					{
						if ((vx_float32)((y-i)*(y-i)) + (vx_float32)((x-j)*(x-j)) <= radius*radius)
						{
							vx_float32 * neighbor = (vx_float32 *)(((char *)pLocalSrc) + (i - y) * (vx_int32)srcVcStrideInBytes + (j - x) * sizeof(vx_float32));
							if (*neighbor < Vc)
								*neighbor = 0;
						}
					}
				}
			}
			pLocalSrc++;
		}

		pSrcVc_NMS = (vx_float32 *)((char *)pSrcVc_NMS + srcVcStrideInBytes);
	}	

	// Populate the sorted list
	vx_keypoint_t cand;
	vx_uint32 numCorners = 0;
	
	for (vx_uint32 y = 0; y < srcHeight; y++)
	{
		pLocalSrc = pSrcVc;
		for (vx_uint32 x = 0; x < srcWidth; x++)
		{
			if (*pLocalSrc)
			{
				cand.x = x;
				cand.y = y;
				cand.strength = *pLocalSrc;
				cand.scale = 0;
				cand.orientation = 0;
				cand.error = 0;
				cand.tracking_status = 1;
				if (numCorners < capacityOfDstCorner)
					AddToTheSortedKeypointList(capacityOfDstCorner, dstCorner, &numCorners, cand);
				else
					numCorners++;
			}
			pLocalSrc++;
		}
		pSrcVc = (vx_float32 *)((char *)pSrcVc + srcVcStrideInBytes);
	}

	*pDstCornerCount = numCorners;

	return AGO_SUCCESS;
}

int HafCpu_NonMaxSupp_XY_ANY_3x3
	(
		vx_uint32               capacityOfList,
		ago_keypoint_xys_t    * dstList,
		vx_uint32             * pDstListCount,
		vx_uint32               srcWidth,
		vx_uint32               srcHeight,
		vx_float32            * pSrcImg,
		vx_uint32               srcStrideInBytes
	)
{
	vx_uint32 count = 0;
	const vx_uint8 * pImg = (const vx_uint8 *)pSrcImg;
	for (vx_uint32 y = 1; y < srcHeight - 1; y++, pImg += srcStrideInBytes) {
		if (count >= capacityOfList)
			break;
		const vx_float32 * p9 = (const vx_float32 *)&pImg[0];
		const vx_float32 * p0 = (const vx_float32 *)&pImg[srcStrideInBytes];
		const vx_float32 * p1 = (const vx_float32 *)&pImg[srcStrideInBytes << 1];
		for (vx_uint32 x = 1; x < srcWidth - 1; x++) {
			if (p0[1] >= p9[0] && p0[1] >= p9[1] && p0[1] >= p9[2] &&
				p0[1] >= p0[0]                   && p0[1] >  p0[2] &&
				p0[1] >  p1[0] && p0[1] >  p1[1] && p0[1] >  p1[2])
			{
				dstList->x = x;
				dstList->y = y;
				dstList->s = p0[1];
				dstList++;
				count++;
				if (count >= capacityOfList)
					break;
			}
			p9++;
			p0++;
			p1++;
		}
	}
	*pDstListCount = count;
	return AGO_SUCCESS;
}

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
	)
{
	// sort the keypoint XYS list
	std::sort((vx_int64 *)&srcList[0], (vx_int64 *)&srcList[srcListCount], std::greater<vx_int64>());
	// extract useful keypoints from XYS list into corners array
	vx_uint32 count = 0;
	if (gridInfo) {
		// get grid info and initialize grid buffer if (-1,-1) coordinate values indicating no presence of values
		vx_uint32 gridWidth = gridInfo->width;
		vx_uint32 gridHeight = gridInfo->height;
		vx_uint32 cellSize = gridInfo->cellSize;
		HafCpu_MemSet_U32(gridInfo->gridBufSize >> 2, (vx_uint32 *)gridBuf, (vx_uint32)-1);
		// filter the keypoints with min_distance
		vx_int32 min_dist2 = (vx_int32)ceilf(min_distance * min_distance);
		vx_keypoint_t * corner = dstCorner;
		for (vx_uint32 i = 0; i < srcListCount; i++) {
			vx_uint32 x = srcList[i].x, y = srcList[i].y;
			bool found = true;
			vx_int32 cx = (vx_int32)x / cellSize, cy = (vx_int32)y / cellSize;
			ago_coord2d_short_t * cgrid = gridBuf + cy * gridWidth + cx;
			if (cgrid->x < 0) {
				vx_int32 cxmin = max(cx - 2, 0), cxmax = min(cx + 2, (vx_int32)gridWidth - 1), cw = cxmax - cxmin + 1;
				vx_int32 cymin = max(cy - 2, 0), cymax = min(cy + 2, (vx_int32)gridHeight - 1), ch = cymax - cymin + 1;
				ago_coord2d_short_t * grid = gridBuf + cxmin + cymin * gridWidth;
				for (vx_int32 icy = 0; icy < ch; icy++, grid += gridWidth) {
					for (vx_int32 icx = 0; icx < cw; icx++) {
						int ix = grid[icx].x;
						if (ix >= 0) {
							int iy = grid[icx].y;
							ix -= x; iy -= y;
							int dist2 = ix*ix + iy*iy;
							if (dist2 < min_dist2) {
								goto search_done;
							}
						}
					}
				}
				found = false;
			}
		search_done:
			if (!found) {
				if (count < capacityOfDstCorner) {
					corner->x = x;
					corner->y = y;
					corner->strength = srcList[i].s;
					corner->tracking_status = 1;
					corner->error = 0;
					corner->scale = 0.0f;
					corner->orientation = 0.0f;
					corner++;
				}
				count++;
				cgrid->x = x;
				cgrid->y = y;
			}
		}
	}
	else {
		// copy all points into output array
		count = (srcListCount < capacityOfDstCorner) ? srcListCount : capacityOfDstCorner;
		for (vx_uint32 i = 0; i < count; i++, dstCorner++, srcList++) {
			dstCorner->x = srcList->x;
			dstCorner->y = srcList->y;
			dstCorner->strength = srcList->s;
			dstCorner->tracking_status = 1;
			dstCorner->error = 0;
			dstCorner->scale = 0.0f;
			dstCorner->orientation = 0.0f;
		}
	}
	*pDstCornerCount = count;
	return AGO_SUCCESS;
}