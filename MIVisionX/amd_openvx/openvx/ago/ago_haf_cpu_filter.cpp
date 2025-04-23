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

extern vx_uint32 dataConvertU1ToU8_4bytes[16];

/* The function assumes at least one pixel padding on the top, left, right and bottom
Separable filter
	1    1 1 1
	1
	1
*/
int HafCpu_Box_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pScratch
	)
{
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	unsigned char *pLocalDst = (unsigned char *)pDstImage;
	

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int tmpWidth = (dstWidth + 15) & ~15;
	vx_uint16 * pPrevRow = (vx_uint16*) pScratch;
	vx_uint16 * pCurrRow = ((vx_uint16*) pScratch) + tmpWidth;
	vx_uint16 * pNextRow = ((vx_uint16*)pScratch) + (tmpWidth + tmpWidth);

	__m128i row0, shiftedR, shiftedL, temp0, temp1, resultH, resultL;
	__m128i zeromask = _mm_setzero_si128();
	__m128i divFactor = _mm_set1_epi16((short)7282);							// ceil((2^16)/9) = 7282

	vx_uint16 * pLocalPrevRow = pPrevRow;
	vx_uint16 * pLocalCurrRow = pCurrRow;
	vx_uint16 * pLocalNextRow = pNextRow;
	vx_uint16 * pTemp;

	// Process first two rows - Horizontal filtering
	for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes] + (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
		*pLocalCurrRow++ = (vx_uint16)pLocalSrc[-1] + (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
	}

	for (int x = 0; x < (alignedWidth >> 4); x++)
	{
		// row above
		row0 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes));
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));

		resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,-1)
		resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,-1)

		shiftedL = _mm_unpackhi_epi8(row0, zeromask);					// H: 1 * (-1, 0)
		row0 = _mm_cvtepu8_epi16(row0);									// L: 1 * (-1, 0)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, row0);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,-1)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, shiftedR);

		_mm_storeu_si128((__m128i *) pLocalPrevRow, resultL);
		_mm_storeu_si128((__m128i *) (pLocalPrevRow + 8), resultH);

		// current row
		row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

		resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,-1)
		resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,-1)

		shiftedL = _mm_unpackhi_epi8(row0, zeromask);					// H: 1 * (0,-1)
		row0 = _mm_cvtepu8_epi16(row0);									// L: 2 * (0,-1)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, row0);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,-1)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, shiftedR);

		_mm_storeu_si128((__m128i *) pLocalCurrRow, resultL);
		_mm_storeu_si128((__m128i *) (pLocalCurrRow + 8), resultH);

		pLocalSrc += 16;
		pLocalPrevRow += 16;
		pLocalCurrRow += 16;
	}

	for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes] + (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
		*pLocalCurrRow++ = (vx_uint16)pLocalSrc[-1] + (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
	}

	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;
	pLocalNextRow = pNextRow;

	// Process rows 3 till the end
	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcImageStrideInBytes);				// Pointing to the row below
		pLocalDst = (unsigned char *) pDstImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			vx_uint16 temp = (vx_uint16)pLocalSrc[-1] + (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
			*pLocalNextRow++ = temp;										// Save the next row temp pixels
			*pLocalDst++ = (char)((float)(temp + *pLocalPrevRow++ + *pLocalCurrRow++) / 9.0f);
		}

		int width = (int)(alignedWidth >> 4);
		while (width)
		{
			// Horizontal Filtering
			// current row
			row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

			resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,-1)
			resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,-1)

			shiftedL = _mm_unpackhi_epi8(row0, zeromask);					// H: 2 * (0,-1)
			row0 = _mm_cvtepu8_epi16(row0);									// L: 2 * (0,-1)
			resultH = _mm_add_epi16(resultH, shiftedL);
			resultL = _mm_add_epi16(resultL, row0);

			temp0 = _mm_loadu_si128((__m128i*) pLocalPrevRow);				// Prev Row
			temp1 = _mm_loadu_si128((__m128i*) (pLocalPrevRow + 8));

			shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,-1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,-1)
			resultH = _mm_add_epi16(resultH, shiftedL);
			resultL = _mm_add_epi16(resultL, shiftedR);

			shiftedL = _mm_loadu_si128((__m128i*) pLocalCurrRow);			// Current Row
			shiftedR = _mm_loadu_si128((__m128i*) (pLocalCurrRow + 8));

			temp1 = _mm_add_epi16(temp1, resultH);							// Prev row + next row
			temp0 = _mm_add_epi16(temp0, resultL);

			_mm_storeu_si128((__m128i*) pLocalNextRow, resultL);			// Save the horizontal filtered pixels from the next row
			_mm_storeu_si128((__m128i*) (pLocalNextRow + 8), resultH);

			temp1 = _mm_add_epi16(temp1, shiftedR);							// Prev row + curr row + next row
			temp0 = _mm_add_epi16(temp0, shiftedL);
			temp1 = _mm_mulhi_epi16(temp1, divFactor);
			temp0 = _mm_mulhi_epi16(temp0, divFactor);

			temp0 = _mm_packus_epi16(temp0, temp1);
			_mm_store_si128((__m128i*) pLocalDst, temp0);

			pLocalSrc += 16;
			pLocalDst += 16;
			pLocalPrevRow += 16;
			pLocalCurrRow += 16;
			pLocalNextRow += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			vx_uint16 temp = (vx_uint16)pLocalSrc[-1] + (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
			*pLocalNextRow++ = temp;										// Save the next row temp pixels
			*pLocalDst++ = (char)((float)(temp + *pLocalPrevRow++ + *pLocalCurrRow++) / 9.0f);
		}

		pTemp = pPrevRow;
		pPrevRow = pCurrRow;
		pCurrRow = pNextRow;
		pNextRow = pTemp;

		pLocalPrevRow = pPrevRow;
		pLocalCurrRow = pCurrRow;
		pLocalNextRow = pNextRow;

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_Dilate_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	unsigned char *pLocalSrc, *pLocalDst;
	__m128i row0, row1, row2, shiftedR, shiftedL;

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;
	
	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (unsigned char *)pDstImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			unsigned char temp1, temp2;
			temp1 = max(max(pLocalSrc[-(int)srcImageStrideInBytes - 1], pLocalSrc[-(int)srcImageStrideInBytes]), pLocalSrc[-(int)srcImageStrideInBytes + 1]);
			temp2 = max(max(pLocalSrc[-1], pLocalSrc[0]), pLocalSrc[1]);
			temp1 = max(temp1, temp2);
			temp2 = max(max(pLocalSrc[(int)srcImageStrideInBytes - 1], pLocalSrc[(int)srcImageStrideInBytes]), pLocalSrc[(int)srcImageStrideInBytes + 1]);
			*pLocalDst++ = max(temp1, temp2);
		}

		for (int width = 0; width < (int)(alignedWidth >> 4); width++, pLocalSrc += 16, pLocalDst += 16)
		{
			// For the row above
			row0 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes));
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));
			row0 = _mm_max_epu8(row0, shiftedL);
			row0 = _mm_max_epu8(row0, shiftedR);

			// For the current row
			row1 = _mm_loadu_si128((__m128i *) pLocalSrc);
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));
			row1 = _mm_max_epu8(row1, shiftedL);
			row1 = _mm_max_epu8(row1, shiftedR);

			// For the row below
			row2 = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes));
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes + 1));
			row2 = _mm_max_epu8(row2, shiftedL);
			row2 = _mm_max_epu8(row2, shiftedR);

			row0 = _mm_max_epu8(row0, row1);
			row0 = _mm_max_epu8(row0, row2);
			_mm_store_si128((__m128i *) pLocalDst, row0);
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			unsigned char temp1, temp2;
			temp1 = max(max(pLocalSrc[-(int)srcImageStrideInBytes - 1], pLocalSrc[-(int)srcImageStrideInBytes]), pLocalSrc[-(int)srcImageStrideInBytes + 1]);
			temp2 = max(max(pLocalSrc[-1], pLocalSrc[0]), pLocalSrc[1]);
			temp1 = max(temp1, temp2);
			temp2 = max(max(pLocalSrc[(int)srcImageStrideInBytes - 1], pLocalSrc[(int)srcImageStrideInBytes]), pLocalSrc[(int)srcImageStrideInBytes + 1]);
			*pLocalDst++ = max(temp1, temp2);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_Erode_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	unsigned char *pLocalSrc, *pLocalDst;
	__m128i row0, row1, row2, shiftedR, shiftedL;

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (unsigned char *)pDstImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			unsigned char temp1, temp2;
			temp1 = min(min(pLocalSrc[-(int)srcImageStrideInBytes - 1], pLocalSrc[-(int)srcImageStrideInBytes]), pLocalSrc[-(int)srcImageStrideInBytes + 1]);
			temp2 = min(min(pLocalSrc[-1], pLocalSrc[0]), pLocalSrc[1]);
			temp1 = min(temp1, temp2);
			temp2 = min(min(pLocalSrc[(int)srcImageStrideInBytes - 1], pLocalSrc[(int)srcImageStrideInBytes]), pLocalSrc[(int)srcImageStrideInBytes + 1]);
			*pLocalDst++ = min(temp1, temp2);
		}

		for (int width = 0; width < (int)(alignedWidth >> 4); width++, pLocalSrc += 16, pLocalDst += 16)
		{
			// For the row above
			row0 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes));
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));
			row0 = _mm_min_epu8(row0, shiftedL);
			row0 = _mm_min_epu8(row0, shiftedR);

			// For the current row
			row1 = _mm_loadu_si128((__m128i *) pLocalSrc);
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));
			row1 = _mm_min_epu8(row1, shiftedL);
			row1 = _mm_min_epu8(row1, shiftedR);

			// For the row below
			row2 = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes));
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes + 1));
			row2 = _mm_min_epu8(row2, shiftedL);
			row2 = _mm_min_epu8(row2, shiftedR);

			row0 = _mm_min_epu8(row0, row1);
			row0 = _mm_min_epu8(row0, row2);
			_mm_store_si128((__m128i *) pLocalDst, row0);
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			unsigned char temp1, temp2;
			temp1 = min(min(pLocalSrc[-(int)srcImageStrideInBytes - 1], pLocalSrc[-(int)srcImageStrideInBytes]), pLocalSrc[-(int)srcImageStrideInBytes + 1]);
			temp2 = min(min(pLocalSrc[-1], pLocalSrc[0]), pLocalSrc[1]);
			temp1 = min(temp1, temp2);
			temp2 = min(min(pLocalSrc[(int)srcImageStrideInBytes - 1], pLocalSrc[(int)srcImageStrideInBytes]), pLocalSrc[(int)srcImageStrideInBytes + 1]);
			*pLocalDst++ = min(temp1, temp2);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

#if USE_BMI2
/* The function assumes that the source image pointer is 16 byte aligned, and the source stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth.
The function assumes at least one pixel padding on the top, left, right and bottom */
int HafCpu_Dilate_U1_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * src = (__m128i*) pSrcImage;

	__m128i row0, row1, row2, shiftedR, shiftedL, temp;
	__m128i maskL = _mm_set_epi8((char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0xFF);
	__m128i maskR = _mm_set_epi8((char)0xFF, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0);
	
	uint64_t maskConv = 0x0101010101010101;
	uint64_t result[2];
	char lpixel, rpixel;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			row0 = _mm_load_si128(&src[(width >> 4) - (srcImageStrideInBytes >> 4)]);			// row above
			row1 = _mm_load_si128(&src[width >> 4]);
			row2 = _mm_load_si128(&src[(width >> 4) + (srcImageStrideInBytes >> 4)]);			// row below

			// For the row above
			lpixel = (char)*(pSrcImage - srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage - srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row0, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row0, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row0 = _mm_or_si128(row0, shiftedL);
			row0 = _mm_or_si128(row0, shiftedR);

			// For the current row
			lpixel = (char)*(pSrcImage - 1);
			rpixel = (char)*(pSrcImage + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row1, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row1, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row1 = _mm_or_si128(row1, shiftedL);
			row1 = _mm_or_si128(row1, shiftedR);

			// For the row below
			lpixel = (char)*(pSrcImage + srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage + srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row2, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row2, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row2 = _mm_or_si128(row2, shiftedL);
			row2 = _mm_or_si128(row2, shiftedR);

			row0 = _mm_or_si128(row0, row1);
			row0 = _mm_or_si128(row0, row2);

			// Convert U8 to U1
#ifdef _WIN64
			result[0] = _pext_u64(row0.m128i_u64[0], maskConv);
			result[1] = _pext_u64(row0.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			
			*((unsigned char*)pDstImage + (width >> 4))= (unsigned char)(result[0]);
			*((unsigned char*)pDstImage + (width >> 4) + 1) = (unsigned char)(result[1]);
		}
		src += (srcImageStrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the source image pointer is 16 byte aligned, and the source stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth.
The function assumes at least one pixel padding on the top, left, right and bottom */
int HafCpu_Erode_U1_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * src = (__m128i*) pSrcImage;

	__m128i row0, row1, row2, shiftedR, shiftedL, temp;
	__m128i maskL = _mm_set_epi8((char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0xFF);
	__m128i maskR = _mm_set_epi8((char)0xFF, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0);

	uint64_t maskConv = 0x0101010101010101;
	uint64_t result[2];
	char lpixel, rpixel;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			row0 = _mm_load_si128(&src[(width >> 4) - (srcImageStrideInBytes >> 4)]);			// row above
			row1 = _mm_load_si128(&src[width >> 4]);											// current row
			row2 = _mm_load_si128(&src[(width >> 4) + (srcImageStrideInBytes >> 4)]);			// row below

			// For the row above
			lpixel = (char)*(pSrcImage - srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage - srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row0, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row0, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row0 = _mm_and_si128(row0, shiftedL);
			row0 = _mm_and_si128(row0, shiftedR);

			// For the current row
			lpixel = (char)*(pSrcImage - 1);
			rpixel = (char)*(pSrcImage + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row1, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row1, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row1 = _mm_and_si128(row1, shiftedL);
			row1 = _mm_and_si128(row1, shiftedR);

			// For the row below
			lpixel = (char)*(pSrcImage + srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage + srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row2, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row2, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row2 = _mm_and_si128(row2, shiftedL);
			row2 = _mm_and_si128(row2, shiftedR);

			row0 = _mm_and_si128(row0, row1);
			row0 = _mm_and_si128(row0, row2);

			// Convert U8 to U1
#ifdef _WIN64
			result[0] = _pext_u64(row0.m128i_u64[0], maskConv);
			result[1] = _pext_u64(row0.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif

			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((result[1] & 0xFF) << 8) | (result[0] & 0xFF));
		}
		src += (srcImageStrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the destination image pointer is 16 byte aligned, and the destination stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth.
The function assumes at least one pixel padding on the top, left, right and bottom */
int HafCpu_Dilate_U8_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * dst = (__m128i*) pDstImage;

	__m128i row0, row1, row2, shiftedR, shiftedL, temp;
	__m128i maskL = _mm_set_epi8((char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0xFF);
	__m128i maskR = _mm_set_epi8((char)0xFF, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0);
	
	__declspec(align(16)) uint64_t pixels[2];
	uint64_t maskConv = 0x0101010101010101;
	char lpixel, rpixel;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the row above
			pixels[0] = (uint64_t)(*(pSrcImage - srcImageStrideInBytes));
			pixels[1] = (uint64_t)(*(pSrcImage - srcImageStrideInBytes + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row0 = _mm_load_si128((__m128i*) pixels);

			// Read the current row
			pixels[0] = (uint64_t)(*pSrcImage);
			pixels[1] = (uint64_t)(*(pSrcImage + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row1 = _mm_load_si128((__m128i*) pixels);

			// Read the row below
			pixels[0] = (uint64_t)(*(pSrcImage + srcImageStrideInBytes));
			pixels[1] = (uint64_t)(*(pSrcImage + srcImageStrideInBytes + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row2 = _mm_load_si128((__m128i*) pixels);

			// For the row above
			lpixel = (char)*(pSrcImage - srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage - srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row0, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row0, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row0 = _mm_or_si128(row0, shiftedL);
			row0 = _mm_or_si128(row0, shiftedR);

			// For the current row
			lpixel = (char)*(pSrcImage - 1);
			rpixel = (char)*(pSrcImage + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row1, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row1, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row1 = _mm_or_si128(row1, shiftedL);
			row1 = _mm_or_si128(row1, shiftedR);

			// For the row below
			lpixel = (char)*(pSrcImage + srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage + srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row2, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row2, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row2 = _mm_or_si128(row2, shiftedL);
			row2 = _mm_or_si128(row2, shiftedR);

			row0 = _mm_or_si128(row0, row1);
			row0 = _mm_or_si128(row0, row2);
			
			// Convert the bytes from 0x01 -> 0xFF and 0x0 -> 0x0
			temp = _mm_setzero_si128();
			row0 = _mm_cmpgt_epi8(row0, temp);

			_mm_store_si128(&dst[width >> 4], row0);
		}
		pSrcImage += srcImageStrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function assumes that the destination image pointer is 16 byte aligned, and the destination stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth.
The function assumes at least one pixel padding on the top, left, right and bottom */
int HafCpu_Erode_U8_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * dst = (__m128i*) pDstImage;

	__m128i row0, row1, row2, shiftedR, shiftedL, temp;
	__m128i maskL = _mm_set_epi8((char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0xFF);
	__m128i maskR = _mm_set_epi8((char)0xFF, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0);

	__declspec(align(16)) uint64_t pixels[2];
	uint64_t maskConv = 0x0101010101010101;
	char lpixel, rpixel;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the row above
			pixels[0] = (uint64_t)(*(pSrcImage - srcImageStrideInBytes));
			pixels[1] = (uint64_t)(*(pSrcImage - srcImageStrideInBytes + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row0 = _mm_load_si128((__m128i*) pixels);

			// Read the current row
			pixels[0] = (uint64_t)(*pSrcImage);
			pixels[1] = (uint64_t)(*(pSrcImage + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row1 = _mm_load_si128((__m128i*) pixels);

			// Read the row below
			pixels[0] = (uint64_t)(*(pSrcImage + srcImageStrideInBytes));
			pixels[1] = (uint64_t)(*(pSrcImage + srcImageStrideInBytes + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row2 = _mm_load_si128((__m128i*) pixels);

			// For the row above
			lpixel = (char)*(pSrcImage - srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage - srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row0, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row0, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row0 = _mm_and_si128(row0, shiftedL);
			row0 = _mm_and_si128(row0, shiftedR);

			// For the current row
			lpixel = (char)*(pSrcImage - 1);
			rpixel = (char)*(pSrcImage + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row1, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row1, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row1 = _mm_and_si128(row1, shiftedL);
			row1 = _mm_and_si128(row1, shiftedR);

			// For the row below
			lpixel = (char)*(pSrcImage + srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage + srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row2, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row2, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row2 = _mm_and_si128(row2, shiftedL);
			row2 = _mm_and_si128(row2, shiftedR);

			row0 = _mm_and_si128(row0, row1);
			row0 = _mm_and_si128(row0, row2);

			// Convert the bytes from 0x01 -> 0xFF and 0x0 -> 0x0
			temp = _mm_setzero_si128();
			row0 = _mm_cmpgt_epi8(row0, temp);

			_mm_store_si128(&dst[width >> 4], row0);
		}
		pSrcImage += srcImageStrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function processes the pixels in a width which is the next highest multiple of 16 after dstWidth.
The function assumes at least one pixel padding on the top, left, right and bottom */
int HafCpu_Dilate_U1_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i row0, row1, row2, shiftedR, shiftedL, temp;
	__m128i maskL = _mm_set_epi8((char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0xFF);
	__m128i maskR = _mm_set_epi8((char)0xFF, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0);

	__declspec(align(16)) uint64_t pixels[2];
	uint64_t maskConv = 0x0101010101010101;
	char lpixel, rpixel;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the row above
			pixels[0] = (uint64_t)(*(pSrcImage - srcImageStrideInBytes));
			pixels[1] = (uint64_t)(*(pSrcImage - srcImageStrideInBytes + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row0 = _mm_load_si128((__m128i*) pixels);

			// Read the current row
			pixels[0] = (uint64_t)(*pSrcImage);
			pixels[1] = (uint64_t)(*(pSrcImage + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row1 = _mm_load_si128((__m128i*) pixels);

			// Read the row below
			pixels[0] = (uint64_t)(*(pSrcImage + srcImageStrideInBytes));
			pixels[1] = (uint64_t)(*(pSrcImage + srcImageStrideInBytes + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row2 = _mm_load_si128((__m128i*) pixels);

			// For the row above
			lpixel = (char)*(pSrcImage - srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage - srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row0, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row0, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row0 = _mm_or_si128(row0, shiftedL);
			row0 = _mm_or_si128(row0, shiftedR);

			// For the current row
			lpixel = (char)*(pSrcImage - 1);
			rpixel = (char)*(pSrcImage + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row1, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row1, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row1 = _mm_or_si128(row1, shiftedL);
			row1 = _mm_or_si128(row1, shiftedR);

			// For the row below
			lpixel = (char)*(pSrcImage + srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage + srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row2, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row2, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row2 = _mm_or_si128(row2, shiftedL);
			row2 = _mm_or_si128(row2, shiftedR);

			row0 = _mm_or_si128(row0, row1);
			row0 = _mm_or_si128(row0, row2);

			// Convert U8 to U1
#ifdef _WIN64
			pixels[0] = _pext_u64(row0.m128i_u64[0], maskConv);
			pixels[1] = _pext_u64(row0.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels[1] & 0xFF) << 8) | (pixels[0] & 0xFF));
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function processes the pixels in a width which is the next highest multiple of 16 after dstWidth.
The function assumes at least one pixel padding on the top, left, right and bottom */
int HafCpu_Erode_U1_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i row0, row1, row2, shiftedR, shiftedL, temp;
	__m128i maskL = _mm_set_epi8((char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0xFF);
	__m128i maskR = _mm_set_epi8((char)0xFF, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0, (char)0);

	__declspec(align(16)) uint64_t pixels[2];
	uint64_t maskConv = 0x0101010101010101;
	char lpixel, rpixel;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the row above
			pixels[0] = (uint64_t)(*(pSrcImage - srcImageStrideInBytes));
			pixels[1] = (uint64_t)(*(pSrcImage - srcImageStrideInBytes + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row0 = _mm_load_si128((__m128i*) pixels);

			// Read the current row
			pixels[0] = (uint64_t)(*pSrcImage);
			pixels[1] = (uint64_t)(*(pSrcImage + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row1 = _mm_load_si128((__m128i*) pixels);

			// Read the row below
			pixels[0] = (uint64_t)(*(pSrcImage + srcImageStrideInBytes));
			pixels[1] = (uint64_t)(*(pSrcImage + srcImageStrideInBytes + 8));
#ifdef _WIN64
			pixels[0] = _pdep_u64(pixels[0], maskConv);
			pixels[1] = _pdep_u64(pixels[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			row2 = _mm_load_si128((__m128i*) pixels);

			// For the row above
			lpixel = (char)*(pSrcImage - srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage - srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row0, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row0, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row0 = _mm_and_si128(row0, shiftedL);
			row0 = _mm_and_si128(row0, shiftedR);

			// For the current row
			lpixel = (char)*(pSrcImage - 1);
			rpixel = (char)*(pSrcImage + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row1, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row1, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row1 = _mm_and_si128(row1, shiftedL);
			row1 = _mm_and_si128(row1, shiftedR);

			// For the row below
			lpixel = (char)*(pSrcImage + srcImageStrideInBytes - 1);
			rpixel = (char)*(pSrcImage + srcImageStrideInBytes + 17);
			temp = _mm_set1_epi8(lpixel);
			shiftedL = _mm_slli_si128(row2, 1);
			shiftedL = _mm_blendv_epi8(shiftedL, temp, maskL);
			temp = _mm_set1_epi8(rpixel);
			shiftedR = _mm_srli_si128(row2, 1);
			shiftedR = _mm_blendv_epi8(shiftedR, temp, maskR);
			row2 = _mm_and_si128(row2, shiftedL);
			row2 = _mm_and_si128(row2, shiftedR);

			row0 = _mm_and_si128(row0, row1);
			row0 = _mm_and_si128(row0, row2);

			// Convert U8 to U1
#ifdef _WIN64
			pixels[0] = _pext_u64(row0.m128i_u64[0], maskConv);
			pixels[1] = _pext_u64(row0.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels[1] & 0xFF) << 8) | (pixels[0] & 0xFF));
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

#else
/* The function assumes that the source width is a multiple of 8 pixels and the source image pointer points to the second row of the image (first row of the valid region)*/
int HafCpu_Dilate_U1_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i *pLocalSrcCurrRow, *pLocalSrcPrevRow, *pLocalSrcNextRow;
	vx_int16 * pLocalDst;

	__m128i row0, row1, row2, shiftedR, shiftedL;
	
	int pixelmask;
	int alignedWidth = (int)dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	int strideDiv16 = (int)(srcImageStrideInBytes >> 4);
	int height = (int)dstHeight;

	while (height > 0)
	{
		pLocalSrcCurrRow = (__m128i*) pSrcImage;
		pLocalSrcPrevRow = pLocalSrcCurrRow - strideDiv16;
		pLocalSrcNextRow = pLocalSrcCurrRow + strideDiv16;
		pLocalDst = (vx_int16 *)pDstImage;

		int width = alignedWidth >> 4;						// 16 pixels (bits) are processed at a time in the inner loop
		while (width > 0)
		{
			row0 = _mm_loadu_si128(pLocalSrcPrevRow);
			row1 = _mm_loadu_si128(pLocalSrcCurrRow);
			row2 = _mm_loadu_si128(pLocalSrcNextRow);
			
			// For the row above
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrcPrevRow - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrcPrevRow + 1));
			row0 = _mm_or_si128(row0, shiftedL);
			row0 = _mm_or_si128(row0, shiftedR);

			// For the current row
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrcCurrRow - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrcCurrRow + 1));
			row1 = _mm_or_si128(row1, shiftedL);
			row1 = _mm_or_si128(row1, shiftedR);

			// For the row below
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrcNextRow - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrcNextRow + 1));
			row2 = _mm_or_si128(row2, shiftedL);
			row2 = _mm_or_si128(row2, shiftedR);

			row0 = _mm_or_si128(row0, row1);
			row0 = _mm_or_si128(row0, row2);
			pixelmask = _mm_movemask_epi8(row0);						// Convert U8 to U1
			*pLocalDst++ = (vx_int16)(pixelmask & 0xFFFF);
			pLocalSrcCurrRow++;
			pLocalSrcPrevRow++;
			pLocalSrcNextRow++;
			width--;
		}

		if (postfixWidth)					// XX XX valid XX
		{
			vx_int16 * pRow = ((vx_int16 *)pLocalSrcPrevRow) - 1;
			pixelmask = *((int *)pRow);
			pixelmask = (pixelmask << 1) | pixelmask | (pixelmask >> 1);

			pRow = ((vx_int16 *)pLocalSrcCurrRow) - 1;
			int temp = *((int *)pRow);
			pixelmask |= ((temp << 1) | temp | (temp >> 1));

			pRow = ((vx_int16 *)pLocalSrcNextRow) - 1;
			temp = *((int *)pRow);
			pixelmask |= ((temp << 1) | temp | (temp >> 1));

			*((vx_uint8*)pLocalDst) = (vx_uint8)((pixelmask >> 8) & 0xFF);
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_Erode_U1_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i *pLocalSrcCurrRow, *pLocalSrcPrevRow, *pLocalSrcNextRow;
	vx_int16 * pLocalDst;

	__m128i row0, row1, row2, shiftedR, shiftedL;

	int pixelmask;
	int alignedWidth = (int)dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	int strideDiv16 = (int)(srcImageStrideInBytes >> 4);
	int height = (int)dstHeight;

	while (height > 0)
	{
		pLocalSrcCurrRow = (__m128i*) pSrcImage;
		pLocalSrcPrevRow = pLocalSrcCurrRow - (srcImageStrideInBytes >> 4);
		pLocalSrcNextRow = pLocalSrcCurrRow + (srcImageStrideInBytes >> 4);
		pLocalDst = (vx_int16 *)pDstImage;

		int width = alignedWidth >> 4;					// 16 pixels (bits) are processed at a time in the inner loop
		while (width > 0)
		{
			row0 = _mm_loadu_si128(pLocalSrcPrevRow);
			row1 = _mm_loadu_si128(pLocalSrcCurrRow);
			row2 = _mm_loadu_si128(pLocalSrcNextRow);

			// For the row above
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrcPrevRow - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrcPrevRow + 1));
			row0 = _mm_and_si128(row0, shiftedL);
			row0 = _mm_and_si128(row0, shiftedR);

			// For the current row
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrcCurrRow - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrcCurrRow + 1));
			row1 = _mm_and_si128(row1, shiftedL);
			row1 = _mm_and_si128(row1, shiftedR);

			// For the row below
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrcNextRow - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrcNextRow + 1));
			row2 = _mm_and_si128(row2, shiftedL);
			row2 = _mm_and_si128(row2, shiftedR);

			row0 = _mm_and_si128(row0, row1);
			row0 = _mm_and_si128(row0, row2);
			pixelmask = _mm_movemask_epi8(row0);						// Convert U8 to U1
			*pLocalDst++ = (vx_int16)(pixelmask & 0xFFFF);
			pLocalSrcCurrRow++;
			pLocalSrcPrevRow++;
			pLocalSrcNextRow++;
			width--;
		}

		if (postfixWidth)					// XX XX valid XX
		{
			vx_int16 * pRow = ((vx_int16 *)pLocalSrcPrevRow) - 1;
			pixelmask = *((int *)pRow);
			pixelmask = (pixelmask << 1) & pixelmask & (pixelmask >> 1);

			pRow = ((vx_int16 *)pLocalSrcCurrRow) - 1;
			int temp = *((int *)pRow);
			pixelmask &= ((temp << 1) & temp & (temp >> 1));

			pRow = ((vx_int16 *)pLocalSrcNextRow) - 1;
			temp = *((int *)pRow);
			pixelmask &= ((temp << 1) & temp & (temp >> 1));

			*((vx_uint8*)pLocalDst) = (vx_uint8)((pixelmask >> 8) & 0xFF);
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_Dilate_U1_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	short *pLocalSrc, *pLocalDst;

	int pixels, row, shiftedL, shiftedR;

	int alignedWidth = (int)dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (short *) (pSrcImage - 1);
		pLocalDst = (short *) pDstImage;
		int width = alignedWidth >> 4;					// 16 pixels processed at a time in the inner loop

		int strideDiv2 = (int)(srcImageStrideInBytes >> 1);
		while (width)
		{
			// Each read, reads 32 bits, first 8 bits don't care, next 16 bits useful and last 8 again don't care
			// Previous row
			row = *((int*)(pLocalSrc - strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels = row | shiftedL | shiftedR;

			// Current row
			row = *((int*)pLocalSrc);
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels |= (row | shiftedL | shiftedR);

			// Next row
			row = *((int*)(pLocalSrc + strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels |= (row | shiftedL | shiftedR);

			*pLocalDst++ = (short)((pixels >> 8) & 0xFFFF);
			pLocalSrc++;
			width--;
		}

		if (postfixWidth)
		{
			row = *((int*)(pLocalSrc - strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels = row | shiftedL | shiftedR;

			// Current row
			row = *((int*)pLocalSrc);
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels |= (row | shiftedL | shiftedR);

			// Next row
			row = *((int*)(pLocalSrc + strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels |= (row | shiftedL | shiftedR);

			*((vx_uint8*)pLocalDst) = (vx_uint8)((pixels >> 16) & 0xFF);
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_Erode_U1_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	short *pLocalSrc, *pLocalDst;

	int pixels, row, shiftedL, shiftedR;

	int alignedWidth = (int)dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (short *)(pSrcImage - 1);
		pLocalDst = (short *)pDstImage;
		int width = alignedWidth >> 4;
		int strideDiv2 = (int)(srcImageStrideInBytes >> 1);
		while (width)
		{
			// Each read, reads 32 bits, first 8 bits don't care, next 16 bits useful and last 8 again don't care

			// Previous row
			row = *((int*)(pLocalSrc - strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels = row & shiftedL & shiftedR;

			// Current row
			row = *((int*)pLocalSrc);
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels &= (row & shiftedL & shiftedR);

			// Next row
			row = *((int*)(pLocalSrc + strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels &= (row & shiftedL & shiftedR);

			*pLocalDst++ = (short)((pixels >> 8) & 0xFFFF);
			pLocalSrc++;
			width--;
		}

		if (postfixWidth)
		{
			// Previous row
			row = *((int*)(pLocalSrc - strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels = row & shiftedL & shiftedR;

			// Current row
			row = *((int*)pLocalSrc);
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels &= (row & shiftedL & shiftedR);

			// Next row
			row = *((int*)(pLocalSrc + strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels &= (row & shiftedL & shiftedR);

			*((vx_uint8*)pLocalDst) = (vx_uint8)((pixels >> 16) & 0xFF);
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_Dilate_U8_U1_3x3
	(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint8    * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes
	)
{
	vx_int16 * pLocalSrc;
	vx_int32 * pLocalDst;

	int pixels, row, shiftedL, shiftedR;

	int alignedWidth = (int)dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	int height = (int)dstHeight;
	int strideDiv2 = (int)(srcImageStrideInBytes >> 1);
	while (height)
	{
		pLocalSrc = (vx_int16 *)(pSrcImage - 1);
		pLocalDst = (vx_int32 *)pDstImage;
		int width = alignedWidth >> 4;

		while (width)
		{
			// Each read, reads 32 bits, first 8 bits don't care, next 16 bits useful and last 8 again don't care
			// Previous row
			row = *((int*)(pLocalSrc - strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels = row | shiftedL | shiftedR;

			// Current row
			row = *((int*)pLocalSrc);
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels |= (row | shiftedL | shiftedR);

			// Next row
			row = *((int*)(pLocalSrc + strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels |= (row | shiftedL | shiftedR);

			pixels >>= 8;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pLocalSrc++;
			width--;
		}

		if (postfixWidth)
		{
			row = *((int*)(pLocalSrc - strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels = row | shiftedL | shiftedR;

			// Current row
			row = *((int*)pLocalSrc);
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels |= (row | shiftedL | shiftedR);

			// Next row
			row = *((int*)(pLocalSrc + strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels |= (row | shiftedL | shiftedR);

			pixels >>= 16;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_Erode_U8_U1_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	short * pLocalSrc;
	int * pLocalDst;

	int pixels, row, shiftedL, shiftedR;

	int alignedWidth = (int)dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	int height = (int)dstHeight;
	int strideDiv2 = (int)(srcImageStrideInBytes >> 1);

	while (height)
	{
		pLocalSrc = (short *)(pSrcImage - 1);
		pLocalDst = (int *)pDstImage;
		int width = alignedWidth >> 4;

		while (width)
		{
			// Each read, reads 32 bits, first 8 bits don't care, next 16 bits useful and last 8 again don't care
			// Previous row
			row = *((int*)(pLocalSrc - strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels = row & shiftedL & shiftedR;

			// Current row
			row = *((int*)pLocalSrc);
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels &= (row & shiftedL & shiftedR);

			// Next row
			row = *((int*)(pLocalSrc + strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels &= (row & shiftedL & shiftedR);

			pixels >>= 8;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pLocalSrc++;
			width--;
		}

		if (postfixWidth)
		{
			// Previous row
			row = *((int*)(pLocalSrc - strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels = row & shiftedL & shiftedR;

			// Current row
			row = *((int*)pLocalSrc);
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels &= (row & shiftedL & shiftedR);

			// Next row
			row = *((int*)(pLocalSrc + strideDiv2));
			shiftedL = row << 1;
			shiftedR = row >> 1;
			pixels &= (row & shiftedL & shiftedR);

			pixels >>= 16;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
			pixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels & 0xF];
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

#endif

/* The function assumes at least one pixel padding on the top, left, right and bottom 
   Separable filter
	1    1 2 1
	2
	1
*/
int HafCpu_Gaussian_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	)
{
#if USE_AVX
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	unsigned char *pLocalDst = (unsigned char *)pDstImage;

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int tmpWidth = (dstWidth + 15) & ~15;
	vx_uint16 * pPrevRow = (vx_uint16*)pScratch;
	vx_uint16 * pCurrRow = ((vx_uint16*)pScratch) + tmpWidth;
	vx_uint16 * pNextRow = ((vx_uint16*)pScratch) + (tmpWidth + tmpWidth);

	__m128i row0, shiftedR, shiftedL;
	__m256i temp0, resultL;

	vx_uint16 * pLocalPrevRow = pPrevRow;
	vx_uint16 * pLocalCurrRow = pCurrRow;
	vx_uint16 * pLocalNextRow = pNextRow;
	vx_uint16 * pTemp;

	// Process first two rows - Horizontal filtering
	for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + 2 * (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes] + (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
		*pLocalCurrRow++ = (vx_uint16)pLocalSrc[-1] + 2 * (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
	}

	for (int x = 0; x < (alignedWidth >> 4); x++)
	{
		// row above
		row0 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes));
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));

		resultL = _mm256_cvtepu8_epi16(shiftedL);						// 1 * (-1,-1)
		
		temp0 = _mm256_cvtepu8_epi16(row0);
		temp0 = _mm256_slli_epi16(temp0, 1);							// 2 * (0,-1)
		resultL = _mm256_add_epi16(resultL, temp0);

		temp0 = _mm256_cvtepu8_epi16(shiftedR);							// 1 * (1,-1)
		resultL = _mm256_add_epi16(resultL, temp0);

		_mm256_storeu_si256((__m256i *) pLocalPrevRow, resultL);

		// current row
		row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

		resultL = _mm256_cvtepu8_epi16(shiftedL);						// 1 * (-1,-1)

		temp0 = _mm256_cvtepu8_epi16(row0);
		temp0 = _mm256_slli_epi16(temp0, 1);							// 2 * (0,-1)
		resultL = _mm256_add_epi16(resultL, temp0);

		temp0 = _mm256_cvtepu8_epi16(shiftedR);							// 1 * (1,-1)
		resultL = _mm256_add_epi16(resultL, temp0);

		_mm256_storeu_si256((__m256i *) pLocalCurrRow, resultL);

		pLocalSrc += 16;
		pLocalPrevRow += 16;
		pLocalCurrRow += 16;
	}

	for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + 2 * (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes] + (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
		*pLocalCurrRow++ = (vx_uint16)pLocalSrc[-1] + 2 * (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
	}

	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;
	pLocalNextRow = pNextRow;

	// Process rows 3 till the end
	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcImageStrideInBytes);				// Pointing to the row below
		pLocalDst = (unsigned char *)pDstImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			vx_uint16 temp = (vx_uint16)pLocalSrc[-1] + 2 * (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
			*pLocalNextRow++ = temp;													// Save the next row temp pixels
			*pLocalDst++ = (char)((temp + *pLocalPrevRow++ + 2*(*pLocalCurrRow++)) >> 4);
		}

		int width = (int)(alignedWidth >> 4);
		while (width)
		{
			// Horizontal Filtering
			// current row
			row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

			resultL = _mm256_cvtepu8_epi16(shiftedL);						// 1 * (-1,-1)

			temp0 = _mm256_cvtepu8_epi16(row0);
			temp0 = _mm256_slli_epi16(temp0, 1);							// 2 * (0,-1)
			resultL = _mm256_add_epi16(resultL, temp0);

			temp0 = _mm256_cvtepu8_epi16(shiftedR);							// 1 * (1,-1)
			resultL = _mm256_add_epi16(resultL, temp0);
			_mm256_storeu_si256((__m256i*) pLocalNextRow, resultL);			// Save the horizontal filtered pixels from the next row

			temp0 = _mm256_loadu_si256((__m256i*) pLocalPrevRow);			// Prev Row
			resultL = _mm256_add_epi16(resultL, temp0);						// Prev Row + Next Row

			temp0 = _mm256_loadu_si256((__m256i*) pLocalCurrRow);			// Current Row
			temp0 = _mm256_slli_epi16(temp0, 1);							// Current Row * 2
			
			resultL = _mm256_add_epi16(resultL, temp0);						// Prev row + 2*curr row + next row
			resultL = _mm256_srli_epi16(resultL, 4);						// Div by 16 (normalization)
			
			resultL = _mm256_packus_epi16(resultL, resultL);				// Convert to 8 bit
			row0 = _mm256_castsi256_si128(resultL);							// Lower 128 bits 
			_mm_store_si128((__m128i*) pLocalDst, row0);

			pLocalSrc += 16;
			pLocalDst += 16;
			pLocalPrevRow += 16;
			pLocalCurrRow += 16;
			pLocalNextRow += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			vx_uint16 temp = (vx_uint16)pLocalSrc[-1] + 2 * (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
			*pLocalNextRow++ = temp;										// Save the next row temp pixels
			*pLocalDst++ = (char)((temp + *pLocalPrevRow++ + 2*(*pLocalCurrRow++)) >> 4);
		}

		pTemp = pPrevRow;
		pPrevRow = pCurrRow;
		pCurrRow = pNextRow;
		pNextRow = pTemp;

		pLocalPrevRow = pPrevRow;
		pLocalCurrRow = pCurrRow;
		pLocalNextRow = pNextRow;

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
#else
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	unsigned char *pLocalDst = (unsigned char *)pDstImage;

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int tmpWidth = (dstWidth + 15) & ~15;
	vx_uint16 * pPrevRow = (vx_uint16*)pScratch;
	vx_uint16 * pCurrRow = ((vx_uint16*)pScratch) + tmpWidth;
	vx_uint16 * pNextRow = ((vx_uint16*)pScratch) + (tmpWidth + tmpWidth);

	__m128i row0, shiftedR, shiftedL, temp0, temp1, resultH, resultL;
	__m128i zeromask = _mm_setzero_si128();

	vx_uint16 * pLocalPrevRow = pPrevRow;
	vx_uint16 * pLocalCurrRow = pCurrRow;
	vx_uint16 * pLocalNextRow = pNextRow;
	vx_uint16 * pTemp;

	// Process first two rows - Horizontal filtering
	for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + 2 * (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes] + (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
		*pLocalCurrRow++ = (vx_uint16)pLocalSrc[-1] + 2 * (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
	}

	for (int x = 0; x < (alignedWidth >> 4); x++)
	{
		// row above
		row0 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes));
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));

		resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,-1)
		resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,-1)

		shiftedL = _mm_unpackhi_epi8(row0, zeromask);
		shiftedL = _mm_slli_epi16(shiftedL, 1);							// H: 2 * (0,-1)
		row0 = _mm_cvtepu8_epi16(row0);
		row0 = _mm_slli_epi16(row0, 1);									// L: 2 * (0,-1)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, row0);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,-1)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, shiftedR);

		_mm_storeu_si128((__m128i *) pLocalPrevRow, resultL);
		_mm_storeu_si128((__m128i *) (pLocalPrevRow + 8), resultH);

		// current row
		row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

		resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,0)
		resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,0)

		shiftedL = _mm_unpackhi_epi8(row0, zeromask);
		shiftedL = _mm_slli_epi16(shiftedL, 1);							// H: 2 * (0,0)
		row0 = _mm_cvtepu8_epi16(row0);
		row0 = _mm_slli_epi16(row0, 1);									// L: 2 * (0,0)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, row0);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,0)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,0)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, shiftedR);

		_mm_storeu_si128((__m128i *) pLocalCurrRow, resultL);
		_mm_storeu_si128((__m128i *) (pLocalCurrRow + 8), resultH);

		pLocalSrc += 16;
		pLocalPrevRow += 16;
		pLocalCurrRow += 16;
	}

	for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + 2 * (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes] + (vx_uint16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
		*pLocalCurrRow++ = (vx_uint16)pLocalSrc[-1] + 2 * (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
	}

	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;
	pLocalNextRow = pNextRow;

	// Process rows 3 till the end
	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcImageStrideInBytes);				// Pointing to the row below
		pLocalDst = (unsigned char *)pDstImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			vx_uint16 temp = (vx_uint16)pLocalSrc[-1] + 2 * (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
			*pLocalNextRow++ = temp;													// Save the next row temp pixels
			*pLocalDst++ = (char)((temp + *pLocalPrevRow++ + 2 * (*pLocalCurrRow++)) >> 4);
		}

		int width = (int)(alignedWidth >> 4);
		while (width)
		{
			// Horizontal Filtering
			// current row
			row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

			resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,-1)
			resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,-1)

			shiftedL = _mm_unpackhi_epi8(row0, zeromask);
			shiftedL = _mm_slli_epi16(shiftedL, 1);							// H: 2 * (0,0)
			row0 = _mm_cvtepu8_epi16(row0);
			row0 = _mm_slli_epi16(row0, 1);									// L: 2 * (0,0)
			resultH = _mm_add_epi16(resultH, shiftedL);
			resultL = _mm_add_epi16(resultL, row0);

			temp0 = _mm_loadu_si128((__m128i*) pLocalPrevRow);				// Prev Row
			temp1 = _mm_loadu_si128((__m128i*) (pLocalPrevRow + 8));

			shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,-1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,-1)
			resultH = _mm_add_epi16(resultH, shiftedL);
			resultL = _mm_add_epi16(resultL, shiftedR);

			shiftedL = _mm_loadu_si128((__m128i*) pLocalCurrRow);			// Current Row
			shiftedL = _mm_slli_epi16(shiftedL, 1);
			shiftedR = _mm_loadu_si128((__m128i*) (pLocalCurrRow + 8));
			shiftedR = _mm_slli_epi16(shiftedR, 1);

			temp1 = _mm_add_epi16(temp1, resultH);							// Prev row + next row
			temp0 = _mm_add_epi16(temp0, resultL);

			_mm_storeu_si128((__m128i*) pLocalNextRow, resultL);			// Save the horizontal filtered pixels from the next row
			_mm_storeu_si128((__m128i*) (pLocalNextRow + 8), resultH);

			temp1 = _mm_add_epi16(temp1, shiftedR);							// Prev row + curr row + next row
			temp0 = _mm_add_epi16(temp0, shiftedL);
			temp1 = _mm_srli_epi16(temp1, 4);
			temp0 = _mm_srli_epi16(temp0, 4);

			temp0 = _mm_packus_epi16(temp0, temp1);
			_mm_store_si128((__m128i*) pLocalDst, temp0);

			pLocalSrc += 16;
			pLocalDst += 16;
			pLocalPrevRow += 16;
			pLocalCurrRow += 16;
			pLocalNextRow += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			vx_uint16 temp = (vx_uint16)pLocalSrc[-1] + 2 * (vx_uint16)pLocalSrc[0] + (vx_uint16)pLocalSrc[1];
			*pLocalNextRow++ = temp;										// Save the next row temp pixels
			*pLocalDst++ = (char)((temp + *pLocalPrevRow++ + 2 * (*pLocalCurrRow++)) >> 4);
		}

		pTemp = pPrevRow;
		pPrevRow = pCurrRow;
		pCurrRow = pNextRow;
		pNextRow = pTemp;

		pLocalPrevRow = pPrevRow;
		pLocalCurrRow = pCurrRow;
		pLocalNextRow = pNextRow;

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
#endif
	
	return AGO_SUCCESS;
}

/* The function assumes at least one pixel padding on the top, left, right and bottom
   Separable filter
	 1    -1 0 1
	 2
	 1
*/
int HafCpu_Sobel_S16_U8_3x3_GX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstGxImage,
		vx_uint32     dstGxImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	)
{
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	short * pLocalDst;

	int prefixWidth = intptr_t(pDstGxImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int tmpWidth = (dstWidth + 15) & ~15;
	vx_int16 * pPrevRow = (vx_int16*)pScratch;
	vx_int16 * pCurrRow = ((vx_int16*)pScratch) + tmpWidth;
	vx_int16 * pNextRow = ((vx_int16*)pScratch) + (tmpWidth + tmpWidth);

	__m128i row0, shiftedR, shiftedL, temp0, temp1, resultH, resultL;
	__m128i zeromask = _mm_setzero_si128();

	vx_int16 * pLocalPrevRow = pPrevRow;
	vx_int16 * pLocalCurrRow = pCurrRow;
	vx_int16 * pLocalNextRow = pNextRow;
	vx_int16 * pTemp;

	// Process first two rows - Horizontal filtering
	for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1];
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
	}

	for (int x = 0; x < (int)(alignedWidth >> 4); x++)
	{
		// row above
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));

		resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: -1 * (-1,-1)
		resultL = _mm_cvtepu8_epi16(shiftedL);							// L: -1 * (-1,-1)

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,-1)
		resultH = _mm_sub_epi16(shiftedL, resultH);
		resultL = _mm_sub_epi16(shiftedR, resultL);

		_mm_store_si128((__m128i *) pLocalPrevRow, resultL);
		_mm_store_si128((__m128i *) (pLocalPrevRow + 8), resultH);

		// current row
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

		resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: -1 * (-1,0)
		resultL = _mm_cvtepu8_epi16(shiftedL);							// L: -1 * (-1,0)

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,0)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,0)
		resultH = _mm_sub_epi16(shiftedL, resultH);
		resultL = _mm_sub_epi16(shiftedR, resultL);

		_mm_store_si128((__m128i *) pLocalCurrRow, resultL);
		_mm_store_si128((__m128i *) (pLocalCurrRow + 8), resultH);

		pLocalSrc += 16;
		pLocalPrevRow += 16;
		pLocalCurrRow += 16;
	}

	for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1];
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
	}

	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;
	pLocalNextRow = pNextRow;

	// Process rows 3 till the end
	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcImageStrideInBytes);				// Pointing to the row below
		pLocalDst = (short *)pDstGxImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			vx_int16 temp = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
			*pLocalNextRow++ = temp;													// Save the next row temp pixels
			*pLocalDst++ = temp + *pLocalPrevRow++ + 2 * (*pLocalCurrRow++);
		}

		int width = (int)(alignedWidth >> 4);
		while (width)
		{
			// Horizontal Filtering
			// current row
			row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

			resultL = _mm_cvtepu8_epi16(shiftedL);							// L: -1 * (-1,-1)
			resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: -1 * (-1,-1)

			temp0 = _mm_load_si128((__m128i*) pLocalPrevRow);				// Prev Row
			temp1 = _mm_load_si128((__m128i*) (pLocalPrevRow + 8));

			shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,-1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,-1)
			resultH = _mm_sub_epi16(shiftedL, resultH);
			resultL = _mm_sub_epi16(shiftedR, resultL);

			shiftedL = _mm_load_si128((__m128i*) pLocalCurrRow);			// Current Row
			shiftedR = _mm_load_si128((__m128i*) (pLocalCurrRow + 8));

			temp1 = _mm_add_epi16(temp1, resultH);							// Prev row + next row
			temp0 = _mm_add_epi16(temp0, resultL);

			shiftedR = _mm_slli_epi16(shiftedR, 1);
			shiftedL = _mm_slli_epi16(shiftedL, 1);

			_mm_store_si128((__m128i*) pLocalNextRow, resultL);				// Save the horizontal filtered pixels from the next row
			_mm_store_si128((__m128i*) (pLocalNextRow + 8), resultH);

			temp1 = _mm_add_epi16(temp1, shiftedR);							// Prev row + 2*curr row + next row
			temp0 = _mm_add_epi16(temp0, shiftedL);

			_mm_store_si128((__m128i*) pLocalDst, temp0);
			_mm_store_si128((__m128i*) (pLocalDst + 8), temp1);

			pLocalSrc += 16;
			pLocalDst += 16;
			pLocalPrevRow += 16;
			pLocalCurrRow += 16;
			pLocalNextRow += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			vx_int16 temp = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
			*pLocalNextRow++ = temp;										// Save the next row temp pixels
			*pLocalDst++ = temp + *pLocalPrevRow++ + 2 * (*pLocalCurrRow++);
		}

		pTemp = pPrevRow;
		pPrevRow = pCurrRow;
		pCurrRow = pNextRow;
		pNextRow = pTemp;

		pLocalPrevRow = pPrevRow;
		pLocalCurrRow = pCurrRow;
		pLocalNextRow = pNextRow;

		pSrcImage += srcImageStrideInBytes;
		pDstGxImage += (dstGxImageStrideInBytes >> 1);
		height--;
	}
	return AGO_SUCCESS;
}

/* The function assumes at least one pixel padding on the top, left, right and bottom
	Separable filter:
	-1		1	2	1
	0
	1
*/
int HafCpu_Sobel_S16_U8_3x3_GY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstGyImage,
		vx_uint32     dstGyImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	)
{
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	short * pLocalDst;

	int prefixWidth = intptr_t(pDstGyImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int tmpWidth = (dstWidth + 15) & ~15;
	vx_int16 * pPrevRow = (vx_int16*)pScratch;
	vx_int16 * pCurrRow = ((vx_int16*)pScratch) + tmpWidth;
	vx_int16 * pNextRow = ((vx_int16*)pScratch) + (tmpWidth + tmpWidth);

	__m128i row0, shiftedR, shiftedL, temp0, temp1, resultH, resultL;
	__m128i zeromask = _mm_setzero_si128();

	vx_int16 * pLocalPrevRow = pPrevRow;
	vx_int16 * pLocalCurrRow = pCurrRow;
	vx_int16 * pLocalNextRow = pNextRow;
	vx_int16 * pTemp;

	// Process first two rows - Horizontal filtering
	for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + ((vx_int16)pLocalSrc[-(int)srcImageStrideInBytes] << 1) + (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];
	}

	for (int x = 0; x < (int)(alignedWidth >> 4); x++)
	{
		// row above
		row0 = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes));
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));

		resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,-1)
		resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,-1)

		shiftedL = _mm_unpackhi_epi8(row0, zeromask);
		shiftedL = _mm_slli_epi16(shiftedL, 1);							// H: 2 * (0,-1)
		row0 = _mm_cvtepu8_epi16(row0);
		row0 = _mm_slli_epi16(row0, 1);									// L: 2 * (0,-1)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, row0);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,-1)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, shiftedR);

		_mm_storeu_si128((__m128i *) pLocalPrevRow, resultL);
		_mm_storeu_si128((__m128i *) (pLocalPrevRow + 8), resultH);

		// current row
		row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

		resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,0)
		resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,0)

		shiftedL = _mm_unpackhi_epi8(row0, zeromask);
		shiftedL = _mm_slli_epi16(shiftedL, 1);							// H: 2 * (0,0)
		row0 = _mm_cvtepu8_epi16(row0);
		row0 = _mm_slli_epi16(row0, 1);									// L: 2 * (0,0)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, row0);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,0)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,0)
		resultH = _mm_add_epi16(resultH, shiftedL);
		resultL = _mm_add_epi16(resultL, shiftedR);

		_mm_storeu_si128((__m128i *) pLocalCurrRow, resultL);
		_mm_storeu_si128((__m128i *) (pLocalCurrRow + 8), resultH);

		pLocalSrc += 16;
		pLocalPrevRow += 16;
		pLocalCurrRow += 16;
	}

	for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + ((vx_int16)pLocalSrc[-(int)srcImageStrideInBytes] << 1) + (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];
	}

	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;
	pLocalNextRow = pNextRow;

	// Process rows 3 till the end
	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcImageStrideInBytes);				// Pointing to the row below
		pLocalDst = (short *)pDstGyImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			vx_int16 temp = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];
			*pLocalNextRow++ = temp;													// Save the next row temp pixels
			*pLocalDst++ = *pLocalPrevRow++ - temp;
		}

		int width = (int)(alignedWidth >> 4);
		while (width)
		{
			// Horizontal Filtering
			// current row
			row0 = _mm_loadu_si128((__m128i *) pLocalSrc);
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

			resultL = _mm_cvtepu8_epi16(shiftedL);							// L: 1 * (-1,1)
			resultH = _mm_unpackhi_epi8(shiftedL, zeromask);				// H: 1 * (-1,1)

			temp0 = _mm_load_si128((__m128i*) pLocalPrevRow);				// Prev Row
			temp1 = _mm_load_si128((__m128i*) (pLocalPrevRow + 8));

			shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// H: 1 * (1,1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);							// L: 1 * (1,1)
			resultH = _mm_add_epi16(shiftedL, resultH);
			resultL = _mm_add_epi16(shiftedR, resultL);

			shiftedL = _mm_unpackhi_epi8(row0, zeromask);
			shiftedR = _mm_cvtepu8_epi16(row0);
			shiftedL = _mm_slli_epi16(shiftedL, 1);							// H: 2 * (0,1)
			shiftedR = _mm_slli_epi16(shiftedR, 1);							// L: 2 * (0,1)
			resultH = _mm_add_epi16(shiftedL, resultH);						// Horizontal filtered next row
			resultL = _mm_add_epi16(shiftedR, resultL);

			_mm_store_si128((__m128i*) pLocalNextRow, resultL);				// Save the horizontal filtered pixels from the next row
			_mm_store_si128((__m128i*) (pLocalNextRow + 8), resultH);

			temp1 = _mm_sub_epi16(resultH, temp1);							// Next row - prev row
			temp0 = _mm_sub_epi16(resultL, temp0);

			_mm_store_si128((__m128i*) pLocalDst, temp0);
			_mm_store_si128((__m128i*) (pLocalDst + 8), temp1);

			pLocalSrc += 16;
			pLocalDst += 16;
			pLocalPrevRow += 16;
			pLocalCurrRow += 16;
			pLocalNextRow += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			vx_int16 temp = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];
			*pLocalNextRow++ = temp;													// Save the next row temp pixels
			*pLocalDst++ = *pLocalPrevRow++ - temp;
		}

		pTemp = pPrevRow;
		pPrevRow = pCurrRow;
		pCurrRow = pNextRow;
		pNextRow = pTemp;

		pLocalPrevRow = pPrevRow;
		pLocalCurrRow = pCurrRow;
		pLocalNextRow = pNextRow;

		pSrcImage += srcImageStrideInBytes;
		pDstGyImage += (dstGyImageStrideInBytes >> 1);
		height--;
	}

	return AGO_SUCCESS;
}

/* The function assumes at least one pixel padding on the top, left, right and bottom 
	Separable filter
		1	 -1 0 1					-1	  1 2 1
   Gx = 2						Gy = 0
		1							-1
*/
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
	)
{	
	unsigned char *pLocalSrc = (unsigned char *)pSrcImage;
	short *pLocalDstGx, *pLocalDstGy;

	int prefixWidth = intptr_t(pDstGxImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int tmpWidth = (dstWidth + 15) & ~15;
	vx_int16 * pPrevRow = (vx_int16*)pScratch;
	vx_int16 * pCurrRow = ((vx_int16*)pScratch) + (2 * tmpWidth);
	vx_int16 * pNextRow = ((vx_int16*)pScratch) + (4 * tmpWidth);

	__m128i row0, shiftedR, shiftedL, temp0, temp1, temp2, GxH, GxL, GyH, GyL;
	__m128i zeromask = _mm_setzero_si128();

	vx_int16 * pLocalPrevRow = pPrevRow;
	vx_int16 * pLocalCurrRow = pCurrRow;
	vx_int16 * pLocalNextRow = pNextRow;
	vx_int16 * pTemp;

	// Process first two rows - Horizontal filtering
	for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1];					// Gx
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + ((vx_int16)pLocalSrc[-(int)srcImageStrideInBytes] << 1) + (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1];	// Gy
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];									// Gx
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];	// Gy
	}

	for (int x = 0; x < (int)(alignedWidth >> 4); x++)
	{
		// row above
		row0 = _mm_load_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes));
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));

		GyH = _mm_unpackhi_epi8(row0, zeromask);
		GyH = _mm_slli_epi16(GyH, 1);									// GyH: 2 * (0,-1)
		GyL = _mm_cvtepu8_epi16(row0);
		GyL = _mm_slli_epi16(GyL, 1);									// GyL: 2 * (0,-1)

		GxL = _mm_cvtepu8_epi16(shiftedL);								// GxL: -1 * (-1,-1)	GyL: 1 * (-1,-1)
		GxH = _mm_unpackhi_epi8(shiftedL, zeromask);					// GxH: -1 * (-1,-1)	GyH: 1 * (-1,-1)
		GyH = _mm_add_epi16(GyH, GxH);
		GyL = _mm_add_epi16(GyL, GxL);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// GxH: 1 * (1,-1)		GyH: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// GxL: 1 * (1,-1)		GyL: 1 * (1,-1)
		GxH = _mm_sub_epi16(shiftedL, GxH);
		GxL = _mm_sub_epi16(shiftedR, GxL);
		GyH = _mm_add_epi16(GyH, shiftedL);
		GyL = _mm_add_epi16(GyL, shiftedR);

		_mm_store_si128((__m128i *) pLocalPrevRow, GxL);
		_mm_store_si128((__m128i *) (pLocalPrevRow + 8), GxH);
		_mm_store_si128((__m128i *) (pLocalPrevRow + 16), GyL);
		_mm_store_si128((__m128i *) (pLocalPrevRow + 24), GyH);

		// current row
		row0 = _mm_load_si128((__m128i *)pLocalSrc);
		shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
		shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

		GyH = _mm_unpackhi_epi8(row0, zeromask);
		GyH = _mm_slli_epi16(GyH, 1);									// GyH: 2 * (-1, 0)
		GyL = _mm_cvtepu8_epi16(row0);
		GyL = _mm_slli_epi16(GyL, 1);									// GyL: 2 * (-1, 0)

		GxL = _mm_cvtepu8_epi16(shiftedL);								// GxL: -1 * (-1,-1)	GyL: 1 * (-1,-1)
		GxH = _mm_unpackhi_epi8(shiftedL, zeromask);					// GxH: -1 * (-1,-1)	GyH: 1 * (-1,-1)
		GyH = _mm_add_epi16(GyH, GxH);
		GyL = _mm_add_epi16(GyL, GxL);

		shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// GxH: 1 * (1,-1)		GyH: 1 * (1,-1)
		shiftedR = _mm_cvtepu8_epi16(shiftedR);							// GxL: 1 * (1,-1)		GyL: 1 * (1,-1)
		GxH = _mm_sub_epi16(shiftedL, GxH);
		GxL = _mm_sub_epi16(shiftedR, GxL);
		GyH = _mm_add_epi16(GyH, shiftedL);
		GyL = _mm_add_epi16(GyL, shiftedR);

		_mm_store_si128((__m128i *) pLocalCurrRow, GxL);
		_mm_store_si128((__m128i *) (pLocalCurrRow + 8), GxH);
		_mm_store_si128((__m128i *) (pLocalCurrRow + 16), GyL);
		_mm_store_si128((__m128i *) (pLocalCurrRow + 24), GyH);

		pLocalSrc += 16;
		pLocalPrevRow += 32;
		pLocalCurrRow += 32;
	}

	for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
	{
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1];					// Gx
		*pLocalPrevRow++ = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + ((vx_int16)pLocalSrc[-(int)srcImageStrideInBytes] << 1) + (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1];	// Gy
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];									// Gx
		*pLocalCurrRow++ = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];	// Gy
	}

	pLocalPrevRow = pPrevRow;
	pLocalCurrRow = pCurrRow;
	pLocalNextRow = pNextRow;

	// Process rows 3 till the end
	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)(pSrcImage + srcImageStrideInBytes);				// Pointing to the row below
		pLocalDstGx = (short *) pDstGxImage;
		pLocalDstGy = (short *) pDstGyImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			vx_int16 tempGx = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
			*pLocalNextRow++ = tempGx;
			vx_int16 tempGy = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];
			*pLocalNextRow++ = tempGy;

			*pLocalDstGx++ = *pLocalPrevRow++ + ((*pLocalCurrRow++) << 1) + tempGx;
			*pLocalDstGy++ = tempGy - *pLocalPrevRow++;
			pLocalCurrRow++;
		}

		int width = (int)(dstWidth >> 4);
		while (width)
		{
			// Horizontal Filtering
			// next row
			row0 = _mm_load_si128((__m128i *) pLocalSrc);
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

			GyH = _mm_unpackhi_epi8(row0, zeromask);
			GyH = _mm_slli_epi16(GyH, 1);									// GyH: 2 * (-1, 0)
			GyL = _mm_cvtepu8_epi16(row0);
			GyL = _mm_slli_epi16(GyL, 1);									// GyL: 2 * (-1, 0)

			GxL = _mm_cvtepu8_epi16(shiftedL);								// GxL: -1 * (-1,-1)	GyL: 1 * (-1,-1)
			GxH = _mm_unpackhi_epi8(shiftedL, zeromask);					// GxH: -1 * (-1,-1)	GyH: 1 * (-1,-1)
			GyH = _mm_add_epi16(GyH, GxH);
			GyL = _mm_add_epi16(GyL, GxL);

			temp0 = _mm_load_si128((__m128i *) pLocalPrevRow);				// Prev Row - Gx
			temp1 = _mm_load_si128((__m128i *) (pLocalPrevRow + 8));
			row0 = _mm_load_si128((__m128i *) (pLocalPrevRow + 16));		// Prev Row - Gy
			temp2 = _mm_load_si128((__m128i *) (pLocalPrevRow + 24));

			shiftedL = _mm_unpackhi_epi8(shiftedR, zeromask);				// GxH: 1 * (1,-1)		GyH: 1 * (1,-1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);							// GxL: 1 * (1,-1)		GyL: 1 * (1,-1)
			GxH = _mm_sub_epi16(shiftedL, GxH);
			GxL = _mm_sub_epi16(shiftedR, GxL);
			GyH = _mm_add_epi16(GyH, shiftedL);
			GyL = _mm_add_epi16(GyL, shiftedR);

			shiftedL = _mm_load_si128((__m128i *) pLocalCurrRow);			// Current Row
			shiftedR = _mm_load_si128((__m128i *) (pLocalCurrRow + 8));

			temp1 = _mm_add_epi16(temp1, GxH);								// Prev row + next row
			temp0 = _mm_add_epi16(temp0, GxL);

			shiftedR = _mm_slli_epi16(shiftedR, 1);
			shiftedL = _mm_slli_epi16(shiftedL, 1);

			_mm_store_si128((__m128i *) pLocalNextRow, GxL);				// Save the horizontal filtered pixels from the next row - Gx
			_mm_store_si128((__m128i *) (pLocalNextRow + 8), GxH);
			_mm_store_si128((__m128i *) (pLocalNextRow + 16), GyL);			// Save the horizontal filtered pixels from the next row - Gy
			_mm_store_si128((__m128i *) (pLocalNextRow + 24), GyH);


			temp1 = _mm_add_epi16(temp1, shiftedR);							// next row - Prev row 
			temp0 = _mm_add_epi16(temp0, shiftedL);
			row0 = _mm_sub_epi16(GyL, row0);
			temp2 = _mm_sub_epi16(GyH, temp2);

			_mm_store_si128((__m128i *) pLocalDstGx, temp0);
			_mm_store_si128((__m128i *) (pLocalDstGx + 8), temp1);
			_mm_store_si128((__m128i *) pLocalDstGy, row0);
			_mm_store_si128((__m128i *) (pLocalDstGy + 8), temp2);

			pLocalSrc += 16;
			pLocalDstGx += 16;
			pLocalDstGy += 16;
			pLocalPrevRow += 32;
			pLocalCurrRow += 32;
			pLocalNextRow += 32;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			vx_int16 tempGx = (vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1];
			*pLocalNextRow++ = tempGx;
			vx_int16 tempGy = (vx_int16)pLocalSrc[-1] + ((vx_int16)pLocalSrc[0] << 1) + (vx_int16)pLocalSrc[1];
			*pLocalNextRow++ = tempGy;

			*pLocalDstGx++ = *pLocalPrevRow++ + ((*pLocalCurrRow++) << 1) + tempGx;
			*pLocalDstGy++ = tempGy - *pLocalPrevRow++;
			pLocalCurrRow++;
		}

		pTemp = pPrevRow;
		pPrevRow = pCurrRow;
		pCurrRow = pNextRow;
		pNextRow = pTemp;

		pLocalPrevRow = pPrevRow;
		pLocalCurrRow = pCurrRow;
		pLocalNextRow = pNextRow;

		pSrcImage += srcImageStrideInBytes;
		pDstGxImage += (dstGxImageStrideInBytes >> 1);
		pDstGyImage += (dstGyImageStrideInBytes >> 1);
		height--;
	}
	return AGO_SUCCESS;
}

/* The function assumes at least one pixel padding on the top, left, right and bottom */
int HafCpu_SobelMagnitude_S16_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstMagImage,
		vx_uint32     dstMagImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	unsigned char * pLocalSrc;
	short * pLocalDst;

	int prefixWidth = intptr_t(pDstMagImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i row0, row1, row2, shiftedR, shiftedL, temp, GxH, GxL, GyH, GyL;
	__m128i zeromask = _mm_setzero_si128();

	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (short *) pDstMagImage;
		
		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			vx_int16 tempGx = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1) + 
				(vx_int16)pLocalSrc[(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[(int)srcImageStrideInBytes - 1];
			vx_int16 tempGy = (vx_int16)pLocalSrc[(int)srcImageStrideInBytes - 1] + ((vx_int16)pLocalSrc[(int)srcImageStrideInBytes] << 1) + (vx_int16)pLocalSrc[(int)srcImageStrideInBytes + 1] -
				(vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] - ((vx_int16)pLocalSrc[-(int)srcImageStrideInBytes] << 1) - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
			float mag = (float)(tempGx*tempGx) + (float)(tempGy*tempGy);
			mag = sqrtf(mag);
			*pLocalDst++ = (vx_int16)mag;
		}

		int width = (int)(alignedWidth >> 4);						// 16 pixels processed at a time
		while (width)
		{
			row0 = _mm_load_si128((__m128i *) (pLocalSrc - srcImageStrideInBytes));		// row above
			row1 = _mm_load_si128((__m128i *) pLocalSrc);								// current row
			row2 = _mm_load_si128((__m128i *) (pLocalSrc + srcImageStrideInBytes));		// row below

			// For the row below
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes + 1));

			GxH = _mm_unpackhi_epi8(shiftedL, zeromask);				// Gx, H: -1 * (-1,1)
			GxL = _mm_cvtepu8_epi16(shiftedL);							// Gx, L: -1 * (-1,1)
			GyH = _mm_add_epi16(GxH, zeromask);							// Gy, H: 1 * (-1,1)
			GyL = _mm_add_epi16(GxL, zeromask);							// Gy, L: 1 * (-1,1)

			temp = _mm_unpackhi_epi8(row2, zeromask);
			temp = _mm_slli_epi16(temp, 1);								// Gy, H: 2 * (0,1)
			row2 = _mm_cvtepu8_epi16(row2);
			row2 = _mm_slli_epi16(row2, 1);								// Gy, L: 2 * (0,1)
			GyH = _mm_add_epi16(GyH, temp);
			GyL = _mm_add_epi16(GyL, row2);

			temp = _mm_unpackhi_epi8(shiftedR, zeromask);				// Gy, H: 1 * (1,1),	Gx, H: 1 * (1,1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);						// Gy, L: 1 * (1,1),	Gx, L: 1 * (1,1)
			GyH = _mm_add_epi16(GyH, temp);
			GyL = _mm_add_epi16(GyL, shiftedR);
			GxH = _mm_sub_epi16(temp, GxH);
			GxL = _mm_sub_epi16(shiftedR, GxL);

			// For the current row
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

			temp = _mm_unpackhi_epi8(shiftedR, zeromask);
			temp = _mm_slli_epi16(temp, 1);								// Gx, H: 2 * (1,0)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);
			shiftedR = _mm_slli_epi16(shiftedR, 1);						// Gx, L: 2 * (1,0)
			GxH = _mm_add_epi16(GxH, temp);
			GxL = _mm_add_epi16(GxL, shiftedR);

			temp = _mm_unpackhi_epi8(shiftedL, zeromask);
			temp = _mm_slli_epi16(temp, 1);								// Gx, H: -2 * (-1,0)
			shiftedL = _mm_cvtepu8_epi16(shiftedL);
			shiftedL = _mm_slli_epi16(shiftedL, 1);						// Gx, L: -2 * (-1,0)
			GxH = _mm_sub_epi16(GxH, temp);
			GxL = _mm_sub_epi16(GxL, shiftedL);

			// For the row above
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));

			temp = _mm_unpackhi_epi8(shiftedR, zeromask);				// Gy, H: -1 * (1,-1),	Gx, H: 1 * (1,-1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);						// Gy, L: -1 * (1,-1),	Gx, L: 1 * (1,-1)
			GxH = _mm_add_epi16(GxH, temp);
			GxL = _mm_add_epi16(GxL, shiftedR);
			GyH = _mm_sub_epi16(GyH, temp);
			GyL = _mm_sub_epi16(GyL, shiftedR);

			temp = _mm_unpackhi_epi8(row0, zeromask);
			row0 = _mm_cvtepu8_epi16(row0);
			temp = _mm_slli_epi16(temp, 1);								// Gy, H: -2 * (0,-1)
			row0 = _mm_slli_epi16(row0, 1);								// Gy, L: -2 * (0,-1)
			GyH = _mm_sub_epi16(GyH, temp);
			GyL = _mm_sub_epi16(GyL, row0);

			temp = _mm_unpackhi_epi8(shiftedL, zeromask);				// Gy, H: -1 * (-1,-1),	Gx, H: -1 * (-1,-1)
			shiftedL = _mm_cvtepu8_epi16(shiftedL);						// Gy, L: -1 * (-1,-1),	Gx, L: -1 * (-1,-1)
			GxH = _mm_sub_epi16(GxH, temp);
			GxL = _mm_sub_epi16(GxL, shiftedL);
			GyH = _mm_sub_epi16(GyH, temp);
			GyL = _mm_sub_epi16(GyL, shiftedL);

			// Magnitude
			row0 = _mm_srli_si128(GxH, 8);
			row1 = _mm_srli_si128(GxL, 8);
			row0 = _mm_cvtepi16_epi32(row0);							// GxH: Upper 4 words to dwords
			GxH = _mm_cvtepi16_epi32(GxH);								// GxH: Lower 4 words to dwords
			row1 = _mm_cvtepi16_epi32(row1);							// GxL: Upper 4 words to dwords
			GxL = _mm_cvtepi16_epi32(GxL);								// GxL: Lower 4 words to dwords

			row2 = _mm_srli_si128(GyH, 8);
			temp = _mm_srli_si128(GyL, 8);
			row2 = _mm_cvtepi16_epi32(row2);							// GyH: Upper 4 words to dwords
			GyH = _mm_cvtepi16_epi32(GyH);								// GyH: Lower 4 words to dwords
			temp = _mm_cvtepi16_epi32(temp);							// GyL: Upper 4 words to dwords
			GyL = _mm_cvtepi16_epi32(GyL);								// GyL: Lower 4 words to dwords
			
			row0 = _mm_mullo_epi32(row0, row0);							// Square
			GxH = _mm_mullo_epi32(GxH, GxH);
			row1 = _mm_mullo_epi32(row1, row1);
			GxL = _mm_mullo_epi32(GxL, GxL);
			row2 = _mm_mullo_epi32(row2, row2);
			GyH = _mm_mullo_epi32(GyH, GyH);
			temp = _mm_mullo_epi32(temp, temp);
			GyL = _mm_mullo_epi32(GyL, GyL);

			row0 = _mm_add_epi32(row0, row2);							// Add
			GxH = _mm_add_epi32(GxH, GyH);
			row1 = _mm_add_epi32(row1, temp);
			GxL = _mm_add_epi32(GxL, GyL);

			temp = _mm_srli_si128(row0, 8);
			__m128d d_pix1 = _mm_cvtepi32_pd(temp);						// Pixels 15, 14
			__m128d d_pix0 = _mm_cvtepi32_pd(row0);						// Pixels 13, 12
			d_pix1 = _mm_sqrt_pd(d_pix1);
			d_pix0 = _mm_sqrt_pd(d_pix0);
			row0 = _mm_cvtpd_epi32(d_pix1);
			temp = _mm_cvtpd_epi32(d_pix0);
			row0 = _mm_slli_si128(row0, 8);
			row0 = _mm_or_si128(row0, temp);							// Pixels 15, 14, 13, 12 (DWORDS)

			temp = _mm_srli_si128(GxH, 8);
			d_pix1 = _mm_cvtepi32_pd(temp);								// Pixels 11, 10
			d_pix0 = _mm_cvtepi32_pd(GxH);								// Pixels 9, 8
			d_pix1 = _mm_sqrt_pd(d_pix1);
			d_pix0 = _mm_sqrt_pd(d_pix0);
			GxH = _mm_cvtpd_epi32(d_pix1);
			temp = _mm_cvtpd_epi32(d_pix0);
			GxH = _mm_slli_si128(GxH, 8);
			GxH = _mm_or_si128(GxH, temp);								// Pixels 11, 10, 9, 8 (DWORDS)
			row0 = _mm_packus_epi32(GxH, row0);							// Pixels 15, 14, 13, 12, 11, 10, 9, 8 (WORDS)

			temp = _mm_srli_si128(row1, 8);
			d_pix1 = _mm_cvtepi32_pd(temp);								// Pixels 7, 6
			d_pix0 = _mm_cvtepi32_pd(row1);								// Pixels 5, 4
			d_pix1 = _mm_sqrt_pd(d_pix1);
			d_pix0 = _mm_sqrt_pd(d_pix0);
			row1 = _mm_cvtpd_epi32(d_pix1);
			temp = _mm_cvtpd_epi32(d_pix0);
			row1 = _mm_slli_si128(row1, 8);
			row1 = _mm_or_si128(row1, temp);							// Pixels 7, 6, 5, 4 (DWORDS)

			temp = _mm_srli_si128(GxL, 8);
			d_pix1 = _mm_cvtepi32_pd(temp);								// Pixels 3, 2
			d_pix0 = _mm_cvtepi32_pd(GxL);								// Pixels 1, 0
			d_pix1 = _mm_sqrt_pd(d_pix1);
			d_pix0 = _mm_sqrt_pd(d_pix0);
			GxL = _mm_cvtpd_epi32(d_pix1);
			temp = _mm_cvtpd_epi32(d_pix0);
			GxL = _mm_slli_si128(GxL, 8);
			GxL = _mm_or_si128(GxL, temp);								// Pixels 3, 2, 1, 0 (DWORDS)
			row1 = _mm_packus_epi32(GxL, row1);							// Pixels 7, 6, 5, 4, 3, 2, 1, 0 (WORDS)

			_mm_store_si128((__m128i *) pLocalDst, row1);
			_mm_store_si128((__m128i *) (pLocalDst + 8), row0);

			pLocalSrc += 16;
			pLocalDst += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			vx_int16 tempGx = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1) +
				(vx_int16)pLocalSrc[(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[(int)srcImageStrideInBytes - 1];
			vx_int16 tempGy = (vx_int16)pLocalSrc[(int)srcImageStrideInBytes - 1] + ((vx_int16)pLocalSrc[(int)srcImageStrideInBytes] << 1) + (vx_int16)pLocalSrc[(int)srcImageStrideInBytes + 1] -
				(vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] - ((vx_int16)pLocalSrc[-(int)srcImageStrideInBytes] << 1) - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
			float mag = (float)(tempGx*tempGx) + (float)(tempGy*tempGy);
			mag = sqrtf(mag);
			*pLocalDst++ = (vx_int16)mag;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstMagImage += (dstMagImageStrideInBytes >> 1);
		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i *pLocalDst_xmm;
	unsigned char *pLocalSrc;
	short * pLocalDst;
	short *pLocalConvMat;

	__m128i result0, result1, result2, result3, row, mul, temp0, temp1;
	__m128i zeromask = _mm_setzero_si128();

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	int srcStride = (int)srcImageStrideInBytes;
	int rowLimit = (int)(convolutionHeight >> 1);
	int numConvCoeffs = 3 * (int)convolutionHeight;

	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (short *)pDstImage;

		for (int w = 0; w < prefixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, SHRT_MAX);
			temp = max(temp, SHRT_MIN);
			*pLocalDst++ = (short)temp;
		}

		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);							// Each loop processess 16 pixels
		while (width)
		{
			pLocalConvMat = convMatrix + numConvCoeffs - 1;
			result0 = _mm_setzero_si128();
			result1 = _mm_setzero_si128();
			result2 = _mm_setzero_si128();
			result3 = _mm_setzero_si128();

			for (int y = -rowLimit; y <= rowLimit; y++)
			{
				int offset = y * srcStride;

				row = _mm_loadu_si128((__m128i *)(pLocalSrc + offset - 1));				// shifted left pixels
				mul = _mm_set1_epi32((int)(*pLocalConvMat--));

				// Upper 4 bytes - shiftedL pixels
				temp1 = _mm_unpackhi_epi8(row, zeromask);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result3 = _mm_add_epi32(result3, temp0);

				// Next 4 bytes - shiftedL pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result2 = _mm_add_epi32(result2, temp0);

				// Next 4 bytes - shiftedL pixels
				temp1 = _mm_cvtepu8_epi16(row);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result1 = _mm_add_epi32(result1, temp0);

				row = _mm_loadu_si128((__m128i *)(pLocalSrc + offset));				// pixels at the location

				// Lowest 4 bytes - shiftedL pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result0 = _mm_add_epi32(result0, temp0);

				mul = _mm_set1_epi32((int)(*pLocalConvMat--));
				// Upper 4 bytes - at loc pixels
				temp1 = _mm_unpackhi_epi8(row, zeromask);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result3 = _mm_add_epi32(result3, temp0);

				// Next 4 bytes - at loc pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result2 = _mm_add_epi32(result2, temp0);

				// Next 4 bytes - at loc pixels
				temp1 = _mm_cvtepu8_epi16(row);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result1 = _mm_add_epi32(result1, temp0);

				row = _mm_loadu_si128((__m128i *)(pLocalSrc + offset + 1));				// shifted right pixels

				// Lowest 4 bytes - at loc pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result0 = _mm_add_epi32(result0, temp0);

				mul = _mm_set1_epi32((int)(*pLocalConvMat--));
				// Upper 4 bytes - shiftedR pixels
				temp1 = _mm_unpackhi_epi8(row, zeromask);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result3 = _mm_add_epi32(result3, temp0);

				// Next 4 bytes - shiftedR pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result2 = _mm_add_epi32(result2, temp0);

				// Next 4 bytes - shiftedR pixels
				temp1 = _mm_cvtepu8_epi16(row);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result1 = _mm_add_epi32(result1, temp0);

				// Lowest 4 bytes - shiftedR pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result0 = _mm_add_epi32(result0, temp0);
			}

			result0 = _mm_srli_epi32(result0, shift);
			result1 = _mm_srli_epi32(result1, shift);
			result2 = _mm_srli_epi32(result2, shift);
			result3 = _mm_srli_epi32(result3, shift);

			row = _mm_packs_epi32(result2, result3);
			temp0 = _mm_packs_epi32(result0, result1);
			_mm_store_si128(pLocalDst_xmm++, temp0);
			_mm_store_si128(pLocalDst_xmm++, row);

			pLocalSrc += 16;
			width--;
		}

		pLocalDst = (short *)pLocalDst_xmm;
		for (int w = 0; w < postfixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, SHRT_MAX);
			temp = max(temp, SHRT_MIN);
			*pLocalDst++ = (short)temp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes >> 1);

		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i *pLocalDst_xmm;
	unsigned char *pLocalSrc, *pLocalDst;
	short *pLocalConvMat;

	__m128i result0, result1, result2, result3, row, mul, temp0, temp1;
	__m128i zeromask = _mm_setzero_si128();

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	int srcStride = (int)srcImageStrideInBytes;
	int rowLimit = (int)(convolutionHeight >> 1);
	int numConvCoeffs = 3 * (int)convolutionHeight;

	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (unsigned char *)pDstImage;

		for (int w = 0; w < prefixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, 255);
			temp = max(temp, 0);
			*pLocalDst++ = (unsigned char)temp;
		}

		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);							// Each loop processess 16 pixels
		while (width)
		{
			pLocalConvMat = convMatrix + numConvCoeffs - 1;
			result0 = _mm_setzero_si128();
			result1 = _mm_setzero_si128();
			result2 = _mm_setzero_si128();
			result3 = _mm_setzero_si128();

			for (int y = -rowLimit; y <= rowLimit; y++)
			{
				int offset = y * srcStride;
				row = _mm_loadu_si128((__m128i *)(pLocalSrc + offset - 1));				// shifted left pixels
				mul = _mm_set1_epi32((int)(*pLocalConvMat--));

				// Upper 4 bytes - shiftedL pixels
				temp1 = _mm_unpackhi_epi8(row, zeromask);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result3 = _mm_add_epi32(result3, temp0);

				// Next 4 bytes - shiftedL pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result2 = _mm_add_epi32(result2, temp0);

				// Next 4 bytes - shiftedL pixels
				temp1 = _mm_cvtepu8_epi16(row);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result1 = _mm_add_epi32(result1, temp0);

				row = _mm_loadu_si128((__m128i *)(pLocalSrc + offset));				// pixels at the location

				// Lowest 4 bytes - shiftedL pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result0 = _mm_add_epi32(result0, temp0);

				mul = _mm_set1_epi32((int)(*pLocalConvMat--));
				// Upper 4 bytes - at loc pixels
				temp1 = _mm_unpackhi_epi8(row, zeromask);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result3 = _mm_add_epi32(result3, temp0);

				// Next 4 bytes - at loc pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result2 = _mm_add_epi32(result2, temp0);

				// Next 4 bytes - at loc pixels
				temp1 = _mm_cvtepu8_epi16(row);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result1 = _mm_add_epi32(result1, temp0);

				row = _mm_loadu_si128((__m128i *)(pLocalSrc + offset + 1));				// shifted right pixels

				// Lowest 4 bytes - at loc pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result0 = _mm_add_epi32(result0, temp0);

				mul = _mm_set1_epi32((int)(*pLocalConvMat--));
				// Upper 4 bytes - shiftedR pixels
				temp1 = _mm_unpackhi_epi8(row, zeromask);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result3 = _mm_add_epi32(result3, temp0);

				// Next 4 bytes - shiftedR pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result2 = _mm_add_epi32(result2, temp0);

				// Next 4 bytes - shiftedR pixels
				temp1 = _mm_cvtepu8_epi16(row);
				temp0 = _mm_unpackhi_epi16(temp1, zeromask);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result1 = _mm_add_epi32(result1, temp0);

				// Lowest 4 bytes - shiftedR pixels
				temp0 = _mm_cvtepi16_epi32(temp1);
				temp0 = _mm_mullo_epi32(temp0, mul);
				result0 = _mm_add_epi32(result0, temp0);
			}

			result0 = _mm_srli_epi32(result0, shift);
			result1 = _mm_srli_epi32(result1, shift);
			result2 = _mm_srli_epi32(result2, shift);
			result3 = _mm_srli_epi32(result3, shift);

			row = _mm_packs_epi32(result2, result3);
			temp0 = _mm_packs_epi32(result0, result1);
			row = _mm_packus_epi16(temp0, row);
			_mm_store_si128(pLocalDst_xmm++, row);

			pLocalSrc += 16;
			width--;
		}

		pLocalDst = (unsigned char *)pLocalDst_xmm;
		for (int w = 0; w < postfixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, 255);
			temp = max(temp, 0);
			*pLocalDst++ = (unsigned char)temp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;

		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i *pLocalDst_xmm;
	unsigned char *pLocalSrc;
	short * pLocalDst;
	short *pLocalConvMat;

	__m128i result0, result1, result2, result3, row, mul, temp0, temp1;
	__m128i zeromask = _mm_setzero_si128();

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	int srcStride = (int)srcImageStrideInBytes;
	int rowLimit = (int)(convolutionHeight >> 1);
	int numConvCoeffs = 5 * (int)convolutionHeight;

	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (short *)pDstImage;

		for (int w = 0; w < prefixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -2; j <= 2; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, SHRT_MAX);
			temp = max(temp, SHRT_MIN);
			*pLocalDst++ = (short)temp;
		}

		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);							// Each loop processess 16 pixels
		while (width)
		{
			pLocalConvMat = convMatrix + numConvCoeffs - 1;
			result0 = _mm_setzero_si128();
			result1 = _mm_setzero_si128();
			result2 = _mm_setzero_si128();
			result3 = _mm_setzero_si128();

			for (int y = -rowLimit; y <= rowLimit; y++)
			{
				for (int x = -2; x <= 2; x++)
				{
					row = _mm_loadu_si128((__m128i *)(pLocalSrc + (y * srcStride) + x));
					mul = _mm_set1_epi32((int)(*pLocalConvMat--));

					// Upper 4 bytes
					temp1 = _mm_unpackhi_epi8(row, zeromask);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result3 = _mm_add_epi32(result3, temp0);

					// Next 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result2 = _mm_add_epi32(result2, temp0);

					// Next 4 bytes
					temp1 = _mm_cvtepu8_epi16(row);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result1 = _mm_add_epi32(result1, temp0);

					// Lowest 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result0 = _mm_add_epi32(result0, temp0);
				}
			}

			result0 = _mm_srli_epi32(result0, shift);
			result1 = _mm_srli_epi32(result1, shift);
			result2 = _mm_srli_epi32(result2, shift);
			result3 = _mm_srli_epi32(result3, shift);

			row = _mm_packs_epi32(result2, result3);
			temp0 = _mm_packs_epi32(result0, result1);
			_mm_store_si128(pLocalDst_xmm++, temp0);
			_mm_store_si128(pLocalDst_xmm++, row);

			pLocalSrc += 16;
			width--;
		}

		pLocalDst = (short *)pLocalDst_xmm;
		for (int w = 0; w < postfixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -2; j <= 2; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, SHRT_MAX);
			temp = max(temp, SHRT_MIN);
			*pLocalDst++ = (short)temp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes >> 1);

		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i *pLocalDst_xmm;
	unsigned char *pLocalSrc, *pLocalDst;
	short *pLocalConvMat;

	__m128i result0, result1, result2, result3, row, mul, temp0, temp1;
	__m128i zeromask = _mm_setzero_si128();

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	int srcStride = (int)srcImageStrideInBytes;
	int rowLimit = (int)(convolutionHeight >> 1);
	int numConvCoeffs = 5 * (int)convolutionHeight;

	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (unsigned char *)pDstImage;

		for (int w = 0; w < prefixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -2; j <= 2; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, 255);
			temp = max(temp, 0);
			*pLocalDst++ = (unsigned char)temp;
		}

		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);							// Each loop processess 16 pixels
		while (width)
		{
			pLocalConvMat = convMatrix + numConvCoeffs - 1;
			result0 = _mm_setzero_si128();
			result1 = _mm_setzero_si128();
			result2 = _mm_setzero_si128();
			result3 = _mm_setzero_si128();

			for (int y = -rowLimit; y <= rowLimit; y++)
			{
				for (int x = -2; x <= 2; x++)
				{
					row = _mm_loadu_si128((__m128i *)(pLocalSrc + (y * srcStride) + x));
					mul = _mm_set1_epi32((int)(*pLocalConvMat--));

					// Upper 4 bytes
					temp1 = _mm_unpackhi_epi8(row, zeromask);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result3 = _mm_add_epi32(result3, temp0);

					// Next 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result2 = _mm_add_epi32(result2, temp0);

					// Next 4 bytes
					temp1 = _mm_cvtepu8_epi16(row);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result1 = _mm_add_epi32(result1, temp0);

					// Lowest 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result0 = _mm_add_epi32(result0, temp0);
				}
			}

			result0 = _mm_srli_epi32(result0, shift);
			result1 = _mm_srli_epi32(result1, shift);
			result2 = _mm_srli_epi32(result2, shift);
			result3 = _mm_srli_epi32(result3, shift);

			row = _mm_packs_epi32(result2, result3);
			temp0 = _mm_packs_epi32(result0, result1);
			row = _mm_packus_epi16(temp0, row);
			_mm_store_si128(pLocalDst_xmm++, row);

			pLocalSrc += 16;
			width--;
		}

		pLocalDst = (unsigned char *)pLocalDst_xmm;
		for (int w = 0; w < postfixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -2; j <= 2; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, 255);
			temp = max(temp, 0);
			*pLocalDst++ = (unsigned char)temp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;

		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_Convolve_S16_U8_7xN
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * convMatrix,
		vx_size	      convolutionHeight,
		vx_int32      shift
	)
{
	__m128i *pLocalDst_xmm;
	unsigned char *pLocalSrc;
	short * pLocalDst;
	short *pLocalConvMat;

	__m128i result0, result1, result2, result3, row, mul, temp0, temp1;
	__m128i zeromask = _mm_setzero_si128();

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	int srcStride = (int)srcImageStrideInBytes;
	int rowLimit = (int)(convolutionHeight >> 1);
	int numConvCoeffs = 7 * (int)convolutionHeight;

	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (short *)pDstImage;

		for (int w = 0; w < prefixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -3; j <= 3; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride - j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, SHRT_MAX);
			temp = max(temp, SHRT_MIN);
			*pLocalDst++ = (short)temp;
		}

		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);							// Each loop processess 16 pixels
		while (width)
		{
			pLocalConvMat = convMatrix + numConvCoeffs - 1;
			result0 = _mm_setzero_si128();
			result1 = _mm_setzero_si128();
			result2 = _mm_setzero_si128();
			result3 = _mm_setzero_si128();

			for (int y = -rowLimit; y <= rowLimit; y++)
			{
				for (int x = -3; x <= 3; x++)
				{
					row = _mm_loadu_si128((__m128i *)(pLocalSrc + (y * srcStride) + x));
					mul = _mm_set1_epi32((int)(*pLocalConvMat--));

					// Upper 4 bytes
					temp1 = _mm_unpackhi_epi8(row, zeromask);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result3 = _mm_add_epi32(result3, temp0);

					// Next 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result2 = _mm_add_epi32(result2, temp0);

					// Next 4 bytes
					temp1 = _mm_cvtepu8_epi16(row);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result1 = _mm_add_epi32(result1, temp0);

					// Lowest 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result0 = _mm_add_epi32(result0, temp0);
				}
			}

			result0 = _mm_srli_epi32(result0, shift);
			result1 = _mm_srli_epi32(result1, shift);
			result2 = _mm_srli_epi32(result2, shift);
			result3 = _mm_srli_epi32(result3, shift);

			row = _mm_packs_epi32(result2, result3);
			temp0 = _mm_packs_epi32(result0, result1);
			_mm_store_si128(pLocalDst_xmm++, temp0);
			_mm_store_si128(pLocalDst_xmm++, row);

			pLocalSrc += 16;
			width--;
		}

		pLocalDst = (short *)pLocalDst_xmm;
		for (int w = 0; w < postfixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -3; j <= 3; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, SHRT_MAX);
			temp = max(temp, SHRT_MIN);
			*pLocalDst++ = (short)temp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes >> 1);

		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i *pLocalDst_xmm;
	unsigned char *pLocalSrc, *pLocalDst;
	short *pLocalConvMat;

	__m128i result0, result1, result2, result3, row, mul, temp0, temp1;
	__m128i zeromask = _mm_setzero_si128();

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	int srcStride = (int)srcImageStrideInBytes;
	int rowLimit = (int)(convolutionHeight >> 1);
	int numConvCoeffs = 7 * (int)convolutionHeight;

	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (unsigned char *)pDstImage;

		for (int w = 0; w < prefixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -3; j <= 3; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, 255);
			temp = max(temp, 0);
			*pLocalDst++ = (unsigned char)temp;
		}

		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);							// Each loop processess 16 pixels
		while (width)
		{
			pLocalConvMat = convMatrix + numConvCoeffs - 1;
			result0 = _mm_setzero_si128();
			result1 = _mm_setzero_si128();
			result2 = _mm_setzero_si128();
			result3 = _mm_setzero_si128();

			for (int y = -rowLimit; y <= rowLimit; y++)
			{
				for (int x = -3; x <= 3; x++)
				{
					row = _mm_loadu_si128((__m128i *)(pLocalSrc + (y * srcStride) + x));
					mul = _mm_set1_epi32((int)(*pLocalConvMat--));

					// Upper 4 bytes
					temp1 = _mm_unpackhi_epi8(row, zeromask);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result3 = _mm_add_epi32(result3, temp0);

					// Next 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result2 = _mm_add_epi32(result2, temp0);

					// Next 4 bytes
					temp1 = _mm_cvtepu8_epi16(row);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result1 = _mm_add_epi32(result1, temp0);

					// Lowest 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result0 = _mm_add_epi32(result0, temp0);
				}
			}

			result0 = _mm_srli_epi32(result0, shift);
			result1 = _mm_srli_epi32(result1, shift);
			result2 = _mm_srli_epi32(result2, shift);
			result3 = _mm_srli_epi32(result3, shift);

			row = _mm_packs_epi32(result2, result3);
			temp0 = _mm_packs_epi32(result0, result1);
			row = _mm_packus_epi16(temp0, row);
			_mm_store_si128(pLocalDst_xmm++, row);

			pLocalSrc += 16;
			width--;
		}

		pLocalDst = (unsigned char *)pLocalDst_xmm;
		for (int w = 0; w < postfixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -3; j <= 3; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, 255);
			temp = max(temp, 0);
			*pLocalDst++ = (unsigned char)temp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;

		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i *pLocalDst_xmm;
	unsigned char *pLocalSrc;
	short * pLocalDst;
	short *pLocalConvMat;

	__m128i result0, result1, result2, result3, row, mul, temp0, temp1;
	__m128i zeromask = _mm_setzero_si128();

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	int srcStride = (int)srcImageStrideInBytes;
	int rowLimit = (int)(convolutionHeight >> 1);
	int numConvCoeffs = 9 * (int)convolutionHeight;

	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (short *)pDstImage;

		for (int w = 0; w < prefixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -4; j <= 4; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, SHRT_MAX);
			temp = max(temp, SHRT_MIN);
			*pLocalDst++ = (short)temp;
		}

		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);							// Each loop processess 16 pixels
		while (width)
		{
			pLocalConvMat = convMatrix + numConvCoeffs - 1;
			result0 = _mm_setzero_si128();
			result1 = _mm_setzero_si128();
			result2 = _mm_setzero_si128();
			result3 = _mm_setzero_si128();

			for (int y = -rowLimit; y <= rowLimit; y++)
			{
				for (int x = -4; x <= 4; x++)
				{
					row = _mm_loadu_si128((__m128i *)(pLocalSrc + (y * srcStride) + x));
					mul = _mm_set1_epi32((int)(*pLocalConvMat--));

					// Upper 4 bytes
					temp1 = _mm_unpackhi_epi8(row, zeromask);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result3 = _mm_add_epi32(result3, temp0);

					// Next 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result2 = _mm_add_epi32(result2, temp0);

					// Next 4 bytes
					temp1 = _mm_cvtepu8_epi16(row);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result1 = _mm_add_epi32(result1, temp0);

					// Lowest 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result0 = _mm_add_epi32(result0, temp0);
				}
			}

			result0 = _mm_srli_epi32(result0, shift);
			result1 = _mm_srli_epi32(result1, shift);
			result2 = _mm_srli_epi32(result2, shift);
			result3 = _mm_srli_epi32(result3, shift);

			row = _mm_packs_epi32(result2, result3);
			temp0 = _mm_packs_epi32(result0, result1);
			_mm_store_si128(pLocalDst_xmm++, temp0);
			_mm_store_si128(pLocalDst_xmm++, row);

			pLocalSrc += 16;
			width--;
		}

		pLocalDst = (short *)pLocalDst_xmm;
		for (int w = 0; w < postfixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -4; j <= 4; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, SHRT_MAX);
			temp = max(temp, SHRT_MIN);
			*pLocalDst++ = (short)temp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes >> 1);

		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i *pLocalDst_xmm;
	unsigned char *pLocalSrc, *pLocalDst;
	short *pLocalConvMat;

	__m128i result0, result1, result2, result3, row, mul, temp0, temp1;
	__m128i zeromask = _mm_setzero_si128();

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;					// 16 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	int srcStride = (int)srcImageStrideInBytes;
	int rowLimit = (int)(convolutionHeight >> 1);
	int numConvCoeffs = 9 * (int)convolutionHeight;

	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		pLocalDst = (unsigned char *)pDstImage;

		for (int w = 0; w < prefixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -4; j <= 4; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, 255);
			temp = max(temp, 0);
			*pLocalDst++ = (unsigned char)temp;
		}

		pLocalDst_xmm = (__m128i *) pLocalDst;
		int width = (int)(alignedWidth >> 4);							// Each loop processess 16 pixels
		while (width)
		{
			pLocalConvMat = convMatrix + numConvCoeffs - 1;
			result0 = _mm_setzero_si128();
			result1 = _mm_setzero_si128();
			result2 = _mm_setzero_si128();
			result3 = _mm_setzero_si128();

			for (int y = -rowLimit; y <= rowLimit; y++)
			{
				for (int x = -4; x <= 4; x++)
				{
					row = _mm_loadu_si128((__m128i *)(pLocalSrc + (y * srcStride) + x));
					mul = _mm_set1_epi32((int)(*pLocalConvMat--));

					// Upper 4 bytes
					temp1 = _mm_unpackhi_epi8(row, zeromask);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result3 = _mm_add_epi32(result3, temp0);

					// Next 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result2 = _mm_add_epi32(result2, temp0);

					// Next 4 bytes
					temp1 = _mm_cvtepu8_epi16(row);
					temp0 = _mm_unpackhi_epi16(temp1, zeromask);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result1 = _mm_add_epi32(result1, temp0);

					// Lowest 4 bytes
					temp0 = _mm_cvtepi16_epi32(temp1);
					temp0 = _mm_mullo_epi32(temp0, mul);
					result0 = _mm_add_epi32(result0, temp0);
				}
			}

			result0 = _mm_srli_epi32(result0, shift);
			result1 = _mm_srli_epi32(result1, shift);
			result2 = _mm_srli_epi32(result2, shift);
			result3 = _mm_srli_epi32(result3, shift);

			row = _mm_packs_epi32(result2, result3);
			temp0 = _mm_packs_epi32(result0, result1);
			row = _mm_packus_epi16(temp0, row);
			_mm_store_si128(pLocalDst_xmm++, row);

			pLocalSrc += 16;
			width--;
		}

		pLocalDst = (unsigned char *)pLocalDst_xmm;
		for (int w = 0; w < postfixWidth; w++, pLocalSrc++)
		{
			int temp = 0;
			int idx = numConvCoeffs - 1;
			for (int i = -rowLimit; i <= rowLimit; i++)
			{
				for (int j = -4; j <= 4; j++)
				{
					temp += ((int)pLocalSrc[i*srcStride + j] * (int)convMatrix[idx--]);
				}
			}
			temp = min(temp, 255);
			temp = max(temp, 0);
			*pLocalDst++ = (unsigned char)temp;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;

		height--;
	}
	return AGO_SUCCESS;
}

static inline void CompareAndSwap(__m128i& p1, __m128i& p2)
{
	__m128i First = _mm_min_epu8(p1, p2);
	__m128i Sec = _mm_max_epu8(p1, p2);
	p1 = First;
	p2 = Sec;
}

int compareTwo(const void * a, const void * b)
{
	return(*(unsigned char *)a > *(unsigned char *)b ? 1 : -1);
}

int HafCpu_Median_U8_U8_3x3
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
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i pixels0, pixels1, pixels2, pixels3, pixels4, pixels5, pixels6, pixels7, pixels8;
	unsigned char *pLocalSrc, *pPrevSrc, *pNextSrc, *pLocalDst;
	unsigned char pixelArr[9];

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalDst = (unsigned char *)pDstImage;
		pLocalSrc = (unsigned char *)pSrcImage;
		pPrevSrc = pLocalSrc - srcImageStrideInBytes;
		pNextSrc = pLocalSrc + srcImageStrideInBytes;

		for (int x = 0; x < prefixWidth; x++, pLocalDst++, pLocalSrc++, pPrevSrc++, pNextSrc++)
		{
			pixelArr[0] = pPrevSrc[-1];
			pixelArr[1] = pPrevSrc[0];
			pixelArr[2] = pPrevSrc[1];
			pixelArr[3] = pLocalSrc[-1];
			pixelArr[4] = pLocalSrc[0];
			pixelArr[5] = pLocalSrc[1];
			pixelArr[6] = pNextSrc[-1];
			pixelArr[7] = pNextSrc[0];
			pixelArr[8] = pNextSrc[1];
			qsort(pixelArr, 9, sizeof(unsigned char), compareTwo);
			*pLocalDst = pixelArr[4];
		}
		
		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			pixels0 = _mm_loadu_si128((__m128i *)(pPrevSrc - 1));
			pixels1 = _mm_loadu_si128((__m128i *)(pPrevSrc));
			pixels2 = _mm_loadu_si128((__m128i *)(pPrevSrc + 1));
			pixels3 = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			pixels4 = _mm_loadu_si128((__m128i *)(pLocalSrc));
			pixels5 = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));
			pixels6 = _mm_loadu_si128((__m128i *)(pNextSrc - 1));
			pixels7 = _mm_loadu_si128((__m128i *)(pNextSrc));
			pixels8 = _mm_loadu_si128((__m128i *)(pNextSrc + 1));

			// sort by compare and swap : no branching required
			CompareAndSwap(pixels1, pixels2);
			CompareAndSwap(pixels4, pixels5);
			CompareAndSwap(pixels7, pixels8);
			CompareAndSwap(pixels0, pixels1);
			CompareAndSwap(pixels3, pixels4);
			CompareAndSwap(pixels6, pixels7);
			CompareAndSwap(pixels1, pixels2);
			CompareAndSwap(pixels4, pixels5);
			CompareAndSwap(pixels7, pixels8);
			CompareAndSwap(pixels0, pixels3);
			CompareAndSwap(pixels5, pixels8);
			CompareAndSwap(pixels4, pixels7);
			CompareAndSwap(pixels3, pixels6);
			CompareAndSwap(pixels1, pixels4);
			CompareAndSwap(pixels2, pixels5);
			CompareAndSwap(pixels4, pixels7);
			CompareAndSwap(pixels4, pixels2);
			CompareAndSwap(pixels6, pixels4);
			CompareAndSwap(pixels4, pixels2);

			// store median value
			_mm_store_si128((__m128i *)pLocalDst, pixels4);

			pPrevSrc += 16;
			pLocalSrc += 16;
			pNextSrc += 16;
			pLocalDst += 16;
		}
		
		for (int x = 0; x < postfixWidth; x++, pLocalDst++, pLocalSrc++, pPrevSrc++, pNextSrc++)
		{
			pixelArr[0] = pPrevSrc[-1];
			pixelArr[1] = pPrevSrc[0];
			pixelArr[2] = pPrevSrc[1];
			pixelArr[3] = pLocalSrc[-1];
			pixelArr[4] = pLocalSrc[0];
			pixelArr[5] = pLocalSrc[1];
			pixelArr[6] = pNextSrc[-1];
			pixelArr[7] = pNextSrc[0];
			pixelArr[8] = pNextSrc[1];
			qsort(pixelArr, 9, sizeof(unsigned char), compareTwo);
			*pLocalDst = pixelArr[4];
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_SobelPhase_U8_U8_3x3
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstPhaseImage,
		vx_uint32     dstPhaseImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8	* pScratch
	)
{
	// calculate Gx and Gy and compute phase
	vx_int16 *Gx, *Gy;
	vx_uint8 *scratchPad;
	vx_uint32 dstride = (dstWidth + 15)&~15;

	Gx = (vx_int16*)pScratch;
	Gy = (vx_int16*)(pScratch + (unsigned long)(((dstride + 15) & ~15) * dstHeight) * sizeof(vx_int16));
	scratchPad = pScratch + (unsigned long)(((dstride + 15) & ~15) * dstHeight) * sizeof(vx_int16) * 2;

	HafCpu_Sobel_S16S16_U8_3x3_GXY(dstWidth, dstHeight, Gx, dstride, Gy, dstride, pSrcImage, srcImageStrideInBytes, scratchPad);
	HafCpu_Phase_U8_S16S16(dstWidth, dstHeight, pDstPhaseImage, dstPhaseImageStrideInBytes, Gx, dstride, Gy, dstride);

	return AGO_SUCCESS;
}

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
	)
{
	vx_uint8 *pLocalSrc, *pLocalDstPhase;
	vx_int16 * pLocalDstMag;

	int prefixWidth = intptr_t(pDstMagImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i row0, row1, row2, shiftedR, shiftedL, temp, GxH, GxL, GyH, GyL;
	__m128i zeromask = _mm_setzero_si128();

	float scale = (float)128 / 180.f;					// For arctan

	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc = (vx_uint8 *)pSrcImage;
		pLocalDstMag = (vx_int16 *)pDstMagImage;
		pLocalDstPhase = (vx_uint8 *)pDstPhaseImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			vx_int16 tempGx = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1) +
				(vx_int16)pLocalSrc[(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[(int)srcImageStrideInBytes - 1];
			vx_int16 tempGy = (vx_int16)pLocalSrc[(int)srcImageStrideInBytes - 1] + ((vx_int16)pLocalSrc[(int)srcImageStrideInBytes] << 1) + (vx_int16)pLocalSrc[(int)srcImageStrideInBytes + 1] -
				(vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] - ((vx_int16)pLocalSrc[-(int)srcImageStrideInBytes] << 1) - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
			float mag = (float)(tempGx*tempGx) + (float)(tempGy*tempGy);
			mag = sqrtf(mag);
			*pLocalDstMag++ = (vx_int16)mag;

			float arct = HafCpu_FastAtan2_deg(tempGx, tempGy);
			*pLocalDstPhase++ = (vx_uint8)((vx_uint32)(arct*scale + 0.5) & 0xFF);
		}

		int width = (int)(alignedWidth >> 4);						// 16 pixels processed at a time
		while (width)
		{
			row0 = _mm_load_si128((__m128i *) (pLocalSrc - srcImageStrideInBytes));		// row above
			row1 = _mm_load_si128((__m128i *) pLocalSrc);								// current row
			row2 = _mm_load_si128((__m128i *) (pLocalSrc + srcImageStrideInBytes));		// row below

			// For the row below
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + srcImageStrideInBytes + 1));

			GxH = _mm_unpackhi_epi8(shiftedL, zeromask);				// Gx, H: -1 * (-1,1)
			GxL = _mm_cvtepu8_epi16(shiftedL);							// Gx, L: -1 * (-1,1)
			GyH = _mm_add_epi16(GxH, zeromask);							// Gy, H: 1 * (-1,1)
			GyL = _mm_add_epi16(GxL, zeromask);							// Gy, L: 1 * (-1,1)

			temp = _mm_unpackhi_epi8(row2, zeromask);
			temp = _mm_slli_epi16(temp, 1);								// Gy, H: 2 * (0,1)
			row2 = _mm_cvtepu8_epi16(row2);
			row2 = _mm_slli_epi16(row2, 1);								// Gy, L: 2 * (0,1)
			GyH = _mm_add_epi16(GyH, temp);
			GyL = _mm_add_epi16(GyL, row2);

			temp = _mm_unpackhi_epi8(shiftedR, zeromask);				// Gy, H: 1 * (1,1),	Gx, H: 1 * (1,1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);						// Gy, L: 1 * (1,1),	Gx, L: 1 * (1,1)
			GyH = _mm_add_epi16(GyH, temp);
			GyL = _mm_add_epi16(GyL, shiftedR);
			GxH = _mm_sub_epi16(temp, GxH);
			GxL = _mm_sub_epi16(shiftedR, GxL);

			// For the current row
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc + 1));

			temp = _mm_unpackhi_epi8(shiftedR, zeromask);
			temp = _mm_slli_epi16(temp, 1);								// Gx, H: 2 * (1,0)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);
			shiftedR = _mm_slli_epi16(shiftedR, 1);						// Gx, L: 2 * (1,0)
			GxH = _mm_add_epi16(GxH, temp);
			GxL = _mm_add_epi16(GxL, shiftedR);

			temp = _mm_unpackhi_epi8(shiftedL, zeromask);
			temp = _mm_slli_epi16(temp, 1);								// Gx, H: -2 * (-1,0)
			shiftedL = _mm_cvtepu8_epi16(shiftedL);
			shiftedL = _mm_slli_epi16(shiftedL, 1);						// Gx, L: -2 * (-1,0)
			GxH = _mm_sub_epi16(GxH, temp);
			GxL = _mm_sub_epi16(GxL, shiftedL);

			// For the row above
			shiftedL = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes - 1));
			shiftedR = _mm_loadu_si128((__m128i *)(pLocalSrc - srcImageStrideInBytes + 1));

			temp = _mm_unpackhi_epi8(shiftedR, zeromask);				// Gy, H: -1 * (1,-1),	Gx, H: 1 * (1,-1)
			shiftedR = _mm_cvtepu8_epi16(shiftedR);						// Gy, L: -1 * (1,-1),	Gx, L: 1 * (1,-1)
			GxH = _mm_add_epi16(GxH, temp);
			GxL = _mm_add_epi16(GxL, shiftedR);
			GyH = _mm_sub_epi16(GyH, temp);
			GyL = _mm_sub_epi16(GyL, shiftedR);

			temp = _mm_unpackhi_epi8(row0, zeromask);
			row0 = _mm_cvtepu8_epi16(row0);
			temp = _mm_slli_epi16(temp, 1);								// Gy, H: -2 * (0,-1)
			row0 = _mm_slli_epi16(row0, 1);								// Gy, L: -2 * (0,-1)
			GyH = _mm_sub_epi16(GyH, temp);
			GyL = _mm_sub_epi16(GyL, row0);

			temp = _mm_unpackhi_epi8(shiftedL, zeromask);				// Gy, H: -1 * (-1,-1),	Gx, H: -1 * (-1,-1)
			shiftedL = _mm_cvtepu8_epi16(shiftedL);						// Gy, L: -1 * (-1,-1),	Gx, L: -1 * (-1,-1)
			GxH = _mm_sub_epi16(GxH, temp);
			GxL = _mm_sub_epi16(GxL, shiftedL);
			GyH = _mm_sub_epi16(GyH, temp);
			GyL = _mm_sub_epi16(GyL, shiftedL);

			// Calculate phase
			for (int i = 0; i < 8; i++)
			{
				float arct = HafCpu_FastAtan2_deg(M128I(GxL).m128i_i16[i], M128I(GyL).m128i_i16[i]);
				*pLocalDstPhase++ = (vx_uint8)((vx_uint32)(arct*scale + 0.5) & 0xFF);
			}

			for (int i = 0; i < 8; i++)
			{
				float arct = HafCpu_FastAtan2_deg(M128I(GxH).m128i_i16[i], M128I(GyH).m128i_i16[i]);
				*pLocalDstPhase++ = (vx_uint8)((vx_uint32)(arct*scale + 0.5) & 0xFF);
			}

			// Magnitude
			row0 = _mm_srli_si128(GxH, 8);
			row1 = _mm_srli_si128(GxL, 8);
			row0 = _mm_cvtepi16_epi32(row0);							// GxH: Upper 4 words to dwords
			GxH = _mm_cvtepi16_epi32(GxH);								// GxH: Lower 4 words to dwords
			row1 = _mm_cvtepi16_epi32(row1);							// GxL: Upper 4 words to dwords
			GxL = _mm_cvtepi16_epi32(GxL);								// GxL: Lower 4 words to dwords

			row2 = _mm_srli_si128(GyH, 8);
			temp = _mm_srli_si128(GyL, 8);
			row2 = _mm_cvtepi16_epi32(row2);							// GyH: Upper 4 words to dwords
			GyH = _mm_cvtepi16_epi32(GyH);								// GyH: Lower 4 words to dwords
			temp = _mm_cvtepi16_epi32(temp);							// GyL: Upper 4 words to dwords
			GyL = _mm_cvtepi16_epi32(GyL);								// GyL: Lower 4 words to dwords

			row0 = _mm_mullo_epi32(row0, row0);							// Square
			GxH = _mm_mullo_epi32(GxH, GxH);
			row1 = _mm_mullo_epi32(row1, row1);
			GxL = _mm_mullo_epi32(GxL, GxL);
			row2 = _mm_mullo_epi32(row2, row2);
			GyH = _mm_mullo_epi32(GyH, GyH);
			temp = _mm_mullo_epi32(temp, temp);
			GyL = _mm_mullo_epi32(GyL, GyL);

			row0 = _mm_add_epi32(row0, row2);							// Add
			GxH = _mm_add_epi32(GxH, GyH);
			row1 = _mm_add_epi32(row1, temp);
			GxL = _mm_add_epi32(GxL, GyL);

			temp = _mm_srli_si128(row0, 8);
			__m128d d_pix1 = _mm_cvtepi32_pd(temp);						// Pixels 15, 14
			__m128d d_pix0 = _mm_cvtepi32_pd(row0);						// Pixels 13, 12
			d_pix1 = _mm_sqrt_pd(d_pix1);
			d_pix0 = _mm_sqrt_pd(d_pix0);
			row0 = _mm_cvtpd_epi32(d_pix1);
			temp = _mm_cvtpd_epi32(d_pix0);
			row0 = _mm_slli_si128(row0, 8);
			row0 = _mm_or_si128(row0, temp);							// Pixels 15, 14, 13, 12 (DWORDS)

			temp = _mm_srli_si128(GxH, 8);
			d_pix1 = _mm_cvtepi32_pd(temp);								// Pixels 11, 10
			d_pix0 = _mm_cvtepi32_pd(GxH);								// Pixels 9, 8
			d_pix1 = _mm_sqrt_pd(d_pix1);
			d_pix0 = _mm_sqrt_pd(d_pix0);
			GxH = _mm_cvtpd_epi32(d_pix1);
			temp = _mm_cvtpd_epi32(d_pix0);
			GxH = _mm_slli_si128(GxH, 8);
			GxH = _mm_or_si128(GxH, temp);								// Pixels 11, 10, 9, 8 (DWORDS)
			row0 = _mm_packus_epi32(GxH, row0);							// Pixels 15, 14, 13, 12, 11, 10, 9, 8 (WORDS)

			temp = _mm_srli_si128(row1, 8);
			d_pix1 = _mm_cvtepi32_pd(temp);								// Pixels 7, 6
			d_pix0 = _mm_cvtepi32_pd(row1);								// Pixels 5, 4
			d_pix1 = _mm_sqrt_pd(d_pix1);
			d_pix0 = _mm_sqrt_pd(d_pix0);
			row1 = _mm_cvtpd_epi32(d_pix1);
			temp = _mm_cvtpd_epi32(d_pix0);
			row1 = _mm_slli_si128(row1, 8);
			row1 = _mm_or_si128(row1, temp);							// Pixels 7, 6, 5, 4 (DWORDS)

			temp = _mm_srli_si128(GxL, 8);
			d_pix1 = _mm_cvtepi32_pd(temp);								// Pixels 3, 2
			d_pix0 = _mm_cvtepi32_pd(GxL);								// Pixels 1, 0
			d_pix1 = _mm_sqrt_pd(d_pix1);
			d_pix0 = _mm_sqrt_pd(d_pix0);
			GxL = _mm_cvtpd_epi32(d_pix1);
			temp = _mm_cvtpd_epi32(d_pix0);
			GxL = _mm_slli_si128(GxL, 8);
			GxL = _mm_or_si128(GxL, temp);								// Pixels 3, 2, 1, 0 (DWORDS)
			row1 = _mm_packus_epi32(GxL, row1);							// Pixels 7, 6, 5, 4, 3, 2, 1, 0 (WORDS)

			_mm_store_si128((__m128i *) pLocalDstMag, row1);
			_mm_store_si128((__m128i *) (pLocalDstMag + 8), row0);

			pLocalSrc += 16;
			pLocalDstMag += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			vx_int16 tempGx = (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] + (((vx_int16)pLocalSrc[1] - (vx_int16)pLocalSrc[-1]) << 1) +
				(vx_int16)pLocalSrc[(int)srcImageStrideInBytes + 1] - (vx_int16)pLocalSrc[(int)srcImageStrideInBytes - 1];
			vx_int16 tempGy = (vx_int16)pLocalSrc[(int)srcImageStrideInBytes - 1] + ((vx_int16)pLocalSrc[(int)srcImageStrideInBytes] << 1) + (vx_int16)pLocalSrc[(int)srcImageStrideInBytes + 1] -
				(vx_int16)pLocalSrc[-(int)srcImageStrideInBytes - 1] - ((vx_int16)pLocalSrc[-(int)srcImageStrideInBytes] << 1) - (vx_int16)pLocalSrc[-(int)srcImageStrideInBytes + 1];
			float mag = (float)(tempGx*tempGx) + (float)(tempGy*tempGy);
			mag = sqrtf(mag);
			*pLocalDstMag++ = (vx_int16)mag;

			float arct = HafCpu_FastAtan2_deg(tempGx, tempGy);
			*pLocalDstPhase++ = (vx_uint8)((vx_uint32)(arct*scale + 0.5) & 0xFF);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstMagImage += (dstMagImageStrideInBytes >> 1);
		pDstPhaseImage += dstPhaseImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}