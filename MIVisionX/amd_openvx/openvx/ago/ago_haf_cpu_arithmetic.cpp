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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m256i pixels1, pixels2;

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels1 = _mm256_add_epi8(pixels1, pixels2);
				_mm256_store_si256(pLocalDst_ymm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 temp = (vx_int16)(*pLocalSrc1++) + (vx_int16)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_loadu_si256(pLocalSrc2_ymm++);
				pixels1 = _mm256_add_epi8(pixels1, pixels2);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 temp = (vx_int16)(*pLocalSrc1++) + (vx_int16)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m128i pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_add_epi8(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 temp = (vx_int16)(*pLocalSrc1++) + (vx_int16)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_add_epi8(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 temp = (vx_int16)(*pLocalSrc1++) + (vx_int16)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m256i pixels1, pixels2;

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels1 = _mm256_adds_epu8(pixels1, pixels2);
				_mm256_store_si256(pLocalDst_ymm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				int temp = (int)(*pLocalSrc1++) + (int)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)min(temp, UINT8_MAX);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		{
			for (int height = 0; height < (int)dstHeight; height++)
			{
				pLocalSrc1_ymm = (__m256i*) pSrcImage1;
				pLocalSrc2_ymm = (__m256i*) pSrcImage2;
				pLocalDst_ymm = (__m256i*) pDstImage;

				for (int width = 0; width < alignedWidth; width += 32)
				{
					pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
					pixels2 = _mm256_loadu_si256(pLocalSrc2_ymm++);
					pixels1 = _mm256_adds_epu8(pixels1, pixels2);
					_mm256_storeu_si256(pLocalDst_ymm++, pixels1);
				}

				pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
				pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
				pLocalDst = (vx_uint8 *)pLocalDst_ymm;

				for (int width = 0; width < postfixWidth; width++)
				{
					int temp = (int)(*pLocalSrc1++) + (int)(*pLocalSrc2++);
					*pLocalDst++ = (vx_uint8)min(temp, UINT8_MAX);
				}

				pSrcImage1 += srcImage1StrideInBytes;
				pSrcImage2 += srcImage2StrideInBytes;
				pDstImage += dstImageStrideInBytes;
			}
		}
	}

#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m128i pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_adds_epu8(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}
			
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				int temp = (int)(*pLocalSrc1++) + (int)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8) min(temp, UINT8_MAX);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		{
			for (int height = 0; height < (int)dstHeight; height++)
			{
				pLocalSrc1_xmm = (__m128i*) pSrcImage1;
				pLocalSrc2_xmm = (__m128i*) pSrcImage2;
				pLocalDst_xmm = (__m128i*) pDstImage;

				for (int width = 0; width < alignedWidth; width += 16)
				{
					pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
					pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
					pixels1 = _mm_adds_epu8(pixels1, pixels2);
					_mm_storeu_si128(pLocalDst_xmm++, pixels1);
				}

				pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
				pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
				pLocalDst = (vx_uint8 *)pLocalDst_xmm;

				for (int width = 0; width < postfixWidth; width++)
				{
					int temp = (int)(*pLocalSrc1++) + (int)(*pLocalSrc2++);
					*pLocalDst++ = (vx_uint8)min(temp, UINT8_MAX);
				}

				pSrcImage1 += srcImage1StrideInBytes;
				pSrcImage2 += srcImage2StrideInBytes;
				pDstImage += dstImageStrideInBytes;
			}
		}
	}
#endif
	
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m256i pixels1, pixels2;

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels1 = _mm256_sub_epi8(pixels1, pixels2);
				_mm256_store_si256(pLocalDst_ymm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 temp = (vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_loadu_si256(pLocalSrc2_ymm++);
				pixels1 = _mm256_sub_epi8(pixels1, pixels2);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				int temp = (int)(*pLocalSrc1++) - (int)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m128i pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_sub_epi8(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 temp = (vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_sub_epi8(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				int temp = (int)(*pLocalSrc1++) - (int)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m256i pixels1, pixels2;

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels1 = _mm256_subs_epu8(pixels1, pixels2);
				_mm256_store_si256(pLocalDst_ymm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				int temp = (int)(*pLocalSrc1++) - (int)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)max(min(temp, UINT8_MAX), 0);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_loadu_si256(pLocalSrc2_ymm++);
				pixels1 = _mm256_subs_epu8(pixels1, pixels2);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				int temp = (int)(*pLocalSrc1++) - (int)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)max(min(temp, UINT8_MAX), 0);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m128i pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_subs_epu8(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				int temp = (int)(*pLocalSrc1++) - (int)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)max(min(temp, UINT8_MAX), 0);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_subs_epu8(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				int temp = (int)(*pLocalSrc1++) - (int)(*pLocalSrc2++);
				*pLocalDst++ = (vx_uint8)max(min(temp, UINT8_MAX), 0);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
#endif	
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int16 *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
				pixels2L = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_add_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_add_epi16(pixels1H, pixels2H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
				_mm256_store_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (int)(*pLocalSrc1++) + (int)(*pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		{
			for (int height = 0; height < (int)dstHeight; height++)
			{
				pLocalSrc1_ymm = (__m256i*) pSrcImage1;
				pLocalSrc2_ymm = (__m256i*) pSrcImage2;
				pLocalDst_ymm = (__m256i*) pDstImage;

				for (int width = 0; width < alignedWidth; width += 32)
				{
					pixels1L = _mm256_loadu_si256(pLocalSrc1_ymm++);
					pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
					pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
					pixels2L = _mm256_loadu_si256(pLocalSrc2_ymm++);
					pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
					pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
					pixels1L = _mm256_add_epi16(pixels1L, pixels2L);
					pixels1H = _mm256_add_epi16(pixels1H, pixels2H);
					_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
					_mm256_storeu_si256(pLocalDst_ymm++, pixels1H);
				}

				pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
				pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
				pLocalDst = (vx_int16 *)pLocalDst_ymm;

				for (int width = 0; width < postfixWidth; width++)
				{
					*pLocalDst++ = (int)(*pLocalSrc1++) + (int)(*pLocalSrc2++);
				}

				pSrcImage1 += srcImage1StrideInBytes;
				pSrcImage2 += srcImage2StrideInBytes;
				pDstImage += (dstImageStrideInBytes >> 1);
			}
		}
	}

#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int16 *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int) dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc1_xmm++);
				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_load_si128(pLocalSrc2_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_add_epi16(pixels1L, pixels2L);
				pixels1H = _mm_add_epi16(pixels1H, pixels2H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
				_mm_store_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (int)(*pLocalSrc1++) + (int)(*pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		{
			for (int height = 0; height < (int) dstHeight; height++)
			{
				pLocalSrc1_xmm = (__m128i*) pSrcImage1;
				pLocalSrc2_xmm = (__m128i*) pSrcImage2;
				pLocalDst_xmm = (__m128i*) pDstImage;

				for (int width = 0; width < alignedWidth; width += 16)
				{
					pixels1L = _mm_loadu_si128(pLocalSrc1_xmm++);
					pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
					pixels1L = _mm_cvtepu8_epi16(pixels1L);
					pixels2L = _mm_loadu_si128(pLocalSrc2_xmm++);
					pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
					pixels2L = _mm_cvtepu8_epi16(pixels2L);
					pixels1L = _mm_add_epi16(pixels1L, pixels2L);
					pixels1H = _mm_add_epi16(pixels1H, pixels2H);
					_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
					_mm_storeu_si128(pLocalDst_xmm++, pixels1H);
				}

				pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
				pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
				pLocalDst = (vx_int16 *)pLocalDst_xmm;

				for (int width = 0; width < postfixWidth; width++)
				{
					*pLocalDst++ = (int)(*pLocalSrc1++) + (int)(*pLocalSrc2++);
				}

				pSrcImage1 += srcImage1StrideInBytes;
				pSrcImage2 += srcImage2StrideInBytes;
				pDstImage += (dstImageStrideInBytes >> 1);
			}
		}
	}
#endif	
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int16 *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
				pixels2L = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_sub_epi16(pixels1H, pixels2H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
				_mm256_store_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
				pixels2L = _mm256_loadu_si256(pLocalSrc2_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_sub_epi16(pixels1H, pixels2H);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (int)(*pLocalSrc1++) - (int)(*pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}

#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int16 *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc1_xmm++);
				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_load_si128(pLocalSrc2_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm_sub_epi16(pixels1H, pixels2H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
				_mm_store_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm_sub_epi16(pixels1H, pixels2H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (int)(*pLocalSrc1++) - (int)(*pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc16_ymm, *pLocalSrc8_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (alignedWidth)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_ymm = (__m256i*) pSrcImage1;
			pLocalSrc8_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc16_ymm++);
				pixels1H = _mm256_load_si256(pLocalSrc16_ymm++);;
				pixels2L = _mm256_load_si256(pLocalSrc8_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_add_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_add_epi16(pixels1H, pixels2H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
				_mm256_store_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc16++ + (vx_int16)(*pLocalSrc8++);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_ymm = (__m256i*) pSrcImage1;
			pLocalSrc8_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_loadu_si256(pLocalSrc16_ymm++);
				pixels1H = _mm256_loadu_si256(pLocalSrc16_ymm++);;
				pixels2L = _mm256_loadu_si256(pLocalSrc8_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_add_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_add_epi16(pixels1H, pixels2H);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc16++ + (vx_int16)(*pLocalSrc8++);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else	
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc16_xmm, *pLocalSrc8_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (alignedWidth)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_xmm = (__m128i*) pSrcImage1;
			pLocalSrc8_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc16_xmm++);
				pixels1H = _mm_load_si128(pLocalSrc16_xmm++);;
				pixels2L = _mm_load_si128(pLocalSrc8_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_add_epi16(pixels1L, pixels2L);
				pixels1H = _mm_add_epi16(pixels1H, pixels2H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
				_mm_store_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc16++ + (vx_int16)(*pLocalSrc8++);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_xmm = (__m128i*) pSrcImage1;
			pLocalSrc8_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc16_xmm++);
				pixels1H = _mm_loadu_si128(pLocalSrc16_xmm++);;
				pixels2L = _mm_loadu_si128(pLocalSrc8_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_add_epi16(pixels1L, pixels2L);
				pixels1H = _mm_add_epi16(pixels1H, pixels2H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc16++ + (vx_int16)(*pLocalSrc8++);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc16_ymm, *pLocalSrc8_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_ymm = (__m256i*) pSrcImage1;
			pLocalSrc8_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc16_ymm++);
				pixels1H = _mm256_load_si256(pLocalSrc16_ymm++);;
				pixels2L = _mm256_load_si256(pLocalSrc8_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_adds_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_adds_epi16(pixels1H, pixels2H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
				_mm256_store_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc16++) + (vx_int32)(*pLocalSrc8++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_ymm = (__m256i*) pSrcImage1;
			pLocalSrc8_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_loadu_si256(pLocalSrc16_ymm++);
				pixels1H = _mm256_loadu_si256(pLocalSrc16_ymm++);;
				pixels2L = _mm256_loadu_si256(pLocalSrc8_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_adds_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_adds_epi16(pixels1H, pixels2H);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc16++) + (vx_int32)(*pLocalSrc8++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc16_xmm, *pLocalSrc8_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_xmm = (__m128i*) pSrcImage1;
			pLocalSrc8_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc16_xmm++);
				pixels1H = _mm_load_si128(pLocalSrc16_xmm++);;
				pixels2L = _mm_load_si128(pLocalSrc8_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_adds_epi16(pixels1L, pixels2L);
				pixels1H = _mm_adds_epi16(pixels1H, pixels2H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
				_mm_store_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc16++) + (vx_int32)(*pLocalSrc8++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_xmm = (__m128i*) pSrcImage1;
			pLocalSrc8_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc16_xmm++);
				pixels1H = _mm_loadu_si128(pLocalSrc16_xmm++);;
				pixels2L = _mm_loadu_si128(pLocalSrc8_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_adds_epi16(pixels1L, pixels2L);
				pixels1H = _mm_adds_epi16(pixels1H, pixels2H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc16++) + (vx_int32)(*pLocalSrc8++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc16_ymm, *pLocalSrc8_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_ymm = (__m256i*) pSrcImage1;
			pLocalSrc8_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc16_ymm++);
				pixels1H = _mm256_load_si256(pLocalSrc16_ymm++);;
				pixels2L = _mm256_load_si256(pLocalSrc8_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_sub_epi16(pixels1H, pixels2H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
				_mm256_store_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc16++ - (vx_int16)(*pLocalSrc8++);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);

		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_ymm = (__m256i*) pSrcImage1;
			pLocalSrc8_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_loadu_si256(pLocalSrc16_ymm++);
				pixels1H = _mm256_loadu_si256(pLocalSrc16_ymm++);;
				pixels2L = _mm256_loadu_si256(pLocalSrc8_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_sub_epi16(pixels1H, pixels2H);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc16++ - (vx_int16)(*pLocalSrc8++);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);

		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc16_xmm, *pLocalSrc8_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_xmm = (__m128i*) pSrcImage1;
			pLocalSrc8_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc16_xmm++);
				pixels1H = _mm_load_si128(pLocalSrc16_xmm++);;
				pixels2L = _mm_load_si128(pLocalSrc8_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm_sub_epi16(pixels1H, pixels2H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
				_mm_store_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc16++ - (vx_int16)(*pLocalSrc8++);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);

		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_xmm = (__m128i*) pSrcImage1;
			pLocalSrc8_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc16_xmm++);
				pixels1H = _mm_loadu_si128(pLocalSrc16_xmm++);;
				pixels2L = _mm_loadu_si128(pLocalSrc8_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm_sub_epi16(pixels1H, pixels2H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc16++ - (vx_int16)(*pLocalSrc8++);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);

		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc16_ymm, *pLocalSrc8_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_ymm = (__m256i*) pSrcImage1;
			pLocalSrc8_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc16_ymm++);
				pixels1H = _mm256_load_si256(pLocalSrc16_ymm++);;
				pixels2L = _mm256_load_si256(pLocalSrc8_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_subs_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_subs_epi16(pixels1H, pixels2H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
				_mm256_store_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc16++) - (vx_int32)(*pLocalSrc8++);
				*pLocalDst++ = max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_ymm = (__m256i*) pSrcImage1;
			pLocalSrc8_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_loadu_si256(pLocalSrc16_ymm++);
				pixels1H = _mm256_loadu_si256(pLocalSrc16_ymm++);;
				pixels2L = _mm256_loadu_si256(pLocalSrc8_ymm++);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);
				pixels1L = _mm256_subs_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_subs_epi16(pixels1H, pixels2H);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc16++) - (vx_int32)(*pLocalSrc8++);
				*pLocalDst++ = max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc16_xmm, *pLocalSrc8_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_xmm = (__m128i*) pSrcImage1;
			pLocalSrc8_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc16_xmm++);
				pixels1H = _mm_load_si128(pLocalSrc16_xmm++);;
				pixels2L = _mm_load_si128(pLocalSrc8_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_subs_epi16(pixels1L, pixels2L);
				pixels1H = _mm_subs_epi16(pixels1H, pixels2H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
				_mm_store_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc16++) - (vx_int32)(*pLocalSrc8++);
				*pLocalDst++ = max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc16_xmm = (__m128i*) pSrcImage1;
			pLocalSrc8_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc16_xmm++);
				pixels1H = _mm_loadu_si128(pLocalSrc16_xmm++);;
				pixels2L = _mm_loadu_si128(pLocalSrc8_xmm++);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);
				pixels1L = _mm_subs_epi16(pixels1L, pixels2L);
				pixels1H = _mm_subs_epi16(pixels1H, pixels2H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc16++) - (vx_int32)(*pLocalSrc8++);
				*pLocalDst++ = max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc16_ymm, *pLocalSrc8_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc8_ymm = (__m256i*) pSrcImage1;
			pLocalSrc16_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc8_ymm++);
				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
				pixels2L = _mm256_load_si256(pLocalSrc16_ymm++);
				pixels2H = _mm256_load_si256(pLocalSrc16_ymm++);;
				pixels1L = _mm256_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_sub_epi16(pixels1H, pixels2H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
				_mm256_store_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_int16)(*pLocalSrc8++) - *pLocalSrc16++;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc8_ymm = (__m256i*) pSrcImage1;
			pLocalSrc16_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_loadu_si256(pLocalSrc8_ymm++);
				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm256_unpacklo_epi16(pixels1L, zeromask);
				pixels2L = _mm256_loadu_si256(pLocalSrc16_ymm++);
				pixels2H = _mm256_loadu_si256(pLocalSrc16_ymm++);;
				pixels1L = _mm256_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_sub_epi16(pixels1H, pixels2H);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_int16)(*pLocalSrc8++) - *pLocalSrc16++;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc16_xmm, *pLocalSrc8_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc8_xmm = (__m128i*) pSrcImage1;
			pLocalSrc16_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc8_xmm++);
				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_load_si128(pLocalSrc16_xmm++);
				pixels2H = _mm_load_si128(pLocalSrc16_xmm++);;
				pixels1L = _mm_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm_sub_epi16(pixels1H, pixels2H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
				_mm_store_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_int16)(*pLocalSrc8++) - *pLocalSrc16++;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc8_xmm = (__m128i*) pSrcImage1;
			pLocalSrc16_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc8_xmm++);
				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_loadu_si128(pLocalSrc16_xmm++);
				pixels2H = _mm_loadu_si128(pLocalSrc16_xmm++);;
				pixels1L = _mm_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm_sub_epi16(pixels1H, pixels2H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_int16)(*pLocalSrc8++) - *pLocalSrc16++;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc16_ymm, *pLocalSrc8_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc8_ymm = (__m256i*) pSrcImage1;
			pLocalSrc16_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc8_ymm++);
				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
				pixels2L = _mm256_load_si256(pLocalSrc16_ymm++);
				pixels2H = _mm256_load_si256(pLocalSrc16_ymm++);;
				pixels1L = _mm256_subs_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_subs_epi16(pixels1H, pixels2H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
				_mm256_store_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc8++) - (vx_int32)(*pLocalSrc16++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc8_ymm = (__m256i*) pSrcImage1;
			pLocalSrc16_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_loadu_si256(pLocalSrc8_ymm++);
				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
				pixels2L = _mm256_loadu_si256(pLocalSrc16_ymm++);
				pixels2H = _mm256_loadu_si256(pLocalSrc16_ymm++);;
				pixels1L = _mm256_subs_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_subs_epi16(pixels1H, pixels2H);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_ymm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc8++) - (vx_int32)(*pLocalSrc16++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc16_xmm, *pLocalSrc8_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc8;
	vx_int16 *pLocalSrc16, *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc8_xmm = (__m128i*) pSrcImage1;
			pLocalSrc16_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc8_xmm++);
				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_load_si128(pLocalSrc16_xmm++);
				pixels2H = _mm_load_si128(pLocalSrc16_xmm++);;
				pixels1L = _mm_subs_epi16(pixels1L, pixels2L);
				pixels1H = _mm_subs_epi16(pixels1H, pixels2H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
				_mm_store_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc8++) - (vx_int32)(*pLocalSrc16++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc8_xmm = (__m128i*) pSrcImage1;
			pLocalSrc16_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc8_xmm++);
				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_loadu_si128(pLocalSrc16_xmm++);
				pixels2H = _mm_loadu_si128(pLocalSrc16_xmm++);;
				pixels1L = _mm_subs_epi16(pixels1L, pixels2L);
				pixels1H = _mm_subs_epi16(pixels1H, pixels2H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1H);
			}

			pLocalSrc16 = (vx_int16 *)pLocalSrc16_xmm;
			pLocalSrc8 = (vx_uint8 *)pLocalSrc8_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc8++) - (vx_int32)(*pLocalSrc16++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m256i pixels1, pixels2, pixels3, pixels4;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels3 = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels4 = _mm256_load_si256(pLocalSrc2_ymm++);

				pixels1 = _mm256_add_epi16(pixels1, pixels3);
				pixels2 = _mm256_add_epi16(pixels2, pixels4);

				_mm256_store_si256(pLocalDst_ymm++, pixels1);
				_mm256_store_si256(pLocalDst_ymm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) + (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels3 = _mm256_loadu_si256(pLocalSrc2_ymm++);
				pixels4 = _mm256_loadu_si256(pLocalSrc2_ymm++);

				pixels1 = _mm256_add_epi16(pixels1, pixels3);
				pixels2 = _mm256_add_epi16(pixels2, pixels4);

				_mm256_storeu_si256(pLocalDst_ymm++, pixels1);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) + (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m128i pixels1, pixels2, pixels3, pixels4;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels3 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels4 = _mm_load_si128(pLocalSrc2_xmm++);

				pixels1 = _mm_add_epi16(pixels1, pixels3);
				pixels2 = _mm_add_epi16(pixels2, pixels4);

				_mm_store_si128(pLocalDst_xmm++, pixels1);
				_mm_store_si128(pLocalDst_xmm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) + (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels3 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels4 = _mm_loadu_si128(pLocalSrc2_xmm++);

				pixels1 = _mm_add_epi16(pixels1, pixels3);
				pixels2 = _mm_add_epi16(pixels2, pixels4);

				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
				_mm_storeu_si128(pLocalDst_xmm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) + (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m256i pixels1, pixels2, pixels3, pixels4;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels3 = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels4 = _mm256_load_si256(pLocalSrc2_ymm++);

				pixels1 = _mm256_adds_epi16(pixels1, pixels3);
				pixels2 = _mm256_adds_epi16(pixels2, pixels4);

				_mm256_store_si256(pLocalDst_ymm++, pixels1);
				_mm256_store_si256(pLocalDst_ymm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) + (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels3 = _mm256_loadu_si256(pLocalSrc2_ymm++);
				pixels4 = _mm256_loadu_si256(pLocalSrc2_ymm++);

				pixels1 = _mm256_adds_epi16(pixels1, pixels3);
				pixels2 = _mm256_adds_epi16(pixels2, pixels4);

				_mm256_storeu_si256(pLocalDst_ymm++, pixels1);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) + (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m128i pixels1, pixels2, pixels3, pixels4;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels3 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels4 = _mm_load_si128(pLocalSrc2_xmm++);

				pixels1 = _mm_adds_epi16(pixels1, pixels3);
				pixels2 = _mm_adds_epi16(pixels2, pixels4);

				_mm_store_si128(pLocalDst_xmm++, pixels1);
				_mm_store_si128(pLocalDst_xmm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) + (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels3 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels4 = _mm_loadu_si128(pLocalSrc2_xmm++);

				pixels1 = _mm_adds_epi16(pixels1, pixels3);
				pixels2 = _mm_adds_epi16(pixels2, pixels4);

				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
				_mm_storeu_si128(pLocalDst_xmm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) + (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m256i pixels1, pixels2, pixels3, pixels4;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels3 = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels4 = _mm256_load_si256(pLocalSrc2_ymm++);

				pixels1 = _mm256_sub_epi16(pixels1, pixels3);
				pixels2 = _mm256_sub_epi16(pixels2, pixels4);

				_mm256_store_si256(pLocalDst_ymm++, pixels1);
				_mm256_store_si256(pLocalDst_ymm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) - (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels3 = _mm256_loadu_si256(pLocalSrc2_ymm++);
				pixels4 = _mm256_loadu_si256(pLocalSrc2_ymm++);

				pixels1 = _mm256_sub_epi16(pixels1, pixels3);
				pixels2 = _mm256_sub_epi16(pixels2, pixels4);

				_mm256_storeu_si256(pLocalDst_ymm++, pixels1);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) - (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m128i pixels1, pixels2, pixels3, pixels4;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels3 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels4 = _mm_load_si128(pLocalSrc2_xmm++);

				pixels1 = _mm_sub_epi16(pixels1, pixels3);
				pixels2 = _mm_sub_epi16(pixels2, pixels4);

				_mm_store_si128(pLocalDst_xmm++, pixels1);
				_mm_store_si128(pLocalDst_xmm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) - (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels3 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels4 = _mm_loadu_si128(pLocalSrc2_xmm++);

				pixels1 = _mm_sub_epi16(pixels1, pixels3);
				pixels2 = _mm_sub_epi16(pixels2, pixels4);

				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
				_mm_storeu_si128(pLocalDst_xmm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) - (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m256i pixels1, pixels2, pixels3, pixels4;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels3 = _mm256_load_si256(pLocalSrc2_ymm++);
				pixels4 = _mm256_load_si256(pLocalSrc2_ymm++);

				pixels1 = _mm256_subs_epi16(pixels1, pixels3);
				pixels2 = _mm256_subs_epi16(pixels2, pixels4);

				_mm256_store_si256(pLocalDst_ymm++, pixels1);
				_mm256_store_si256(pLocalDst_ymm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) - (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels2 = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels3 = _mm256_loadu_si256(pLocalSrc2_ymm++);
				pixels4 = _mm256_loadu_si256(pLocalSrc2_ymm++);

				pixels1 = _mm256_subs_epi16(pixels1, pixels3);
				pixels2 = _mm256_subs_epi16(pixels2, pixels4);

				_mm256_storeu_si256(pLocalDst_ymm++, pixels1);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_ymm;
			pLocalDst = (vx_int16 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) - (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m128i pixels1, pixels2, pixels3, pixels4;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels3 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels4 = _mm_load_si128(pLocalSrc2_xmm++);

				pixels1 = _mm_subs_epi16(pixels1, pixels3);
				pixels2 = _mm_subs_epi16(pixels2, pixels4);

				_mm_store_si128(pLocalDst_xmm++, pixels1);
				_mm_store_si128(pLocalDst_xmm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) - (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16) max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels3 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels4 = _mm_loadu_si128(pLocalSrc2_xmm++);

				pixels1 = _mm_subs_epi16(pixels1, pixels3);
				pixels2 = _mm_subs_epi16(pixels2, pixels4);

				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
				_mm_storeu_si128(pLocalDst_xmm++, pixels2);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32)(*pLocalSrc1++) - (vx_int32)(*pLocalSrc2++);
				*pLocalDst++ = (vx_int16) max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
#if USE_AVX
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0x1F) == 0) ? true : false;

	__m256i *pLocalSrc1_ymm, *pLocalSrc2_ymm, *pLocalDst_ymm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m256i pixels1H, pixels1L, pixels2H, pixels2L;
	__m256i zeromask = _mm256_setzero_si256();

	int alignedWidth = dstWidth & ~31;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_load_si256(pLocalSrc1_ymm++);
				pixels2L = _mm256_load_si256(pLocalSrc2_ymm++);

				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);

				pixels1H = _mm256_sub_epi16(pixels1H, pixels2H);
				pixels1L = _mm256_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_abs_epi16(pixels1H);
				pixels1L = _mm256_abs_epi16(pixels1L);

				pixels1L = _mm256_packus_epi16(pixels1L, pixels1H);
				_mm256_store_si256(pLocalDst_ymm++, pixels1L);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_uint8)abs((vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++));
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_ymm = (__m256i*) pSrcImage1;
			pLocalSrc2_ymm = (__m256i*) pSrcImage2;
			pLocalDst_ymm = (__m256i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 32)
			{
				pixels1L = _mm256_loadu_si256(pLocalSrc1_ymm++);
				pixels2L = _mm256_loadu_si256(pLocalSrc2_ymm++);

				pixels1H = _mm256_unpackhi_epi8(pixels1L, zeromask);
				pixels2H = _mm256_unpackhi_epi8(pixels2L, zeromask);
				pixels1L = _mm256_unpacklo_epi8(pixels1L, zeromask);
				pixels2L = _mm256_unpacklo_epi8(pixels2L, zeromask);

				pixels1H = _mm256_sub_epi16(pixels1H, pixels2H);
				pixels1L = _mm256_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm256_abs_epi16(pixels1H);
				pixels1L = _mm256_abs_epi16(pixels1L);

				pixels1L = _mm256_packus_epi16(pixels1L, pixels1H);
				_mm256_storeu_si256(pLocalDst_ymm++, pixels1L);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_ymm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_ymm;
			pLocalDst = (vx_uint8 *)pLocalDst_ymm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_uint8)abs((vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++));
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
#else
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2L = _mm_load_si128(pLocalSrc2_xmm++);

				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);

				pixels1H = _mm_sub_epi16(pixels1H, pixels2H);
				pixels1L = _mm_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm_abs_epi16(pixels1H);
				pixels1L = _mm_abs_epi16(pixels1L);

				pixels1L = _mm_packus_epi16(pixels1L, pixels1H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_uint8)abs((vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++));
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2L = _mm_loadu_si128(pLocalSrc2_xmm++);

				pixels1H = _mm_unpackhi_epi8(pixels1L, zeromask);
				pixels2H = _mm_unpackhi_epi8(pixels2L, zeromask);
				pixels1L = _mm_cvtepu8_epi16(pixels1L);
				pixels2L = _mm_cvtepu8_epi16(pixels2L);

				pixels1H = _mm_sub_epi16(pixels1H, pixels2H);
				pixels1L = _mm_sub_epi16(pixels1L, pixels2L);
				pixels1H = _mm_abs_epi16(pixels1H);
				pixels1L = _mm_abs_epi16(pixels1L);

				pixels1L = _mm_packus_epi16(pixels1L, pixels1H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = (vx_uint8)abs((vx_int16)(*pLocalSrc1++) - (vx_int16)(*pLocalSrc2++));
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
#endif
	return AGO_SUCCESS;
}

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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;

	__m128i pixels1H, pixels1L, pixels2H, pixels2L;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~7;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 8)
			{
				pixels1L = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2L = _mm_load_si128(pLocalSrc2_xmm++);

				pixels1H = _mm_srli_si128(pixels1L, 8);
				pixels1H = _mm_cvtepi16_epi32(pixels1H);
				pixels1L = _mm_cvtepi16_epi32(pixels1L);
				pixels2H = _mm_srli_si128(pixels2L, 8);
				pixels2H = _mm_cvtepi16_epi32(pixels2H);
				pixels2L = _mm_cvtepi16_epi32(pixels2L);
				
				pixels1H = _mm_sub_epi32(pixels1H, pixels2H);
				pixels1L = _mm_sub_epi32(pixels1L, pixels2L);
				pixels1H = _mm_abs_epi32(pixels1H);
				pixels1L = _mm_abs_epi32(pixels1L);
				
				pixels1L = _mm_packs_epi32(pixels1L, pixels1H);
				_mm_store_si128(pLocalDst_xmm++, pixels1L);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32) abs((vx_int32) (*pLocalSrc1++) - (vx_int32) (*pLocalSrc2++));
				*pLocalDst++ =  (vx_int16) max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i*) pSrcImage1;
			pLocalSrc2_xmm = (__m128i*) pSrcImage2;
			pLocalDst_xmm = (__m128i*) pDstImage;

			for (int width = 0; width < alignedWidth; width += 8)
			{
				pixels1L = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2L = _mm_loadu_si128(pLocalSrc2_xmm++);

				pixels1H = _mm_srli_si128(pixels1L, 8);
				pixels1H = _mm_cvtepi16_epi32(pixels1H);
				pixels1L = _mm_cvtepi16_epi32(pixels1L);
				pixels2H = _mm_srli_si128(pixels2L, 8);
				pixels2H = _mm_cvtepi16_epi32(pixels2H);
				pixels2L = _mm_cvtepi16_epi32(pixels2L);

				pixels1H = _mm_sub_epi32(pixels1H, pixels2H);
				pixels1L = _mm_sub_epi32(pixels1L, pixels2L);
				pixels1H = _mm_abs_epi32(pixels1H);
				pixels1L = _mm_abs_epi32(pixels1L);

				pixels1L = _mm_packs_epi32(pixels1L, pixels1H);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1L);
			}

			pLocalSrc1 = (vx_int16 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_int16 *)pLocalSrc2_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int32 temp = (vx_int32) abs((vx_int32) (*pLocalSrc1++) - (vx_int32) (*pLocalSrc2++));
				*pLocalDst++ =  (vx_int16) max(min(temp, INT16_MAX), INT16_MIN);	
			}

			pSrcImage1 += (srcImage1StrideInBytes >> 1);
			pSrcImage2 += (srcImage2StrideInBytes >> 1);
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_AccumulateSquared_S16_S16U8_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint32     shift
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc;
	vx_int16 *pLocalDst;
	__m128i zeromask = _mm_setzero_si128();
	__m128i resultHH, resultHL, resultLH, resultLL, pixelsHH, pixelsHL, pixelsLH, pixelsLL;

	int height = (int)dstHeight;
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		while (height)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			int width = alignedWidth >> 4;						// 16 pixels at a time
			while (width)
			{
				pixelsLL = _mm_load_si128(pLocalSrc_xmm++);
				resultLL = _mm_load_si128(pLocalDst_xmm);
				resultHL = _mm_load_si128(pLocalDst_xmm + 1);

				// Convert input to 32 bit
				pixelsHL = _mm_unpackhi_epi8(pixelsLL, zeromask);
				pixelsHH = _mm_unpackhi_epi16(pixelsHL, zeromask);
				pixelsHL = _mm_cvtepu16_epi32(pixelsHL);
				pixelsLL = _mm_unpacklo_epi8(pixelsLL, zeromask);
				pixelsLH = _mm_unpackhi_epi16(pixelsLL, zeromask);
				pixelsLL = _mm_cvtepu16_epi32(pixelsLL);

				// Convert result to 32 bit
				resultHH = _mm_srli_si128(resultHL, 8);
				resultHH = _mm_cvtepi16_epi32(resultHH);
				resultHL = _mm_cvtepi16_epi32(resultHL);
				resultLH = _mm_srli_si128(resultLL, 8);
				resultLH = _mm_cvtepi16_epi32(resultLH);
				resultLL = _mm_cvtepi16_epi32(resultLL);

				// Multiply
				pixelsHH = _mm_mullo_epi32(pixelsHH, pixelsHH);
				pixelsHL = _mm_mullo_epi32(pixelsHL, pixelsHL);
				pixelsLH = _mm_mullo_epi32(pixelsLH, pixelsLH);
				pixelsLL = _mm_mullo_epi32(pixelsLL, pixelsLL);

				pixelsHH = _mm_srai_epi32(pixelsHH, shift);
				pixelsHL = _mm_srai_epi32(pixelsHL, shift);
				pixelsLH = _mm_srai_epi32(pixelsLH, shift);
				pixelsLL = _mm_srai_epi32(pixelsLL, shift);

				resultHH = _mm_add_epi32(resultHH, pixelsHH);
				resultHL = _mm_add_epi32(resultHL, pixelsHL);
				resultLH = _mm_add_epi32(resultLH, pixelsLH);
				resultLL = _mm_add_epi32(resultLL, pixelsLL);

				resultHL = _mm_packs_epi32(resultHL, resultHH);
				resultLL = _mm_packs_epi32(resultLL, resultLH);

				_mm_store_si128(pLocalDst_xmm++, resultLL);
				_mm_store_si128(pLocalDst_xmm++, resultHL);

				width--;
			}

			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++, pLocalSrc++)
			{
				vx_int32 temp = ((vx_int32)*pLocalSrc * (vx_int32)*pLocalSrc) >> shift;
				temp += (vx_int32)*pLocalDst;
				temp = max(min(temp, (vx_int32)INT16_MAX), (vx_int32)INT16_MIN);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
			height--;
		}
	}
	else
	{
		while (height)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			int width = alignedWidth >> 4;						// 16 pixels at a time
			while (width)
			{
				pixelsLL = _mm_loadu_si128(pLocalSrc_xmm++);
				resultLL = _mm_loadu_si128(pLocalDst_xmm);
				resultHL = _mm_loadu_si128(pLocalDst_xmm + 1);

				// Convert input to 32 bit
				pixelsHL = _mm_unpackhi_epi8(pixelsLL, zeromask);
				pixelsHH = _mm_unpackhi_epi16(pixelsHL, zeromask);
				pixelsHL = _mm_cvtepu16_epi32(pixelsHL);
				pixelsLL = _mm_unpacklo_epi8(pixelsLL, zeromask);
				pixelsLH = _mm_unpackhi_epi16(pixelsLL, zeromask);
				pixelsLL = _mm_cvtepu16_epi32(pixelsLL);

				// Convert result to 32 bit
				resultHH = _mm_srli_si128(resultHL, 8);
				resultHH = _mm_cvtepi16_epi32(resultHH);
				resultHL = _mm_cvtepi16_epi32(resultHL);
				resultLH = _mm_srli_si128(resultLL, 8);
				resultLH = _mm_cvtepi16_epi32(resultLH);
				resultLL = _mm_cvtepi16_epi32(resultLL);

				// Multiply
				pixelsHH = _mm_mullo_epi32(pixelsHH, pixelsHH);
				pixelsHL = _mm_mullo_epi32(pixelsHL, pixelsHL);
				pixelsLH = _mm_mullo_epi32(pixelsLH, pixelsLH);
				pixelsLL = _mm_mullo_epi32(pixelsLL, pixelsLL);

				pixelsHH = _mm_srai_epi32(pixelsHH, shift);
				pixelsHL = _mm_srai_epi32(pixelsHL, shift);
				pixelsLH = _mm_srai_epi32(pixelsLH, shift);
				pixelsLL = _mm_srai_epi32(pixelsLL, shift);

				resultHH = _mm_add_epi32(resultHH, pixelsHH);
				resultHL = _mm_add_epi32(resultHL, pixelsHL);
				resultLH = _mm_add_epi32(resultLH, pixelsLH);
				resultLL = _mm_add_epi32(resultLL, pixelsLL);

				resultHL = _mm_packs_epi32(resultHL, resultHH);
				resultLL = _mm_packs_epi32(resultLL, resultLH);

				_mm_storeu_si128(pLocalDst_xmm++, resultLL);
				_mm_storeu_si128(pLocalDst_xmm++, resultHL);

				width--;
			}

			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++, pLocalSrc++)
			{
				vx_int32 temp = ((vx_int32)*pLocalSrc * (vx_int32)*pLocalSrc) >> shift;
				temp += (vx_int32)*pLocalDst;
				temp = max(min(temp, (vx_int32)INT16_MAX), (vx_int32)INT16_MIN);
				*pLocalDst++ = (vx_int16)temp;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
			height--;
		}
	}
	return AGO_SUCCESS;
}


int HafCpu_Accumulate_S16_S16U8_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc;
	vx_int16 *pLocalDst;

	__m128i resultL, resultH, pixelsL, pixelsH;
	__m128i zeromask = _mm_setzero_si128();

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				resultL = _mm_load_si128(pLocalDst_xmm);
				resultH = _mm_load_si128(pLocalDst_xmm + 1);
				pixelsL = _mm_load_si128(pLocalSrc_xmm++);
				pixelsH = _mm_unpackhi_epi8(pixelsL, zeromask);
				pixelsL = _mm_cvtepu8_epi16(pixelsL);
				resultL = _mm_adds_epi16(resultL, pixelsL);
				resultH = _mm_adds_epi16(resultH, pixelsH);
				_mm_store_si128(pLocalDst_xmm++, resultL);
				_mm_store_si128(pLocalDst_xmm++, resultH);
			}

			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++, pLocalSrc++)
			{
				vx_int32 temp = (vx_int32)*pLocalDst + (vx_int32)*pLocalSrc;
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				resultL = _mm_loadu_si128(pLocalDst_xmm);
				resultH = _mm_loadu_si128(pLocalDst_xmm + 1);
				pixelsL = _mm_loadu_si128(pLocalSrc_xmm++);
				pixelsH = _mm_unpackhi_epi8(pixelsL, zeromask);
				pixelsL = _mm_cvtepu8_epi16(pixelsL);
				resultL = _mm_adds_epi16(resultL, pixelsL);
				resultH = _mm_adds_epi16(resultH, pixelsH);
				_mm_storeu_si128(pLocalDst_xmm++, resultL);
				_mm_storeu_si128(pLocalDst_xmm++, resultH);
			}

			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_int16 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++, pLocalSrc++)
			{
				vx_int32 temp = (vx_int32)*pLocalDst + (vx_int32)*pLocalSrc;
				*pLocalDst++ = (vx_int16)max(min(temp, INT16_MAX), INT16_MIN);
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += (dstImageStrideInBytes >> 1);
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorDepth_U8_S16_Wrap
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int32      shift
	)
{
	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i maskL = _mm_set_epi8((char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0x0E, (char)0x0C, (char)0x0A, (char)0x08, (char)0x06, (char)0x04, (char)0x02, (char)0x0);
	__m128i maskH = _mm_set_epi8((char)0x0E, (char)0x0C, (char)0x0A, (char)0x08, (char)0x06, (char)0x04, (char)0x02, (char)0x0, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF);
	__m128i pixels1, pixels2;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_int16 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < prefixWidth; width++)
		{
			int pix = (int) (*pLocalSrc++);
			*pLocalDst++ = (vx_uint8)((pix >> shift) & 0xFF);
		}

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels1 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 8));
			pixels1 = _mm_srai_epi16(pixels1, (int) shift);
			pixels2 = _mm_srai_epi16(pixels2, (int) shift);
			pixels1 = _mm_shuffle_epi8(pixels1, maskL);
			pixels2 = _mm_shuffle_epi8(pixels2, maskH);
			pixels1 = _mm_or_si128(pixels1, pixels2);
			_mm_store_si128((__m128i *)pLocalDst, pixels1);

			pLocalSrc += 16;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			int pix = *pLocalSrc++;
			*pLocalDst++ = (vx_uint8)((pix >> shift) & 0xFF);
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorDepth_U8_S16_Sat
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int32      shift
	)
{
	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i pixels1, pixels2;

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_int16 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < prefixWidth; width++)
		{
			int pix = (int) (*pLocalSrc++);
			pix >>= shift;
			pix = min(max(pix, 0), 255);
			*pLocalDst++ = (vx_uint8)(pix);
		}

		for (int width = 0; width < (int)alignedWidth; width += 16)
		{
			pixels1 = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixels2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 8));
			pixels1 = _mm_srai_epi16(pixels1, (int)shift);
			pixels2 = _mm_srai_epi16(pixels2, (int)shift);
			pixels1 = _mm_packus_epi16(pixels1, pixels2);
			_mm_store_si128((__m128i *)pLocalDst, pixels1);

			pLocalSrc += 16;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			int pix = *pLocalSrc++;
			pix >>= shift;
			pix = min(max(pix, 0), 255);
			*pLocalDst++ = (vx_uint8)(pix);
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ColorDepth_S16_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int32      shift
	)
{
	int prefixWidth = intptr_t(pDstImage) & 7;			// Two bytes in output = 1 pixel
	prefixWidth = (prefixWidth == 0) ? 0 : (8 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i zeromask = _mm_setzero_si128();
	__m128i pixelsL, pixelsH;

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_int16 * pLocalDst = pDstImage;

		for (int width = 0; width < prefixWidth; width++)
		{
			int pix = (int) (*pLocalSrc++);
			*pLocalDst++ = (vx_int16) (pix << shift);
		}

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixelsL = _mm_loadu_si128((__m128i *)pLocalSrc);
			pixelsH = _mm_unpackhi_epi8(pixelsL, zeromask);
			pixelsL = _mm_cvtepu8_epi16(pixelsL);
			pixelsL = _mm_slli_epi16(pixelsL, (int)shift);
			pixelsH = _mm_slli_epi16(pixelsH, (int)shift);
			_mm_store_si128((__m128i *)pLocalDst, pixelsL);
			_mm_store_si128((__m128i *)(pLocalDst + 8), pixelsH);

			pLocalSrc += 16;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			int pix = (int)(*pLocalSrc++);
			*pLocalDst++ = (vx_int16)(pix << shift);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += (dstImageStrideInBytes >> 1);
	}
	return AGO_SUCCESS;
}

int HafCpu_Threshold_U8_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;
	__m128i pixels;
	__m128i offset = _mm_set1_epi8((char) 0x80);				// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i thresh = _mm_set1_epi8((char) threshold);
	thresh = _mm_xor_si128(thresh, offset);						// Convert the threshold to the new range

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int) dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_load_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				pixels = _mm_cmpgt_epi8(pixels, thresh);
				_mm_store_si128(pLocalDst_xmm++, pixels);
			}

			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = (pix > threshold) ? (vx_uint8)255 : 0;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int) dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				pixels = _mm_cmpgt_epi8(pixels, thresh);
				_mm_storeu_si128(pLocalDst_xmm++, pixels);
			}

			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = (pix > threshold) ? (vx_uint8)255 : 0;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}

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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;
	__m128i pixels;
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i threshU = _mm_set1_epi8((char)upper);
	__m128i threshL = _mm_set1_epi8((char)lower);
	__m128i ones = _mm_set1_epi8((char)0xFF);
	__m128i temp;
	
	threshU = _mm_xor_si128(threshU, offset);					// Convert the upper threshold to the new range
	threshL = _mm_xor_si128(threshL, offset);					// Convert the lower threshold to the new range

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_load_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				temp = _mm_cmpgt_epi8(pixels, threshU);				// pixels > upper gives 255
				pixels = _mm_cmplt_epi8(pixels, threshL);			// pixels < lower gives 255
				pixels = _mm_or_si128(pixels, temp);
				pixels = _mm_andnot_si128(pixels, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = ((pix > upper) || (pix < lower)) ? 0 : (vx_uint8)255;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				temp = _mm_cmpgt_epi8(pixels, threshU);				// pixels > upper gives 255
				pixels = _mm_cmplt_epi8(pixels, threshL);			// pixels < lower gives 255
				pixels = _mm_or_si128(pixels, temp);
				pixels = _mm_andnot_si128(pixels, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = ((pix > upper) || (pix < lower)) ? 0 : (vx_uint8)255;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

#if USE_BMI2
/* The function assumes that the source image pointer is 16 byte aligned, and the source stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
int HafCpu_Threshold_U1_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	)
{
	__m128i * src = (__m128i*)pSrcImage;

	__m128i pixels;
	__m128i offset = _mm_set1_epi8((char)0x80);				// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i thresh = _mm_set1_epi8((char)threshold);
	thresh = _mm_xor_si128(thresh, offset);						// Convert the threshold to the new range

	uint64_t maskConv = 0x0101010101010101;
	uint64_t result[2];

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels = _mm_load_si128(&src[width >> 4]);
			pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
			pixels = _mm_cmpgt_epi8(pixels, thresh);

			// Convert U8 to U1
#ifdef _WIN64
			result[0] = _pext_u64(pixels.m128i_u64[0], maskConv);
			result[1] = _pext_u64(pixels.m128i_u64[1], maskConv);
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

/* The function assumes that the source image pointer is 16 byte aligned, and the source stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src = (__m128i*)pSrcImage;
	__m128i pixels;
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i threshU = _mm_set1_epi8((char)upper);
	__m128i threshL = _mm_set1_epi8((char)lower);
	__m128i ones = _mm_set1_epi8((char)1);
	__m128i temp;

	threshU = _mm_xor_si128(threshU, offset);					// Convert the upper threshold to the new range
	threshL = _mm_xor_si128(threshL, offset);					// Convert the lower threshold to the new range

	uint64_t maskConv = 0x0101010101010101;
	uint64_t result[2];

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels = _mm_load_si128(&src[width >> 4]);
			pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
			temp = _mm_cmpgt_epi8(pixels, threshU);
			temp = _mm_andnot_si128(temp, ones);				// This gives 255 if pixels <= threshU, a way to implement less than or equal to
			pixels = _mm_cmplt_epi8(pixels, threshL);
			pixels = _mm_andnot_si128(pixels, temp);			// 255 if pixels >= threshL and AND with temp
			
			// Convert U8 to U1
#ifdef _WIN64
			result[0] = _pext_u64(pixels.m128i_u64[0], maskConv);
			result[1] = _pext_u64(pixels.m128i_u64[1], maskConv);
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
#else

int HafCpu_Threshold_U1_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	)
{
	__m128i * pLocalSrc_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;

	__m128i pixels;
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i thresh = _mm_set1_epi8((char)threshold);
	thresh = _mm_xor_si128(thresh, offset);						// Convert the threshold to the new range

	int pixelmask;
	int height = (int)dstHeight;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	while (height)
	{
		pLocalSrc_xmm = (__m128i*) pSrcImage;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

		int width = (int)(alignedWidth >> 4);					// 16 pixels (bits) are processed at a time in the inner loop
		while (width)
		{
			pixels = _mm_loadu_si128(pLocalSrc_xmm++);
			pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
			pixels = _mm_cmpgt_epi8(pixels, thresh);

			pixelmask = _mm_movemask_epi8(pixels);				// Convert U8 to U1
			*pLocalDst_16++ = (vx_int16)(pixelmask & 0xFFFF);
			width--;
		}
		pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
		pLocalDst = (vx_uint8 *)pLocalDst_16;

		width = 0;
		while (width < postfixWidth)
		{
			pixelmask = 0;
			for (int i = 0; i < 8; i++, width++)
			{
				if (width == postfixWidth)
					break;

				pixelmask >>= 1;
				if (*pLocalSrc++ > threshold)
					pixelmask |= 0x80;
			}
			*pLocalDst++ = (vx_uint8)(pixelmask & 0xFF);
		}
		
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i * pLocalSrc_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;

	__m128i pixels, temp;
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i threshU = _mm_set1_epi8((char)upper);
	__m128i threshL = _mm_set1_epi8((char)lower);
	__m128i ones = _mm_set1_epi8((char)0xFF);

	threshU = _mm_xor_si128(threshU, offset);					// Convert the upper threshold to the new range
	threshL = _mm_xor_si128(threshL, offset);					// Convert the lower threshold to the new range

	int pixelmask;
	int height = (int)dstHeight;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	while (height)
	{
		pLocalSrc_xmm = (__m128i*) pSrcImage;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;
		int width = (int)(dstWidth >> 4);						// 16 pixels (bits) are processed at a time in the inner loop

		while (width)
		{
			pixels = _mm_loadu_si128(pLocalSrc_xmm++);
			pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
			temp = _mm_cmpgt_epi8(pixels, threshU);				// pixels > upper gives 255
			pixels = _mm_cmplt_epi8(pixels, threshL);			// pixels < lower gives 255
			pixels = _mm_or_si128(pixels, temp);
			pixels = _mm_andnot_si128(pixels, ones);

			pixelmask = _mm_movemask_epi8(pixels);				// Convert U8 to U1
			*pLocalDst_16++ = (short)(pixelmask & 0xFFFF);
			width--;
		}
		pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
		pLocalDst = (vx_uint8 *)pLocalDst_16;

		width = 0;
		while (width < postfixWidth)
		{
			pixelmask = 0;
			vx_uint8 pix = *pLocalSrc++;
			for (int i = 0; i < 8; i++, width++)
			{
				if (width == postfixWidth)
					break;

				pixelmask >>= 1;
				if ((pix >= lower) && (pix <= upper))
					pixelmask |= 0x80;
			}
			*pLocalDst++ = (vx_uint8)(pixelmask & 0xFF);
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}
#endif

int HafCpu_Threshold_U8_S16_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16      threshold
	)
{
	// C code 
	for (int height = 0; height < (int) dstHeight; height++)
	{
		for (int width = 0; width < (int) dstWidth; width++)
		{
			vx_int16 pix = *pSrcImage++;
			*pDstImage++ = (pix > threshold) ? (vx_uint8)255 : 0;
		}
	}
	
	/*Fix SSE Code
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_int16 *pLocalSrc;
	vx_uint8 *pLocalDst;
	__m128i pixels, pixels_16;
	__m128i thresh = _mm_set1_epi16((short) threshold);

	int alignedWidth = dstWidth & ~7;
	int postfixWidth = dstWidth - alignedWidth;
	if (useAligned)
	{
		for (int height = 0; height < (int) dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 8)
			{
				pixels = _mm_load_si128(pLocalSrc_xmm++);
				pixels = _mm_cmpgt_epi16(pixels, thresh);
				pixels_16 = _mm_packus_epi16 (pixels, _mm_setzero_si128());
				_mm_store_si128(pLocalDst_xmm++, pixels_16);
			}

			pLocalSrc = (vx_int16 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;
			
			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 pix = *pLocalSrc++;
				*pLocalDst++ = (pix > threshold) ? (vx_uint8)255 : 0;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int) dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 8)
			{
				pixels = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels = _mm_cmpgt_epi16(pixels, thresh);
				_mm_storeu_si128(pLocalDst_xmm++, pixels);
			}

			pLocalSrc = (vx_int16 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 pix = *pLocalSrc++;
				*pLocalDst++ = (pix > threshold) ? (vx_uint8)255 : 0;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}*/
	return AGO_SUCCESS;
}

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
	)
{
	for (int height = 0; height < (int)dstHeight; height++)
	{
		for (int width = 0; width < (int)dstWidth; width++)
		{
			vx_int16 pix = *pSrcImage++;
			*pDstImage++ = ((pix > upper) || (pix < lower)) ? 0 : (vx_uint8)255;
		}
	}

	/*Fix SSE Code
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_int16 *pLocalSrc;
	vx_uint8 *pLocalDst;
	__m128i pixels;
	__m128i threshU = _mm_set1_epi16((short)upper);
	__m128i threshL = _mm_set1_epi16((short)lower);
	__m128i zeros = _mm_set1_epi16((short)0x0000);
	__m128i temp, pixels_16;
	
	int alignedWidth = dstWidth & ~7;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 8)
			{
				pixels = _mm_load_si128(pLocalSrc_xmm++);
				temp = _mm_cmpgt_epi16(pixels, threshU);				// pixels > upper gives 0
				pixels = _mm_cmplt_epi16(pixels, threshL);				// pixels < lower gives 0
				pixels = _mm_or_si128(pixels, temp);
				pixels = _mm_andnot_si128(pixels, zeros);
				pixels_16 = _mm_packus_epi16(pixels, _mm_setzero_si128());
				_mm_store_si128(pLocalDst_xmm++, pixels_16);
			}
			pLocalSrc = (vx_int16 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 pix = *pLocalSrc++;
				*pLocalDst++ = ((pix > upper) || (pix < lower)) ? 0 : (vx_uint8)255;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 8)
			{
				pixels = _mm_loadu_si128(pLocalSrc_xmm++);
				temp = _mm_cmpgt_epi16(pixels, threshU);			// pixels > upper gives 0
				pixels = _mm_cmplt_epi16(pixels, threshL);			// pixels < lower gives 0
				pixels = _mm_or_si128(pixels, temp);
				pixels = _mm_andnot_si128(pixels, zeros);
				pixels_16 = _mm_packus_epi16(pixels, _mm_setzero_si128());
				_mm_storeu_si128(pLocalDst_xmm++, pixels_16);
			}
			pLocalSrc = (vx_int16 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_int16 pix = *pLocalSrc++;
				*pLocalDst++ = ((pix > upper) || (pix < lower)) ? 0 : (vx_uint8)255;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}*/
	return AGO_SUCCESS;
}

int HafCpu_ThresholdNot_U8_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;
	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i thresh = _mm_set1_epi8((char)threshold);
	thresh = _mm_xor_si128(thresh, offset);						// Convert the threshold to the new range
	
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_load_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				pixels = _mm_cmpgt_epi8(pixels, thresh);
				pixels = _mm_andnot_si128(pixels, ones);			// NOT
				_mm_store_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = (pix > threshold) ? 0 : (vx_uint8)255;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				pixels = _mm_cmpgt_epi8(pixels, thresh);
				pixels = _mm_andnot_si128(pixels, ones);			// NOT
				_mm_storeu_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = (pix > threshold) ? 0 : (vx_uint8)255;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	
	return AGO_SUCCESS;
}

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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;
	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i threshU = _mm_set1_epi8((char)upper);
	__m128i threshL = _mm_set1_epi8((char)lower);
	__m128i temp;

	threshU = _mm_xor_si128(threshU, offset);					// Convert the upper threshold to the new range
	threshL = _mm_xor_si128(threshL, offset);					// Convert the lower threshold to the new range

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_load_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				temp = _mm_cmpgt_epi8(pixels, threshU);				// pixels > upper gives 255
				pixels = _mm_cmplt_epi8(pixels, threshL);			// pixels < lower gives 255
				pixels = _mm_or_si128(pixels, temp);
				_mm_store_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = ((pix > upper) && (pix < lower)) ? (vx_uint8)255 : 0;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				temp = _mm_cmpgt_epi8(pixels, threshU);				// pixels > upper gives 255
				pixels = _mm_cmplt_epi8(pixels, threshL);			// pixels < lower gives 255
				pixels = _mm_or_si128(pixels, temp);
				_mm_storeu_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = ((pix > upper) && (pix < lower)) ? (vx_uint8)255 : 0;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}

#if USE_BMI2
/* The function assumes that the source image pointer is 16 byte aligned, and the source stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
int HafCpu_ThresholdNot_U1_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	)
{
	__m128i * src = (__m128i*)pSrcImage;

	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	__m128i offset = _mm_set1_epi8((char)0x80);				// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i thresh = _mm_set1_epi8((char)threshold);
	thresh = _mm_xor_si128(thresh, offset);						// Convert the threshold to the new range

	uint64_t maskConv = 0x0101010101010101;
	uint64_t result[2];

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels = _mm_load_si128(&src[width >> 4]);
			pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
			pixels = _mm_cmpgt_epi8(pixels, thresh);
			pixels = _mm_andnot_si128(pixels, ones);			// NOT

			// Convert U8 to U1
#ifdef _WIN64
			result[0] = _pext_u64(pixels.m128i_u64[0], maskConv);
			result[1] = _pext_u64(pixels.m128i_u64[1], maskConv);
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

/* The function assumes that the source image pointer is 16 byte aligned, and the source stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src = (__m128i*)pSrcImage;
	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i threshU = _mm_set1_epi8((char)upper);
	__m128i threshL = _mm_set1_epi8((char)lower);
	__m128i temp;

	threshU = _mm_xor_si128(threshU, offset);					// Convert the upper threshold to the new range
	threshL = _mm_xor_si128(threshL, offset);					// Convert the lower threshold to the new range

	uint64_t maskConv = 0x0101010101010101;
	uint64_t result[2];

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels = _mm_load_si128(&src[width >> 4]);
			pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
			temp = _mm_cmpgt_epi8(pixels, threshU);
			temp = _mm_andnot_si128(temp, ones);				// This gives 255 if pixels <= threshU, a way to implement less than or equal to
			pixels = _mm_cmplt_epi8(pixels, threshL);
			pixels = _mm_andnot_si128(pixels, temp);			// 255 if pixels >= threshL and AND with temp
			pixels = _mm_andnot_si128(pixels, ones);			// NOT

			// Convert U8 to U1
#ifdef _WIN64
			result[0] = _pext_u64(pixels.m128i_u64[0], maskConv);
			result[1] = _pext_u64(pixels.m128i_u64[1], maskConv);
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
#else

int HafCpu_ThresholdNot_U1_U8_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	)
{
	__m128i * pLocalSrc_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;

	__m128i pixels;
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i thresh = _mm_set1_epi8((char)threshold);
	thresh = _mm_xor_si128(thresh, offset);						// Convert the threshold to the new range

	int pixelmask;
	int height = (int)dstHeight;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	while (height)
	{
		pLocalSrc_xmm = (__m128i*) pSrcImage;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

		int width = (int)(dstWidth >> 4);						// 16 pixels (bits) are processed at a time in the inner loop
		while (width)
		{
			pixels = _mm_load_si128(pLocalSrc_xmm++);
			pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
			pixels = _mm_cmpgt_epi8(pixels, thresh);

			pixelmask = _mm_movemask_epi8(pixels);				// Convert U8 to U1
			*pLocalDst_16++ = (vx_int16)(~pixelmask & 0xFFFF);
			width--;
		}
		pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
		pLocalDst = (vx_uint8 *)pLocalDst_16;

		width = 0;
		while (width < postfixWidth)
		{
			pixelmask = 0;
			for (int i = 0; i < 8; i++, width++)
			{
				if (*pLocalSrc++ <= threshold)
					pixelmask |= 1;
				pixelmask <<= 1;
			}
			*pLocalDst++ = (vx_uint8)(pixelmask & 0xFF);
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i * pLocalSrc_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;

	__m128i pixels, temp;
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i threshU = _mm_set1_epi8((char)upper);
	__m128i threshL = _mm_set1_epi8((char)lower);
	
	threshU = _mm_xor_si128(threshU, offset);					// Convert the upper threshold to the new range
	threshL = _mm_xor_si128(threshL, offset);					// Convert the lower threshold to the new range

	int pixelmask;
	int height = (int)dstHeight;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	while (height)
	{
		pLocalSrc_xmm = (__m128i*) pSrcImage;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;
		int width = (int)(dstWidth >> 4);						// 16 pixels (bits) are processed at a time in the inner loop

		while (width)
		{
			pixels = _mm_load_si128(pLocalSrc_xmm++);
			pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
			temp = _mm_cmpgt_epi8(pixels, threshU);				// pixels > upper gives 255
			pixels = _mm_cmplt_epi8(pixels, threshL);			// pixels < lower gives 255
			pixels = _mm_or_si128(pixels, temp);

			pixelmask = _mm_movemask_epi8(pixels);				// Convert U8 to U1
			*pLocalDst_16++ = (vx_int16)(pixelmask & 0xFFFF);
			width--;
		}
		pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
		pLocalDst = (vx_uint8 *)pLocalDst_16;

		width = 0;
		while (width < postfixWidth)
		{
			pixelmask = 0;
			vx_uint8 pix = *pLocalSrc++;
			for (int i = 0; i < 8; i++, width++)
			{
				if ((pix < lower) && (pix > upper))
					pixelmask |= 1;
				pixelmask <<= 1;
			}
			*pLocalDst++ = (vx_uint8)(pixelmask & 0xFF);
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}
#endif

int HafCpu_ThresholdNot_U8_S16_Binary
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8      threshold
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;
	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i thresh = _mm_set1_epi8((char)threshold);
	thresh = _mm_xor_si128(thresh, offset);						// Convert the threshold to the new range
	
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_load_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				pixels = _mm_cmpgt_epi8(pixels, thresh);
				pixels = _mm_andnot_si128(pixels, ones);			// NOT
				_mm_store_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = (pix > threshold) ? 0 : (vx_uint8)255;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				pixels = _mm_cmpgt_epi8(pixels, thresh);
				pixels = _mm_andnot_si128(pixels, ones);			// NOT
				_mm_storeu_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = (pix > threshold) ? 0 : (vx_uint8)255;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	
	return AGO_SUCCESS;
}

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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;
	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	__m128i offset = _mm_set1_epi8((char)0x80);					// To convert the range from 0..255 to -128..127, because SSE does not have compare instructions for unsigned bytes
	__m128i threshU = _mm_set1_epi8((char)upper);
	__m128i threshL = _mm_set1_epi8((char)lower);
	__m128i temp;

	threshU = _mm_xor_si128(threshU, offset);					// Convert the upper threshold to the new range
	threshL = _mm_xor_si128(threshL, offset);					// Convert the lower threshold to the new range

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_load_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				temp = _mm_cmpgt_epi8(pixels, threshU);				// pixels > upper gives 255
				pixels = _mm_cmplt_epi8(pixels, threshL);			// pixels < lower gives 255
				pixels = _mm_or_si128(pixels, temp);
				_mm_store_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = ((pix > upper) && (pix < lower)) ? (vx_uint8)255 : 0;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels = _mm_xor_si128(pixels, offset);				// Convert the pixels to the new range
				temp = _mm_cmpgt_epi8(pixels, threshU);				// pixels > upper gives 255
				pixels = _mm_cmplt_epi8(pixels, threshL);			// pixels < lower gives 255
				pixels = _mm_or_si128(pixels, temp);
				_mm_storeu_si128(pLocalDst_xmm++, pixels);
			}
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				vx_uint8 pix = *pLocalSrc++;
				*pLocalDst++ = ((pix > upper) && (pix < lower)) ? (vx_uint8)255 : 0;
			}
			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}


// compute the dstImage values from the LUT of srcImage
int HafCpu_Lut_U8_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_uint8    * pLut
	)
{
	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;				// Check for multiple of 16
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i pixels1, pixels2;
	int p0, p1, p2, p3;
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		unsigned char * pLocalDst = (unsigned char*)pDstImage;
		unsigned char * pLocalSrc = (unsigned char*)pSrcImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++, pLocalDst++)
		{
			*pLocalDst = pLut[*pLocalSrc];
		}

		for (int x = 0; x < (alignedWidth >> 4); x++)
		{
			pixels1 = _mm_loadu_si128((__m128i *) pLocalSrc);
			p0 = _mm_cvtsi128_si32(pixels1);
			p1 = _mm_extract_epi32(pixels1, 1);
			p2 = _mm_extract_epi32(pixels1, 2);
			p3 = _mm_extract_epi32(pixels1, 3);
			p0 = pLut[p0 & 0xff] | (pLut[(p0 >> 8) & 0xFF] << 8) | (pLut[(p0 >> 16) & 0xFF] << 16) | (pLut[(p0 >> 24) & 0xFF] << 24);
			p1 = pLut[p1 & 0xff] | (pLut[(p1 >> 8) & 0xFF] << 8) | (pLut[(p1 >> 16) & 0xFF] << 16) | (pLut[(p1 >> 24) & 0xFF] << 24);
			p2 = pLut[p2 & 0xff] | (pLut[(p2 >> 8) & 0xFF] << 8) | (pLut[(p2 >> 16) & 0xFF] << 16) | (pLut[(p2 >> 24) & 0xFF] << 24);
			p3 = pLut[p3 & 0xff] | (pLut[(p3 >> 8) & 0xFF] << 8) | (pLut[(p3 >> 16) & 0xFF] << 16) | (pLut[(p3 >> 24) & 0xFF] << 24);
			M128I(pixels2).m128i_u32[0] = p0;
			M128I(pixels2).m128i_u32[1] = p1;
			M128I(pixels2).m128i_u32[2] = p2;
			M128I(pixels2).m128i_u32[3] = p3;
			_mm_store_si128((__m128i *) pLocalDst, pixels2);

			pLocalSrc += 16;
			pLocalDst += 16;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++, pLocalDst++)
		{
			*pLocalDst = pLut[*pLocalSrc];
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

// compute the dstImage values from the LUT of srcImage
int HafCpu_Lut_S16_S16
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_int16    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes,
		vx_int16    * pLut,
		vx_uint32 	  offset
	)
{
	for (int height = 0; height < (int) dstHeight; height++)
	{
		for (int width = 0; width < (int) dstWidth; width++, pSrcImage++, pDstImage++)
		{
			*pDstImage = pLut[*pSrcImage+offset];
		}
	}
	//TBD: Implement SSE Version
	/*int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 15;				// Check for multiple of 16
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i pixels1, pixels2;
	int p0, p1, p2, p3;
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		unsigned char * pLocalDst = (unsigned char*)pDstImage;
		unsigned char * pLocalSrc = (unsigned char*)pSrcImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++, pLocalDst++)
		{
			*pLocalDst = pLut[*pLocalSrc];
		}

		for (int x = 0; x < (alignedWidth >> 4); x++)
		{
			pixels1 = _mm_loadu_si128((__m128i *) pLocalSrc);
			p0 = _mm_cvtsi128_si32(pixels1);
			p1 = _mm_extract_epi32(pixels1, 1);
			p2 = _mm_extract_epi32(pixels1, 2);
			p3 = _mm_extract_epi32(pixels1, 3);
			p0 = pLut[p0 & 0xff] | (pLut[(p0 >> 8) & 0xFF] << 8) | (pLut[(p0 >> 16) & 0xFF] << 16) | (pLut[(p0 >> 24) & 0xFF] << 24);
			p1 = pLut[p1 & 0xff] | (pLut[(p1 >> 8) & 0xFF] << 8) | (pLut[(p1 >> 16) & 0xFF] << 16) | (pLut[(p1 >> 24) & 0xFF] << 24);
			p2 = pLut[p2 & 0xff] | (pLut[(p2 >> 8) & 0xFF] << 8) | (pLut[(p2 >> 16) & 0xFF] << 16) | (pLut[(p2 >> 24) & 0xFF] << 24);
			p3 = pLut[p3 & 0xff] | (pLut[(p3 >> 8) & 0xFF] << 8) | (pLut[(p3 >> 16) & 0xFF] << 16) | (pLut[(p3 >> 24) & 0xFF] << 24);
			M128I(pixels2).m128i_u32[0] = p0;
			M128I(pixels2).m128i_u32[1] = p1;
			M128I(pixels2).m128i_u32[2] = p2;
			M128I(pixels2).m128i_u32[3] = p3;
			_mm_store_si128((__m128i *) pLocalDst, pixels2);

			pLocalSrc += 16;
			pLocalDst += 16;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++, pLocalDst++)
		{
			*pLocalDst = pLut[*pLocalSrc];
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	*/
	return AGO_SUCCESS;
}

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
	)
{
	short *pLocalGx, *pLocalGy, *pLocalDst;
	
	int prefixWidth = intptr_t(pMagImage) & 15;							// check for 16 byte aligned
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;
	int postfixWidth = ((int)dstWidth - prefixWidth) & 7;				// Check for multiple of 8
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	__m128i pixelsGxH, pixelsGxL, pixelsGyH, pixelsGyL;
	__m128d pixels0, pixels1, pixels2, pixels3, temp;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		pLocalGx = (short *)pGxImage;
		pLocalGy = (short *)pGyImage;
		pLocalDst = (short *)pMagImage;

		for (int x = 0; x < prefixWidth; x++, pLocalGx++, pLocalGy++)
		{
			float temp = (float)(*pLocalGx * *pLocalGx) + (float)(*pLocalGy * *pLocalGy);
			temp = sqrtf(temp);
			*pLocalDst++ = (vx_int16)temp;
		}

		for (int width = 0; width < (alignedWidth >> 3); width++)
		{
			pixelsGxH = _mm_loadu_si128((__m128i *) pLocalGx);
			pixelsGyH = _mm_loadu_si128((__m128i *) pLocalGy);

			pixelsGxL = _mm_cvtepi16_epi32(pixelsGxH);					// Convert lower 4 words to dwords
			pixelsGyL = _mm_cvtepi16_epi32(pixelsGyH);					// Convert lower 4 words to dwords
			pixelsGxH = _mm_srli_si128(pixelsGxH, 8);
			pixelsGyH = _mm_srli_si128(pixelsGyH, 8);
			pixelsGxH = _mm_cvtepi16_epi32(pixelsGxH);					// Convert upper 4 words to dwords
			pixelsGyH = _mm_cvtepi16_epi32(pixelsGyH);					// Convert upper 4 words to dwords

			pixelsGxL = _mm_mullo_epi32(pixelsGxL, pixelsGxL);			// square
			pixelsGxH = _mm_mullo_epi32(pixelsGxH, pixelsGxH);
			pixelsGyL = _mm_mullo_epi32(pixelsGyL, pixelsGyL);
			pixelsGyH = _mm_mullo_epi32(pixelsGyH, pixelsGyH);

			// Convert to double precision values
			pixels0 = _mm_cvtepi32_pd(pixelsGxL);
			temp = _mm_cvtepi32_pd(pixelsGyL);
			pixels0 = _mm_add_pd(pixels0, temp);						// Lower two values a^2 + b^2

			pixelsGxL = _mm_srli_si128(pixelsGxL, 8);
			pixelsGyL = _mm_srli_si128(pixelsGyL, 8);
			pixels1 = _mm_cvtepi32_pd(pixelsGxL);
			temp = _mm_cvtepi32_pd(pixelsGyL);
			pixels1 = _mm_add_pd(pixels1, temp);						// Next two values a^2 + b^2

			pixels2 = _mm_cvtepi32_pd(pixelsGxH);
			temp = _mm_cvtepi32_pd(pixelsGyH);
			pixels2 = _mm_add_pd(pixels2, temp);						// Next two values a^2 + b^2

			pixelsGxH = _mm_srli_si128(pixelsGxH, 8);
			pixelsGyH = _mm_srli_si128(pixelsGyH, 8);
			pixels3 = _mm_cvtepi32_pd(pixelsGxH);
			temp = _mm_cvtepi32_pd(pixelsGyH);
			pixels3 = _mm_add_pd(pixels3, temp);						// Upper two values a^2 + b^2

			pixels0 = _mm_sqrt_pd(pixels0);								// square root
			pixels1 = _mm_sqrt_pd(pixels1);								// square root
			pixels2 = _mm_sqrt_pd(pixels2);								// square root
			pixels3 = _mm_sqrt_pd(pixels3);								// square root

			pixelsGxL = _mm_cvtpd_epi32(pixels0);						// Convert double to lower 2 dwords
			pixelsGyL = _mm_cvtpd_epi32(pixels1);						// Convert double to next 2 dwords
			pixelsGxH = _mm_cvtpd_epi32(pixels2);						// Convert double to next 2 dwords
			pixelsGyH = _mm_cvtpd_epi32(pixels3);						// Convert double to upper 2 dwords

			pixelsGyL = _mm_slli_si128(pixelsGyL, 8);
			pixelsGyH = _mm_slli_si128(pixelsGyH, 8);
			pixelsGxL = _mm_or_si128(pixelsGxL, pixelsGyL);
			pixelsGxH = _mm_or_si128(pixelsGxH, pixelsGyH);

			pixelsGxL = _mm_packs_epi32(pixelsGxL, pixelsGxH);
			_mm_store_si128((__m128i *) pLocalDst, pixelsGxL);

			pLocalGx += 8;
			pLocalGy += 8;
			pLocalDst += 8;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalGx++, pLocalGy++)
		{
			float temp = (float)(*pLocalGx * *pLocalGx) + (float)(*pLocalGy * *pLocalGy);
			temp = sqrtf(temp);
			*pLocalDst++ = (vx_int16)(round(temp));
		}

		pGxImage += (gxImageStrideInBytes >> 1);
		pGyImage += (gyImageStrideInBytes >> 1);
		pMagImage += (magImageStrideInBytes >> 1);
	}
	return AGO_SUCCESS;
}

int HafCpu_AccumulateWeighted_U8_U8U8
(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint8    * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_float32    alpha
)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;

	__m128i pixelsI0, pixelsI1, tempI;
	__m128 a, aprime, pixelsF0, pixelsF1, pixelsF2, pixelsF3, temp;
	a = _mm_set_ps1((float) alpha);
	aprime = _mm_set_ps1((float) (1.0 - alpha));

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				// For the input pixels
				pixelsI0 = _mm_load_si128(pLocalSrc_xmm++);

				pixelsI1 = _mm_cvtepu8_epi32(pixelsI0);				// Convert to int32
				pixelsF0 = _mm_cvtepi32_ps(pixelsI1);				// Convert to float32
				pixelsI1 = _mm_srli_si128(pixelsI0, 4);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				pixelsF1 = _mm_cvtepi32_ps(pixelsI1);				// Convert to float32
				pixelsI1 = _mm_srli_si128(pixelsI0, 8);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				pixelsF2 = _mm_cvtepi32_ps(pixelsI1);				// Convert to float32
				pixelsI1 = _mm_srli_si128(pixelsI0, 12);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				pixelsF3 = _mm_cvtepi32_ps(pixelsI1);				// Convert to float32

				pixelsF0 = _mm_mul_ps(pixelsF0, a);					// alpha * input
				pixelsF1 = _mm_mul_ps(pixelsF1, a);					// alpha * input
				pixelsF2 = _mm_mul_ps(pixelsF2, a);					// alpha * input
				pixelsF3 = _mm_mul_ps(pixelsF3, a);					// alpha * input

				// For the output pixels
				pixelsI0 = _mm_load_si128(pLocalDst_xmm);

				pixelsI1 = _mm_cvtepu8_epi32(pixelsI0);				// Convert to int32
				temp = _mm_cvtepi32_ps(pixelsI1);					// Convert to float32
				temp = _mm_mul_ps(temp, aprime);					// (1 - alpha) * output
				pixelsF0 = _mm_add_ps(pixelsF0, temp);				// (1 - alpha) * output + alpha * input

				pixelsI1 = _mm_srli_si128(pixelsI0, 4);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				temp = _mm_cvtepi32_ps(pixelsI1);					// Convert to float32
				temp = _mm_mul_ps(temp, aprime);					// (1 - alpha) * output
				pixelsF1 = _mm_add_ps(pixelsF1, temp);				// (1 - alpha) * output + alpha * input

				pixelsI1 = _mm_srli_si128(pixelsI0, 8);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				temp = _mm_cvtepi32_ps(pixelsI1);					// Convert to float32
				temp = _mm_mul_ps(temp, aprime);					// (1 - alpha) * output
				pixelsF2 = _mm_add_ps(pixelsF2, temp);				// (1 - alpha) * output + alpha * input

				pixelsI1 = _mm_srli_si128(pixelsI0, 12);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				temp = _mm_cvtepi32_ps(pixelsI1);					// Convert to float32
				temp = _mm_mul_ps(temp, aprime);					// (1 - alpha) * output
				pixelsF3 = _mm_add_ps(pixelsF3, temp);				// (1 - alpha) * output + alpha * input

				pixelsI0 = _mm_cvttps_epi32(pixelsF0);
				pixelsI1 = _mm_cvttps_epi32(pixelsF1);
				pixelsI0 = _mm_packus_epi32(pixelsI0, pixelsI1);	// lower 8 values (word)
				pixelsI1 = _mm_cvttps_epi32(pixelsF2);
				tempI = _mm_cvttps_epi32(pixelsF3);
				pixelsI1 = _mm_packus_epi32(pixelsI1, tempI);		// upper 8 values (word)

				pixelsI0 = _mm_packus_epi16(pixelsI0, pixelsI1);
				_mm_store_si128(pLocalDst_xmm++, pixelsI0);
			}

			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++, pLocalSrc++)
			{
				vx_float32 temp = ((1 - alpha) * (vx_float32)*pLocalDst) + (alpha * (vx_float32)*pLocalSrc);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc_xmm = (__m128i *) pSrcImage;
			pLocalDst_xmm = (__m128i *) pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				// For the input pixels
				pixelsI0 = _mm_loadu_si128(pLocalSrc_xmm++);

				pixelsI1 = _mm_cvtepu8_epi32(pixelsI0);				// Convert to int32
				pixelsF0 = _mm_cvtepi32_ps(pixelsI1);				// Convert to float32
				pixelsI1 = _mm_srli_si128(pixelsI0, 4);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				pixelsF1 = _mm_cvtepi32_ps(pixelsI1);				// Convert to float32
				pixelsI1 = _mm_srli_si128(pixelsI0, 8);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				pixelsF2 = _mm_cvtepi32_ps(pixelsI1);				// Convert to float32
				pixelsI1 = _mm_srli_si128(pixelsI0, 12);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				pixelsF3 = _mm_cvtepi32_ps(pixelsI1);				// Convert to float32

				pixelsF0 = _mm_mul_ps(pixelsF0, a);					// alpha * input
				pixelsF1 = _mm_mul_ps(pixelsF1, a);					// alpha * input
				pixelsF2 = _mm_mul_ps(pixelsF2, a);					// alpha * input
				pixelsF3 = _mm_mul_ps(pixelsF3, a);					// alpha * input

				// For the output pixels
				pixelsI0 = _mm_loadu_si128(pLocalDst_xmm);

				pixelsI1 = _mm_cvtepu8_epi32(pixelsI0);				// Convert to int32
				temp = _mm_cvtepi32_ps(pixelsI1);					// Convert to float32
				temp = _mm_mul_ps(temp, aprime);					// (1 - alpha) * output
				pixelsF0 = _mm_add_ps(pixelsF0, temp);				// (1 - alpha) * output + alpha * input

				pixelsI1 = _mm_srli_si128(pixelsI0, 4);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				temp = _mm_cvtepi32_ps(pixelsI1);					// Convert to float32
				temp = _mm_mul_ps(temp, aprime);					// (1 - alpha) * output
				pixelsF1 = _mm_add_ps(pixelsF1, temp);				// (1 - alpha) * output + alpha * input

				pixelsI1 = _mm_srli_si128(pixelsI0, 8);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				temp = _mm_cvtepi32_ps(pixelsI1);					// Convert to float32
				temp = _mm_mul_ps(temp, aprime);					// (1 - alpha) * output
				pixelsF2 = _mm_add_ps(pixelsF2, temp);				// (1 - alpha) * output + alpha * input

				pixelsI1 = _mm_srli_si128(pixelsI0, 12);
				pixelsI1 = _mm_cvtepu8_epi32(pixelsI1);				// Convert to int32
				temp = _mm_cvtepi32_ps(pixelsI1);					// Convert to float32
				temp = _mm_mul_ps(temp, aprime);					// (1 - alpha) * output
				pixelsF3 = _mm_add_ps(pixelsF3, temp);				// (1 - alpha) * output + alpha * input

				pixelsI0 = _mm_cvttps_epi32(pixelsF0);
				pixelsI1 = _mm_cvttps_epi32(pixelsF1);
				pixelsI0 = _mm_packus_epi32(pixelsI0, pixelsI1);	// lower 8 values (word)
				pixelsI1 = _mm_cvttps_epi32(pixelsF2);
				tempI = _mm_cvttps_epi32(pixelsF3);
				pixelsI1 = _mm_packus_epi32(pixelsI1, tempI);		// upper 8 values (word)

				pixelsI0 = _mm_packus_epi16(pixelsI0, pixelsI1);
				_mm_storeu_si128(pLocalDst_xmm++, pixelsI0);
			}

			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++, pLocalSrc++)
			{
				vx_float32 temp = ((1 - alpha) * (vx_float32)*pLocalDst) + (alpha * (vx_float32)*pLocalSrc);
				*pLocalDst++ = (vx_uint8)temp;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}

/* The following are hand optimized CPU based kernels for point-multiply functions */
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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi16((short)0x00FF);
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;

	while (pchDst < pchDstlast)
	{
		__m128i * src1	= (__m128i*)pSrcImage1;
		__m128i * src2	= (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 4);

		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);
			pixels2 = _mm_load_si128(src2++);
			pixels3 = _mm_unpackhi_epi8(pixels1, zeros);
			pixels1 = _mm_cvtepu8_epi16(pixels1);
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			pixels3 = _mm_mullo_epi16(pixels3, pixels4);			// src1*src2 for (8-15)
			pixels1 = _mm_mullo_epi16(pixels1, pixels2);			// src1*src2 for (0-7)
			pixels4 = pixels3;
			pixels2 = pixels1;

			// convert to 32 bit0
			pixels2 = _mm_unpackhi_epi16(pixels2, zeros);			// src1*src2 (4-7)
			pixels1 = _mm_cvtepu16_epi32(pixels1);				// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(pixels4, zeros);			// src1*src2 (12-15)
			pixels3 = _mm_cvtepu16_epi32(pixels3);				// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round towards zero - use convert with truncation: cvttps2dq
			pixels1 = _mm_cvttps_epi32(fpels1);
			pixels2 = _mm_cvttps_epi32(fpels2);
			pixels3 = _mm_cvttps_epi32(fpels3);
			pixels4 = _mm_cvttps_epi32(fpels4);

			// pack to unsigned words 
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			pixels3 = _mm_packus_epi32(pixels3, pixels4);

			// mask for wrap/truncation
			pixels1 = _mm_and_si128(pixels1, mask);			// wrap to U8
			pixels3 = _mm_and_si128(pixels3, mask);			// wrap to U8
			// pack to unsigned bytes
			pixels1 = _mm_packus_epi16(pixels1, pixels3);
			// copy to dest
			_mm_store_si128(dst++, pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

// 
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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi16((short)0x00FF);
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;

	while (pchDst < pchDstlast)
	{
		__m128i * src1	= (__m128i*)pSrcImage1;
		__m128i * src2	= (__m128i*)pSrcImage2;
		__m128i * dst	= (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 4);

		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);
			pixels2 = _mm_load_si128(src2++);
			pixels3 = _mm_unpackhi_epi8(pixels1, zeros);
			pixels1 = _mm_cvtepu8_epi16(pixels1);
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			pixels3 = _mm_mullo_epi16(pixels3, pixels4);			// src1*src2 for (8-15)
			pixels1 = _mm_mullo_epi16(pixels1, pixels2);			// src1*src2 for (0-7)
			pixels4 = pixels3;
			pixels2 = pixels1;

			// convert to 32 bit0
			pixels2 = _mm_unpackhi_epi16(pixels2, zeros);			// src1*src2 (4-7)
			pixels1 = _mm_cvtepu16_epi32(pixels1);				// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(pixels4, zeros);			// src1*src2 (12-15)
			pixels3 = _mm_cvtepu16_epi32(pixels3);				// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round to nearest even - use convert with rounding: cvtps2dq
			pixels1 = _mm_cvtps_epi32(fpels1);
			pixels2 = _mm_cvtps_epi32(fpels2);
			pixels3 = _mm_cvtps_epi32(fpels3);
			pixels4 = _mm_cvtps_epi32(fpels4);

			// pack to unsigned words 
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			pixels3 = _mm_packus_epi32(pixels3, pixels4);

			// mask for wrap/truncation
			pixels1 = _mm_and_si128(pixels1, mask);			// wrap to U8
			pixels3 = _mm_and_si128(pixels3, mask);			// wrap to U8

			// pack to unsigned bytes
			pixels1 = _mm_packus_epi16(pixels1, pixels3);
			// copy to dest
			_mm_store_si128(dst++, pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;

}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi16((short)0x7FFF);
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;

	while (pchDst < pchDstlast)
	{
		__m128i * src1	= (__m128i*)pSrcImage1;
		__m128i * src2	= (__m128i*)pSrcImage2;
		__m128i * dst	= (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 4);
		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);
			pixels2 = _mm_load_si128(src2++);
			pixels3 = _mm_unpackhi_epi8(pixels1, zeros);
			pixels1 = _mm_cvtepu8_epi16(pixels1);
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			pixels3 = _mm_mullo_epi16(pixels3, pixels4);			// src1*src2 for (8-15)
			pixels1 = _mm_mullo_epi16(pixels1, pixels2);			// src1*src2 for (0-7)
			pixels4 = pixels3;
			pixels2 = pixels1;

			// convert to 32 bit0
			pixels2 = _mm_unpackhi_epi16(pixels2, zeros);			// src1*src2 (4-7)
			pixels1 = _mm_cvtepu16_epi32(pixels1);				// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(pixels4, zeros);			// src1*src2 (12-15)
			pixels3 = _mm_cvtepu16_epi32(pixels3);				// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round towards zero - use convert with truncation: cvttps2dq
			pixels1 = _mm_cvttps_epi32(fpels1);
			pixels2 = _mm_cvttps_epi32(fpels2);
			pixels3 = _mm_cvttps_epi32(fpels3);
			pixels4 = _mm_cvttps_epi32(fpels4);

			// pack to unsigned words 
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			pixels3 = _mm_packus_epi32(pixels3, pixels4);
			pixels1 = _mm_min_epu16(pixels1, mask);			// clamp to 0x7fff
			pixels3 = _mm_min_epu16(pixels3, mask);			// clamp to 0x7fff

			// pack to unsigned bytes through unsigned saturation
			pixels1 = _mm_packus_epi16(pixels1, pixels3);
			// copy to dest
			_mm_store_si128(dst++, pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;

}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi16((short)0x7FFF);
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrcImage1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst  = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 4);
		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);
			pixels2 = _mm_load_si128(src2++);
			pixels3 = _mm_unpackhi_epi8(pixels1, zeros);
			pixels1 = _mm_cvtepu8_epi16(pixels1);
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			pixels3 = _mm_mullo_epi16(pixels3, pixels4);			// src1*src2 for (8-15)
			pixels1 = _mm_mullo_epi16(pixels1, pixels2);			// src1*src2 for (0-7)
			pixels4 = pixels3;
			pixels2 = pixels1;

			// convert to 32 bit0
			pixels2 = _mm_unpackhi_epi16(pixels2, zeros);			// src1*src2 (4-7)
			pixels1 = _mm_cvtepu16_epi32(pixels1);				// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(pixels4, zeros);			// src1*src2 (12-15)
			pixels3 = _mm_cvtepu16_epi32(pixels3);				// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round to nearest even - use convert with rounding: cvtps2dq
			pixels1 = _mm_cvtps_epi32(fpels1);
			pixels2 = _mm_cvtps_epi32(fpels2);
			pixels3 = _mm_cvtps_epi32(fpels3);
			pixels4 = _mm_cvtps_epi32(fpels4);

			// pack to unsigned words 
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			pixels3 = _mm_packus_epi32(pixels3, pixels4);
			pixels1 = _mm_min_epu16(pixels1, mask);			// clamp to 0x7fff
			pixels3 = _mm_min_epu16(pixels3, mask);			// clamp to 0x7fff

			// pack to unsigned bytes though unsigned saturation
			pixels1 = _mm_packus_epi16(pixels1, pixels3);
			// copy to dest
			_mm_store_si128(dst++, pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi32((int)0x0000FFFF);
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrcImage1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);
			pixels2 = _mm_load_si128(src2++);
			pixels3 = _mm_unpackhi_epi8(pixels1, zeros);
			pixels1 = _mm_cvtepu8_epi16(pixels1);
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			pixels3 = _mm_mullo_epi16(pixels3, pixels4);			// src1*src2 for (8-15)
			pixels1 = _mm_mullo_epi16(pixels1, pixels2);			// src1*src2 for (0-7)
			pixels4 = pixels3;
			pixels2 = pixels1;

			// convert to 32 bit0
			pixels2 = _mm_unpackhi_epi16(pixels2, zeros);			// src1*src2 (4-7)
			pixels1 = _mm_cvtepu16_epi32(pixels1);				// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(pixels4, zeros);			// src1*src2 (12-15)
			pixels3 = _mm_cvtepu16_epi32(pixels3);				// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round towards zero - use convert with truncation: cvttps2dq
			pixels1 = _mm_cvttps_epi32(fpels1);
			pixels2 = _mm_cvttps_epi32(fpels2);
			pixels3 = _mm_cvttps_epi32(fpels3);
			pixels4 = _mm_cvttps_epi32(fpels4);

			// mask for wrap/truncation
			pixels1 = _mm_and_si128(pixels1, mask);
			pixels2 = _mm_and_si128(pixels2, mask);
			pixels3 = _mm_and_si128(pixels3, mask);
			pixels4 = _mm_and_si128(pixels4, mask);

			// pack with unsigned saturation 
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			pixels3 = _mm_packus_epi32(pixels3, pixels4);

			// copy to dest
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dst++, pixels3);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi32((int)0x0000FFFF);
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	
	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrcImage1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);
			pixels2 = _mm_load_si128(src2++);
			pixels3 = _mm_unpackhi_epi8(pixels1, zeros);
			pixels1 = _mm_cvtepu8_epi16(pixels1);
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			pixels3 = _mm_mullo_epi16(pixels3, pixels4);			// src1*src2 for (8-15)
			pixels1 = _mm_mullo_epi16(pixels1, pixels2);			// src1*src2 for (0-7)
			pixels4 = pixels3;
			pixels2 = pixels1;
			// convert to 32 bit0
			pixels2 = _mm_unpackhi_epi16(pixels2, zeros);			// src1*src2 (4-7)
			pixels1 = _mm_cvtepu16_epi32(pixels1);				// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(pixels4, zeros);			// src1*src2 (12-15)
			pixels3 = _mm_cvtepu16_epi32(pixels3);				// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round to nearest even: cvtps2dq
			pixels1 = _mm_cvtps_epi32(fpels1);
			pixels2 = _mm_cvtps_epi32(fpels2);
			pixels3 = _mm_cvtps_epi32(fpels3);
			pixels4 = _mm_cvtps_epi32(fpels4);

			// mask for wrap/truncation
			pixels1 = _mm_and_si128(pixels1, mask);
			pixels2 = _mm_and_si128(pixels2, mask);
			pixels3 = _mm_and_si128(pixels3, mask);
			pixels4 = _mm_and_si128(pixels4, mask);

			// pack with unsigned saturation 
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			pixels3 = _mm_packus_epi32(pixels3, pixels4);

			// copy to dest
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dst++, pixels3);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}


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
)
{
	__m128i pixels1, pixels2, pixels3, pixels4;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrcImage1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);
			pixels2 = _mm_load_si128(src2++);
			pixels3 = _mm_unpackhi_epi8(pixels1, zeros);
			pixels1 = _mm_cvtepu8_epi16(pixels1);
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			pixels3 = _mm_mullo_epi16(pixels3, pixels4);			// src1*src2 for (8-15)
			pixels1 = _mm_mullo_epi16(pixels1, pixels2);			// src1*src2 for (0-7)
			pixels4 = pixels3;
			pixels2 = pixels1;
			// convert to 32 bit0
			pixels2 = _mm_unpackhi_epi16(pixels2, zeros);			// src1*src2 (4-7)
			pixels1 = _mm_cvtepu16_epi32(pixels1);				// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(pixels4, zeros);			// src1*src2 (12-15)
			pixels3 = _mm_cvtepu16_epi32(pixels3);				// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round towards zero - use convert with truncation: cvttps2dq
			pixels1 = _mm_cvttps_epi32(fpels1);
			pixels2 = _mm_cvttps_epi32(fpels2);
			pixels3 = _mm_cvttps_epi32(fpels3);
			pixels4 = _mm_cvttps_epi32(fpels4);

			// pack signed saturation 
			pixels1 = _mm_packs_epi32(pixels1, pixels2);
			pixels3 = _mm_packs_epi32(pixels3, pixels4);

			// copy to dest
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dst++, pixels3);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrcImage1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);
			pixels2 = _mm_load_si128(src2++);
			pixels3 = _mm_unpackhi_epi8(pixels1, zeros);
			pixels1 = _mm_cvtepu8_epi16(pixels1);
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			pixels3 = _mm_mullo_epi16(pixels3, pixels4);			// src1*src2 for (8-15)
			pixels1 = _mm_mullo_epi16(pixels1, pixels2);			// src1*src2 for (0-7)
			pixels4 = pixels3;
			pixels2 = pixels1;
			// convert to 32 bit0
			pixels2 = _mm_unpackhi_epi16(pixels2, zeros);			// src1*src2 (4-7)
			pixels1 = _mm_cvtepu16_epi32(pixels1);				// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(pixels4, zeros);			// src1*src2 (12-15)
			pixels3 = _mm_cvtepu16_epi32(pixels3);				// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round to nearest even: cvtps2dq
			pixels1 = _mm_cvtps_epi32(fpels1);
			pixels2 = _mm_cvtps_epi32(fpels2);
			pixels3 = _mm_cvtps_epi32(fpels3);
			pixels4 = _mm_cvtps_epi32(fpels4);

			// pack signed saturation 
			pixels1 = _mm_packs_epi32(pixels1, pixels2);
			pixels3 = _mm_packs_epi32(pixels3, pixels4);

			// copy to dest
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dst++, pixels3);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask, temp1, temp2;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi32((int)0x0000FFFF);
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pSrc1 = (unsigned char *)pSrcImage1;

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrc1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
			pixels3 = _mm_load_si128(src1++);		// src1 (8-15)
			pixels2 = _mm_load_si128(src2++);		// src2 (0-15)
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
			temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
			// do mulhi as well since we are multiplying 16x8
			pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
			pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

			// unpack to 32 bit result
			pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
			pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
			pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round towards zero - use convert with truncation: cvttps2dq
			pixels1 = _mm_cvttps_epi32(fpels1);
			pixels2 = _mm_cvttps_epi32(fpels2);
			pixels3 = _mm_cvttps_epi32(fpels3);
			pixels4 = _mm_cvttps_epi32(fpels4);

			// mask for wrap/truncation
			pixels1 = _mm_and_si128(pixels1, mask);
			pixels2 = _mm_and_si128(pixels2, mask);
			pixels3 = _mm_and_si128(pixels3, mask);
			pixels4 = _mm_and_si128(pixels4, mask);

			// pack signed saturation 
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			pixels3 = _mm_packus_epi32(pixels3, pixels4);

			// copy to dest
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dst++, pixels3);
		}
		pSrc1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask, temp1, temp2;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi32((int)0x0000FFFF);
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pSrc1 = (unsigned char *)pSrcImage1;
	uint32_t fpState = agoControlFpSetRoundEven();

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrc1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
			pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
			pixels2 = _mm_load_si128(src2++);		// src2 (0-15)
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
			temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
			// do mulhi as well since we are multiplying 16x8
			pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
			pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

			// unpack to 32 bit result
			pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
			pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
			pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round towards nearest even
			pixels1 = _mm_cvtps_epi32(fpels1);
			pixels2 = _mm_cvtps_epi32(fpels2);
			pixels3 = _mm_cvtps_epi32(fpels3);
			pixels4 = _mm_cvtps_epi32(fpels4);

			// mask for wrap/truncation
			pixels1 = _mm_and_si128(pixels1, mask);
			pixels2 = _mm_and_si128(pixels2, mask);
			pixels3 = _mm_and_si128(pixels3, mask);
			pixels4 = _mm_and_si128(pixels4, mask);

			// pack to words
			pixels1 = _mm_packus_epi32(pixels1, pixels2);
			pixels3 = _mm_packus_epi32(pixels3, pixels4);

			// copy to dest
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dst++, pixels3);
		}
		pSrc1	+= srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	agoControlFpReset(fpState);

	return AGO_SUCCESS;
}


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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, temp1, temp2;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pSrc1 = (unsigned char *)pSrcImage1;

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrc1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);

		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
			pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
			pixels2 = _mm_load_si128(src2++);		// src2 (0-15)
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
			temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
			// do mulhi as well since we are multiplying 16x8
			pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
			pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

			// unpack to 32 bit result
			pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
			pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
			pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round towards zero - use convert with truncation: cvttps2dq
			pixels1 = _mm_cvttps_epi32(fpels1);
			pixels2 = _mm_cvttps_epi32(fpels2);
			pixels3 = _mm_cvttps_epi32(fpels3);
			pixels4 = _mm_cvttps_epi32(fpels4);

			// pack signed saturation 
			pixels1 = _mm_packs_epi32(pixels1, pixels2);
			pixels3 = _mm_packs_epi32(pixels3, pixels4);

			// copy to dest
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dst++, pixels3);
		}
		pSrc1  += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}


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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, temp1, temp2;
	__m128  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	const __m128 fscale = _mm_set1_ps(scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pSrc1 = (unsigned char *)pSrcImage1;
	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrc1;
		__m128i * src2 = (__m128i*)pSrcImage2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);

		while (dst <= dstlast)
		{
			pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
			pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
			pixels2 = _mm_load_si128(src2++);		// src2 (0-15)
			pixels4 = _mm_unpackhi_epi8(pixels2, zeros);
			pixels2 = _mm_cvtepu8_epi16(pixels2);
			temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
			temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
			// do mulhi as well since we are multiplying 16x8
			pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
			pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

			// unpack to 32 bit result
			pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
			pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
			pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
			pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)
			// convert to packed single precision float of src1*src2
			fpels1 = _mm_cvtepi32_ps(pixels1);
			fpels2 = _mm_cvtepi32_ps(pixels2);
			fpels3 = _mm_cvtepi32_ps(pixels3);
			fpels4 = _mm_cvtepi32_ps(pixels4);

			// multiply with scale
			fpels1 = _mm_mul_ps(fpels1, fscale);
			fpels2 = _mm_mul_ps(fpels2, fscale);
			fpels3 = _mm_mul_ps(fpels3, fscale);
			fpels4 = _mm_mul_ps(fpels4, fscale);

			// round towards zero - use convert with round
			pixels1 = _mm_cvtps_epi32(fpels1);
			pixels2 = _mm_cvtps_epi32(fpels2);
			pixels3 = _mm_cvtps_epi32(fpels3);
			pixels4 = _mm_cvtps_epi32(fpels4);

			// pack to words 
			pixels1 = _mm_packs_epi32(pixels1, pixels2);
			pixels3 = _mm_packs_epi32(pixels3, pixels4);

			// copy to dest
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dst++, pixels3);
		}
		pSrc1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask, temp1, temp2;
	__m128d  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi32((int)0x0000FFFF);
	const __m128d fscale = _mm_set1_pd((double)scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pSrc1 = (unsigned char *)pSrcImage1;
	unsigned char *pSrc2 = (unsigned char *)pSrcImage2;
	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrc1;
		__m128i * src2 = (__m128i*)pSrc2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		if (scale == 1.0f){
			while (dst <= dstlast)
			{
				pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
				pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
				pixels2 = _mm_load_si128(src2++);		// src2 (0-7)
				pixels4 = _mm_load_si128(src2++);	// src2 (8-15)
				temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
				temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
				// do mulhi as well since we are multiplying 16x8
				pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
				pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

				// unpack to 32 bit result
				pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
				pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
				pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
				pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

				// mask for wrap/truncation
				pixels1 = _mm_and_si128(pixels1, mask);
				pixels2 = _mm_and_si128(pixels2, mask);
				pixels3 = _mm_and_si128(pixels3, mask);
				pixels4 = _mm_and_si128(pixels4, mask);

				// pack to words 
				pixels1 = _mm_packus_epi32(pixels1, pixels2);
				pixels3 = _mm_packus_epi32(pixels3, pixels4);

				// copy to dest
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
			}
		}
		else
		{
			int x = 0;
			while (dst <= dstlast)
			{
				__m128d  fpels5, fpels6, fpels7, fpels8;
				pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
				pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
				pixels2 = _mm_load_si128(src2++);		// src2 (0-7)
				pixels4 = _mm_load_si128(src2++);	// src2 (8-15)
				temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
				temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
				// do mulhi as well since we are multiplying 16x8
				pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
				pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

				// unpack to 32 bit result
				pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
				pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
				pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
				pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

				// convert to packed double precision float of src1*src2
				fpels1 = _mm_cvtepi32_pd(pixels1);
				fpels2 = _mm_cvtepi32_pd(pixels2);
				fpels3 = _mm_cvtepi32_pd(pixels3);
				fpels4 = _mm_cvtepi32_pd(pixels4);

				
				fpels5 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels1, 0x4e));
				fpels6 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels2, 0x4e));
				fpels7 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels3, 0x4e));
				fpels8 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels4, 0x4e));

				// multiply with scale
				fpels1 = _mm_mul_pd(fpels1, fscale);
				fpels2 = _mm_mul_pd(fpels2, fscale);
				fpels3 = _mm_mul_pd(fpels3, fscale);
				fpels4 = _mm_mul_pd(fpels4, fscale);
				fpels5 = _mm_mul_pd(fpels5, fscale);
				fpels6 = _mm_mul_pd(fpels6, fscale);
				fpels7 = _mm_mul_pd(fpels7, fscale);
				fpels8 = _mm_mul_pd(fpels8, fscale);

				// round towards zero - use convert with truncation: cvttps2dq
				pixels1 = _mm_cvttpd_epi32(fpels1);
				pixels2 = _mm_cvttpd_epi32(fpels2);
				pixels3 = _mm_cvttpd_epi32(fpels3);
				pixels4 = _mm_cvttpd_epi32(fpels4);

				pixels1 = _mm_unpacklo_epi64(pixels1, _mm_cvttpd_epi32(fpels5));
				pixels2 = _mm_unpacklo_epi64(pixels2, _mm_cvttpd_epi32(fpels6));
				pixels3 = _mm_unpacklo_epi64(pixels3, _mm_cvttpd_epi32(fpels7));
				pixels4 = _mm_unpacklo_epi64(pixels4, _mm_cvttpd_epi32(fpels8));

				// mask for wrap/truncation
				pixels1 = _mm_and_si128(pixels1, mask);
				pixels2 = _mm_and_si128(pixels2, mask);
				pixels3 = _mm_and_si128(pixels3, mask);
				pixels4 = _mm_and_si128(pixels4, mask);

				// pack to words 
				pixels1 = _mm_packus_epi32(pixels1, pixels2);
				pixels3 = _mm_packus_epi32(pixels3, pixels4);

				// copy to dest
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
				x += 16;
			}
		}
		//y++;
		pSrc1 += srcImage1StrideInBytes;
		pSrc2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, mask, temp1, temp2;
	__m128d  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	mask = _mm_set1_epi32((int)0x0000FFFF);
	const __m128d fscale = _mm_set1_pd((double)scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pSrc1 = (unsigned char *)pSrcImage1;
	unsigned char *pSrc2 = (unsigned char *)pSrcImage2;

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrc1;
		__m128i * src2 = (__m128i*)pSrc2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		if (scale == 1.0f){
			while (dst <= dstlast)
			{
				pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
				pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
				pixels2 = _mm_load_si128(src2++);		// src2 (0-7)
				pixels4 = _mm_load_si128(src2++);	// src2 (8-15)
				temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
				temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
				// do mulhi as well since we are multiplying 16x8
				pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
				pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

				// unpack to 32 bit result
				pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
				pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
				pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
				pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

				// mask for wrap/truncation
				pixels1 = _mm_and_si128(pixels1, mask);
				pixels2 = _mm_and_si128(pixels2, mask);
				pixels3 = _mm_and_si128(pixels3, mask);
				pixels4 = _mm_and_si128(pixels4, mask);

				// pack to words 
				pixels1 = _mm_packus_epi32(pixels1, pixels2);
				pixels3 = _mm_packus_epi32(pixels3, pixels4);

				// copy to dest
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
			}
		}
		else
		{

			while (dst <= dstlast)
			{
				__m128d  fpels5, fpels6, fpels7, fpels8;

				pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
				pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
				pixels2 = _mm_load_si128(src2++);		// src2 (0-7)
				pixels4 = _mm_load_si128(src2++);	// src2 (8-15)

				temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
				temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
				// do mulhi as well since we are multiplying 16x8
				pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
				pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

				// unpack to 32 bit result
				pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
				pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
				pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
				pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

				// convert to packed double precision float of src1*src2
				fpels1 = _mm_cvtepi32_pd(pixels1);
				fpels2 = _mm_cvtepi32_pd(pixels2);
				fpels3 = _mm_cvtepi32_pd(pixels3);
				fpels4 = _mm_cvtepi32_pd(pixels4);
				fpels5 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels1, 0x4e));
				fpels6 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels2, 0x4e));
				fpels7 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels3, 0x4e));
				fpels8 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels4, 0x4e));

				// multiply with scale
				fpels1 = _mm_mul_pd(fpels1, fscale);
				fpels2 = _mm_mul_pd(fpels2, fscale);
				fpels3 = _mm_mul_pd(fpels3, fscale);
				fpels4 = _mm_mul_pd(fpels4, fscale);
				fpels5 = _mm_mul_pd(fpels5, fscale);
				fpels6 = _mm_mul_pd(fpels6, fscale);
				fpels7 = _mm_mul_pd(fpels7, fscale);
				fpels8 = _mm_mul_pd(fpels8, fscale);

				// round towards zero - use convert with truncation: cvttps2dq
				pixels1 = _mm_cvtpd_epi32(fpels1);
				pixels2 = _mm_cvtpd_epi32(fpels2);
				pixels3 = _mm_cvtpd_epi32(fpels3);
				pixels4 = _mm_cvtpd_epi32(fpels4);
				pixels1 = _mm_unpacklo_epi64(pixels1, _mm_cvtpd_epi32(fpels5));
				pixels2 = _mm_unpacklo_epi64(pixels2, _mm_cvtpd_epi32(fpels6));
				pixels3 = _mm_unpacklo_epi64(pixels3, _mm_cvtpd_epi32(fpels7));
				pixels4 = _mm_unpacklo_epi64(pixels4, _mm_cvtpd_epi32(fpels8));

				// mask for wrap/truncation
				pixels1 = _mm_and_si128(pixels1, mask);
				pixels2 = _mm_and_si128(pixels2, mask);
				pixels3 = _mm_and_si128(pixels3, mask);
				pixels4 = _mm_and_si128(pixels4, mask);

				// pack signed saturation 
				pixels1 = _mm_packus_epi32(pixels1, pixels2);
				pixels3 = _mm_packus_epi32(pixels3, pixels4);

				// copy to dest
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
			}
		}
		pSrc1 += srcImage1StrideInBytes;
		pSrc2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, temp1, temp2;
	__m128d  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	const __m128d fscale = _mm_set1_pd((double)scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pSrc1 = (unsigned char *)pSrcImage1;
	unsigned char *pSrc2 = (unsigned char *)pSrcImage2;

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrc1;
		__m128i * src2 = (__m128i*)pSrc2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		if (scale == 1.0f){
			while (dst <= dstlast)
			{
				pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
				pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
				pixels2 = _mm_load_si128(src2++);		// src2 (0-7)
				pixels4 = _mm_load_si128(src2++);	// src2 (8-15)
				temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
				temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
				// do mulhi as well since we are multiplying 16x8
				pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
				pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

				// unpack to 32 bit result
				pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
				pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
				pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
				pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)
				// pack to words 
				pixels1 = _mm_packs_epi32(pixels1, pixels2);
				pixels3 = _mm_packs_epi32(pixels3, pixels4);

				// copy to dest
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
			}
		}
		else
		{
			while (dst <= dstlast)
			{
				__m128d  fpels5, fpels6, fpels7, fpels8;
				pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
				pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
				pixels2 = _mm_load_si128(src2++);		// src2 (0-7)
				pixels4 = _mm_load_si128(src2++);	// src2 (8-15)

				temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
				temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
				// do mulhi as well since we are multiplying 16x8
				pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
				pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

				// unpack to 32 bit result
				pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
				pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
				pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
				pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

				// convert to packed double precision float of src1*src2
				fpels1 = _mm_cvtepi32_pd(pixels1);
				fpels2 = _mm_cvtepi32_pd(pixels2);
				fpels3 = _mm_cvtepi32_pd(pixels3);
				fpels4 = _mm_cvtepi32_pd(pixels4);
				fpels5 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels1, 0x4e));
				fpels6 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels2, 0x4e));
				fpels7 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels3, 0x4e));
				fpels8 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels4, 0x4e));

				// multiply with scale
				fpels1 = _mm_mul_pd(fpels1, fscale);
				fpels2 = _mm_mul_pd(fpels2, fscale);
				fpels3 = _mm_mul_pd(fpels3, fscale);
				fpels4 = _mm_mul_pd(fpels4, fscale);
				fpels5 = _mm_mul_pd(fpels5, fscale);
				fpels6 = _mm_mul_pd(fpels6, fscale);
				fpels7 = _mm_mul_pd(fpels7, fscale);
				fpels8 = _mm_mul_pd(fpels8, fscale);

				// round towards zero - use convert with truncation: cvttps2dq
				pixels1 = _mm_cvttpd_epi32(fpels1);
				pixels2 = _mm_cvttpd_epi32(fpels2);
				pixels3 = _mm_cvttpd_epi32(fpels3);
				pixels4 = _mm_cvttpd_epi32(fpels4);
				pixels1 = _mm_unpacklo_epi64(pixels1, _mm_cvttpd_epi32(fpels5));
				pixels2 = _mm_unpacklo_epi64(pixels2, _mm_cvttpd_epi32(fpels6));
				pixels3 = _mm_unpacklo_epi64(pixels3, _mm_cvttpd_epi32(fpels7));
				pixels4 = _mm_unpacklo_epi64(pixels4, _mm_cvttpd_epi32(fpels8));

				// pack signed saturation 
				pixels1 = _mm_packs_epi32(pixels1, pixels2);
				pixels3 = _mm_packs_epi32(pixels3, pixels4);

				// copy to dest
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
			}
		}
		pSrc1 += srcImage1StrideInBytes;
		pSrc2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
)
{
	// do generic floating point calculation
	__m128i pixels1, pixels2, pixels3, pixels4, temp1, temp2;
	__m128d  fpels1, fpels2, fpels3, fpels4;
	const __m128i zeros = _mm_setzero_si128();
	const __m128d fscale = _mm_set1_pd((double)scale);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pSrc1 = (unsigned char *)pSrcImage1;
	unsigned char *pSrc2 = (unsigned char *)pSrcImage2;

	while (pchDst < pchDstlast)
	{
		__m128i * src1 = (__m128i*)pSrc1;
		__m128i * src2 = (__m128i*)pSrc2;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 3);
		if (scale == 1.0f){
			while (dst <= dstlast)
			{
				pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
				pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
				pixels2 = _mm_load_si128(src2++);		// src2 (0-7)
				pixels4 = _mm_load_si128(src2++);	// src2 (8-15)
				temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
				temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
				// do mulhi as well since we are multiplying 16x8
				pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
				pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

				// unpack to 32 bit result
				pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
				pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
				pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
				pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)
				// pack to words 
				pixels1 = _mm_packs_epi32(pixels1, pixels2);
				pixels3 = _mm_packs_epi32(pixels3, pixels4);

				// copy to dest
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
			}
		}
		else
		{
			while (dst <= dstlast)
			{
				__m128d  fpels5, fpels6, fpels7, fpels8;
				pixels1 = _mm_load_si128(src1++);		// src1 (0-7)
				pixels3 = _mm_load_si128(src1++);	// src1 (8-15)
				pixels2 = _mm_load_si128(src2++);		// src2 (0-7)
				pixels4 = _mm_load_si128(src2++);	// src2 (8-15)

				temp1 = _mm_mullo_epi16(pixels3, pixels4);			// low for src1*src2 for (8-15)
				temp2 = _mm_mullo_epi16(pixels1, pixels2);			// low for src1*src2 for (0-7)
				// do mulhi as well since we are multiplying 16x8
				pixels3 = _mm_mulhi_epi16(pixels3, pixels4);					// high for src1*src2 for (8-15)
				pixels1 = _mm_mulhi_epi16(pixels1, pixels2);					// high for src1*src2 for (0-7)

				// unpack to 32 bit result
				pixels2 = _mm_unpackhi_epi16(temp2, pixels1);		// src1*src2 (4-7)
				pixels1 = _mm_unpacklo_epi16(temp2, pixels1);		// src1*src2 (0-3)
				pixels4 = _mm_unpackhi_epi16(temp1, pixels3);		// src1*src2 (12-15)
				pixels3 = _mm_unpacklo_epi16(temp1, pixels3);		// src1*src2 (8-11)

				// convert to packed double precision float of src1*src2
				fpels1 = _mm_cvtepi32_pd(pixels1);
				fpels2 = _mm_cvtepi32_pd(pixels2);
				fpels3 = _mm_cvtepi32_pd(pixels3);
				fpels4 = _mm_cvtepi32_pd(pixels4);
				fpels5 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels1, 0x4e));
				fpels6 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels2, 0x4e));
				fpels7 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels3, 0x4e));
				fpels8 = _mm_cvtepi32_pd(_mm_shuffle_epi32(pixels4, 0x4e));

				// multiply with scale
				fpels1 = _mm_mul_pd(fpels1, fscale);
				fpels2 = _mm_mul_pd(fpels2, fscale);
				fpels3 = _mm_mul_pd(fpels3, fscale);
				fpels4 = _mm_mul_pd(fpels4, fscale);
				fpels5 = _mm_mul_pd(fpels5, fscale);
				fpels6 = _mm_mul_pd(fpels6, fscale);
				fpels7 = _mm_mul_pd(fpels7, fscale);
				fpels8 = _mm_mul_pd(fpels8, fscale);

				// round towards zero - use convert with truncation: cvttps2dq
				pixels1 = _mm_cvtpd_epi32(fpels1);
				pixels2 = _mm_cvtpd_epi32(fpels2);
				pixels3 = _mm_cvtpd_epi32(fpels3);
				pixels4 = _mm_cvtpd_epi32(fpels4);
				pixels1 = _mm_unpacklo_epi64(pixels1, _mm_cvtpd_epi32(fpels5));
				pixels2 = _mm_unpacklo_epi64(pixels2, _mm_cvtpd_epi32(fpels6));
				pixels3 = _mm_unpacklo_epi64(pixels3, _mm_cvtpd_epi32(fpels7));
				pixels4 = _mm_unpacklo_epi64(pixels4, _mm_cvtpd_epi32(fpels8));

				// pack signed saturation 
				pixels1 = _mm_packs_epi32(pixels1, pixels2);
				pixels3 = _mm_packs_epi32(pixels3, pixels4);

				// copy to dest
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
			}
		}
		pSrc1 += srcImage1StrideInBytes;
		pSrc2 += srcImage2StrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_MeanStdDev_DATA_U8
	(
		vx_float32  * pSum,
		vx_float32  * pSumOfSquared,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	unsigned char * pLocalSrc;
	__m128i pixels, pixels_16, pixels_32, pixels_64;
	__m128i zeromask = _mm_setzero_si128();
	__m128i sum = _mm_setzero_si128();
	__m128i sum_squared = _mm_setzero_si128();

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - prefixWidth - postfixWidth;
	if(alignedWidth < 0) {
		postfixWidth = prefixWidth + postfixWidth + alignedWidth;
		prefixWidth = 0;
		alignedWidth = 0;
	}
	unsigned int prefixSum = 0, postfixSum = 0;
	unsigned long long prefixSumSquared = 0, postfixSumSquared = 0;

	int height = (int) srcHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *) pSrcImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			prefixSum += (unsigned int) *pLocalSrc;
			prefixSumSquared += (unsigned long long)*pLocalSrc * (unsigned long long)*pLocalSrc;
		}
		int width = (int) (alignedWidth >> 4);								// 16 pixels processed at a time
		while (width)
		{
			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			pixels_16 = _mm_unpackhi_epi8(pixels, zeromask);				// 15, 14, 13, 12, 11, 10, 9, 8
			pixels_32 = _mm_unpackhi_epi16(pixels_16, zeromask);			// 15, 14, 13, 12

			sum = _mm_add_epi32(sum, pixels_32);							// Pixels 15, 14, 13, 12
			pixels_64 = _mm_unpackhi_epi32(pixels_32, zeromask);			// 15, 14
			pixels_32 = _mm_cvtepi32_epi64(pixels_32);						// 13, 12
			pixels_64 = _mm_mul_epu32(pixels_64, pixels_64);				// square
			pixels_32 = _mm_mul_epu32(pixels_32, pixels_32);
			sum_squared = _mm_add_epi64(sum_squared, pixels_64);
			sum_squared = _mm_add_epi64(sum_squared, pixels_32);

			pixels_32 = _mm_cvtepi16_epi32(pixels_16);
			sum = _mm_add_epi32(sum, pixels_32);							// Pixels 11, 10, 9, 8
			pixels_64 = _mm_unpackhi_epi32(pixels_32, zeromask);			// 11, 10
			pixels_32 = _mm_cvtepi32_epi64(pixels_32);						// 9, 8
			pixels_64 = _mm_mul_epu32(pixels_64, pixels_64);				// square
			pixels_32 = _mm_mul_epu32(pixels_32, pixels_32);
			sum_squared = _mm_add_epi64(sum_squared, pixels_64);
			sum_squared = _mm_add_epi64(sum_squared, pixels_32);

			pixels_16 = _mm_cvtepu8_epi16(pixels);							// 7, 6, 5, 4, 3, 2, 1, 0
			pixels_32 = _mm_unpackhi_epi16(pixels_16, zeromask);			// 7, 6, 5, 4

			sum = _mm_add_epi32(sum, pixels_32);							// Pixels 7, 6, 5, 4
			pixels_64 = _mm_unpackhi_epi32(pixels_32, zeromask);			// 7, 6
			pixels_32 = _mm_cvtepi32_epi64(pixels_32);						// 5, 4
			pixels_64 = _mm_mul_epu32(pixels_64, pixels_64);				// square
			pixels_32 = _mm_mul_epu32(pixels_32, pixels_32);
			sum_squared = _mm_add_epi64(sum_squared, pixels_64);
			sum_squared = _mm_add_epi64(sum_squared, pixels_32);

			pixels_32 = _mm_cvtepi16_epi32(pixels_16);
			sum = _mm_add_epi32(sum, pixels_32);							// Pixels 3, 2, 1, 0
			pixels_64 = _mm_unpackhi_epi32(pixels_32, zeromask);			// 3, 2
			pixels_32 = _mm_cvtepi32_epi64(pixels_32);						// 1, 0
			pixels_64 = _mm_mul_epu32(pixels_64, pixels_64);				// square
			pixels_32 = _mm_mul_epu32(pixels_32, pixels_32);
			sum_squared = _mm_add_epi64(sum_squared, pixels_64);
			sum_squared = _mm_add_epi64(sum_squared, pixels_32);

			pLocalSrc += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			postfixSum += (unsigned int)*pLocalSrc;
			postfixSumSquared += (unsigned long long)*pLocalSrc * (unsigned long long)*pLocalSrc;
		}

		pSrcImage += srcImageStrideInBytes;
		height--;
	}

	sum = _mm_hadd_epi32(sum, sum);											// Lowest int of sum has sum of last two ints of sum
	sum = _mm_hadd_epi32(sum, sum);											// Lowest int of sum has the sum of all four ints
	pixels = _mm_srli_si128(sum_squared, 8);
	sum_squared = _mm_add_epi64(sum_squared, pixels);

	*pSum = (vx_float32)(M128I(sum).m128i_u32[0] + prefixSum + postfixSum);
	*pSumOfSquared = (vx_float32)(M128I(sum_squared).m128i_u64[0] + prefixSumSquared + postfixSumSquared);

	return AGO_SUCCESS;
}

int HafCpu_MeanStdDev_DATA_U1
	(
		vx_float32  * pSum,
		vx_float32  * pSumOfSquared,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	unsigned char * pLocalSrc;
	__m128i pixels, pixels_16, pixels_32, pixels_64;
	__m128i zeromask = _mm_setzero_si128();
	__m128i sum = _mm_setzero_si128();
	__m128i sum_squared = _mm_setzero_si128();
	
	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - prefixWidth - postfixWidth;
	unsigned int prefixSum = 0, postfixSum = 0;
	unsigned long long prefixSumSquared = 0, postfixSumSquared = 0;

	int height = (int) srcHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *) pSrcImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			prefixSum += (unsigned int) *pLocalSrc;
			prefixSumSquared += (unsigned long long)*pLocalSrc * (unsigned long long)*pLocalSrc;
		}
		int width = (int) (alignedWidth >> 4);								// 16 pixels processed at a time
		while (width)
		{
			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			pixels_16 = _mm_unpackhi_epi8(pixels, zeromask);				// 15, 14, 13, 12, 11, 10, 9, 8
			pixels_32 = _mm_unpackhi_epi16(pixels_16, zeromask);			// 15, 14, 13, 12

			sum = _mm_add_epi32(sum, pixels_32);							// Pixels 15, 14, 13, 12
			pixels_64 = _mm_unpackhi_epi32(pixels_32, zeromask);			// 15, 14
			pixels_32 = _mm_cvtepi32_epi64(pixels_32);						// 13, 12
			pixels_64 = _mm_mul_epu32(pixels_64, pixels_64);				// square
			pixels_32 = _mm_mul_epu32(pixels_32, pixels_32);
			sum_squared = _mm_add_epi64(sum_squared, pixels_64);
			sum_squared = _mm_add_epi64(sum_squared, pixels_32);

			pixels_32 = _mm_cvtepi16_epi32(pixels_16);
			sum = _mm_add_epi32(sum, pixels_32);							// Pixels 11, 10, 9, 8
			pixels_64 = _mm_unpackhi_epi32(pixels_32, zeromask);			// 11, 10
			pixels_32 = _mm_cvtepi32_epi64(pixels_32);						// 9, 8
			pixels_64 = _mm_mul_epu32(pixels_64, pixels_64);				// square
			pixels_32 = _mm_mul_epu32(pixels_32, pixels_32);
			sum_squared = _mm_add_epi64(sum_squared, pixels_64);
			sum_squared = _mm_add_epi64(sum_squared, pixels_32);

			pixels_16 = _mm_cvtepu8_epi16(pixels);							// 7, 6, 5, 4, 3, 2, 1, 0
			pixels_32 = _mm_unpackhi_epi16(pixels_16, zeromask);			// 7, 6, 5, 4

			sum = _mm_add_epi32(sum, pixels_32);							// Pixels 7, 6, 5, 4
			pixels_64 = _mm_unpackhi_epi32(pixels_32, zeromask);			// 7, 6
			pixels_32 = _mm_cvtepi32_epi64(pixels_32);						// 5, 4
			pixels_64 = _mm_mul_epu32(pixels_64, pixels_64);				// square
			pixels_32 = _mm_mul_epu32(pixels_32, pixels_32);
			sum_squared = _mm_add_epi64(sum_squared, pixels_64);
			sum_squared = _mm_add_epi64(sum_squared, pixels_32);

			pixels_32 = _mm_cvtepi16_epi32(pixels_16);
			sum = _mm_add_epi32(sum, pixels_32);							// Pixels 3, 2, 1, 0
			pixels_64 = _mm_unpackhi_epi32(pixels_32, zeromask);			// 3, 2
			pixels_32 = _mm_cvtepi32_epi64(pixels_32);						// 1, 0
			pixels_64 = _mm_mul_epu32(pixels_64, pixels_64);				// square
			pixels_32 = _mm_mul_epu32(pixels_32, pixels_32);
			sum_squared = _mm_add_epi64(sum_squared, pixels_64);
			sum_squared = _mm_add_epi64(sum_squared, pixels_32);

			pLocalSrc += 16;
			width--;
		}

		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			postfixSum += (unsigned int)*pLocalSrc;
			postfixSumSquared += (unsigned long long)*pLocalSrc * (unsigned long long)*pLocalSrc;
		}

		pSrcImage += srcImageStrideInBytes;
		height--;
	}
	
	sum = _mm_hadd_epi32(sum, sum);											// Lowest int of sum has sum of last two ints of sum
	sum = _mm_hadd_epi32(sum, sum);											// Lowest int of sum has the sum of all four ints
	pixels = _mm_srli_si128(sum_squared, 8);
	sum_squared = _mm_add_epi64(sum_squared, pixels);

	*pSum = (vx_float32)(M128I(sum).m128i_u32[0] + prefixSum + postfixSum);
	*pSumOfSquared = (vx_float32)(M128I(sum_squared).m128i_u64[0] + prefixSumSquared + postfixSumSquared);

	return AGO_SUCCESS;
}

int HafCpu_MeanStdDevMerge_DATA_DATA
	(
		vx_float32  * mean,
		vx_float32  * stddev,
		vx_uint32	  totalSampleCount,
		vx_uint32     numPartitions,
		vx_float32    partSum[],
		vx_float32    partSumOfSquared[]
	)
{
	vx_float32 lmean = 0, lstd = 0;

	for (unsigned int i = 0; i < numPartitions; i++)
	{
		lmean += partSum[i];
		lstd += partSumOfSquared[i];
	}

	lmean /= totalSampleCount;
	lstd = sqrtf((lstd / totalSampleCount) - (lmean * lmean));

	*mean = lmean;
	*stddev = lstd;

	return AGO_SUCCESS;
}

int HafCpu_IntegralImage_U32_U8
(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint32   * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes
)
{
	__m128i pixels1, pixels2, pixels3, pixels4;
	__m128i zeromask = _mm_setzero_si128();
	// process 16 at a time (shift and add for cur and previous)
	unsigned char *pSrcImage1 = pSrcImage;
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	
	while (pchDst < pchDstlast)
	{
		__m128i * src = (__m128i*)pSrcImage1;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstlast = dst + (dstWidth >> 2);
		__m128i prevsum = _mm_setzero_si128();
		if (pSrcImage1 == pSrcImage){
			while (dst < dstlast)
			{
				pixels1 = _mm_loadu_si128(src++);		// src (0-15)
				pixels2 = _mm_unpackhi_epi8(pixels1, zeromask);
				pixels1 = _mm_cvtepu8_epi16(pixels1);
				// shift and add
				pixels3 = pixels1;
				pixels4 = pixels2;
				for (int i = 0; i < 7; i++)
				{
					pixels3 = _mm_slli_si128(pixels3, 2);
					pixels4 = _mm_slli_si128(pixels4, 2);
					pixels1 = _mm_add_epi16(pixels1, pixels3);
					pixels2 = _mm_add_epi16(pixels2, pixels4);
				}
				// for the second 8 sum, add to the first 8
				pixels3 = _mm_shufflehi_epi16(pixels1, 0xff);
				pixels3 = _mm_shuffle_epi32(pixels3, 0xff);
				pixels2 = _mm_add_epi16(pixels2, pixels3);
				// unpack to dwords and add with prevsum
				pixels3 = _mm_unpackhi_epi16(pixels1, zeromask);
				pixels4 = _mm_unpackhi_epi16(pixels2, zeromask);
				pixels1 = _mm_cvtepu16_epi32(pixels1);
				pixels2 = _mm_cvtepu16_epi32(pixels2);
				pixels1 = _mm_add_epi32(pixels1, prevsum);
				pixels2 = _mm_add_epi32(pixels2, prevsum);
				pixels3 = _mm_add_epi32(pixels3, prevsum);
				pixels4 = _mm_add_epi32(pixels4, prevsum);

				// copy to dst (sum in words)
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
				_mm_store_si128(dst++, pixels2);
				_mm_store_si128(dst++, pixels4);
				prevsum = _mm_shuffle_epi32(pixels4, 0xff);

			}
		}
		else
		{
			unsigned int prev_dword = 0;
			__m128i prevdword = _mm_setzero_si128();
			__m128i prevsum1 = _mm_setzero_si128();
			__m128i * prevdst = (__m128i*)(pchDst - dstImageStrideInBytes);
			while (dst < dstlast)
			{
				__m128i prev1, prev2, prev3, prev4, temp, temp1, temp2, temp3;
				pixels1 = _mm_loadu_si128(src++);		// src (0-15)
				pixels2 = _mm_unpackhi_epi8(pixels1, zeromask);
				pixels1 = _mm_cvtepu8_epi16(pixels1);
				// shift and add
				pixels3 = pixels1;
				pixels4 = pixels2;
				for (int i = 0; i < 7; i++)
				{
					pixels3 = _mm_slli_si128(pixels3, 2);
					pixels4 = _mm_slli_si128(pixels4, 2);
					pixels1 = _mm_add_epi16(pixels1, pixels3);
					pixels2 = _mm_add_epi16(pixels2, pixels4);
				}
				// for the second 8 sum, add to the first 8
				pixels3 = _mm_shufflehi_epi16(pixels1, 0xff);
				pixels3 = _mm_shuffle_epi32(pixels3, 0xff);
				pixels2 = _mm_add_epi16(pixels2, pixels3);
				// unpack to dwords and add with prevsum
				pixels3 = _mm_unpackhi_epi16(pixels1, zeromask);
				pixels4 = _mm_unpackhi_epi16(pixels2, zeromask);
				pixels1 = _mm_cvtepu16_epi32(pixels1);
				pixels2 = _mm_cvtepu16_epi32(pixels2);

				// calculate with prevsum(x) - prevsum(x-1)
				prev1 = _mm_load_si128(prevdst++);

				// subtract sum(x-1, y-1)
				temp = _mm_srli_si128(prev1, 12);
				temp1 = _mm_slli_si128(prev1, 4);
				prev2 = _mm_load_si128(prevdst++);
				temp1 = _mm_or_si128(temp1, prevdword);
				prev1 = _mm_sub_epi32(prev1, temp1);

				prevdword = _mm_srli_si128(prev2, 12);
				temp1 = _mm_slli_si128(prev2, 4);
				prev3 = _mm_load_si128(prevdst++);
				temp1 = _mm_or_si128(temp1, temp);
				prev2 = _mm_sub_epi32(prev2, temp1);

				temp = _mm_srli_si128(prev3, 12);
				temp1 = _mm_slli_si128(prev3, 4);
				prev4 = _mm_load_si128(prevdst++);
				temp1 = _mm_or_si128(temp1, prevdword);
				prev3 = _mm_sub_epi32(prev3, temp1);

				prevdword = _mm_srli_si128(prev4, 12);
				temp1 = _mm_slli_si128(prev4, 4);
				temp1 = _mm_or_si128(temp1, temp);
				prev4 = _mm_sub_epi32(prev4, temp1);
				temp = prev1;
				temp1 = prev2;
				temp2 = prev3;
				temp3 = prev4;

				for (int i = 0; i < 3; i++)
				{
					temp = _mm_slli_si128(temp, 4);
					temp1 = _mm_slli_si128(temp1, 4);
					temp2 = _mm_slli_si128(temp2, 4);
					temp3 = _mm_slli_si128(temp3, 4);
					prev1 = _mm_add_epi32(prev1, temp);
					prev2 = _mm_add_epi32(prev2, temp1);
					prev3 = _mm_add_epi32(prev3, temp2);
					prev4 = _mm_add_epi32(prev4, temp3);
				}
				// for the second 4 sum, add to the first 4
				temp = _mm_shuffle_epi32(prev1, 0xff);
				prev2 = _mm_add_epi32(prev2, temp);
				temp1 = _mm_shuffle_epi32(prev2, 0xff);
				prev3 = _mm_add_epi32(prev3, temp1);
				temp = _mm_shuffle_epi32(prev3, 0xff);
				prev4 = _mm_add_epi32(prev4, temp);

				// add to pixels1 to pixels4
				pixels1 = _mm_add_epi32(pixels1, prev1);
				pixels3 = _mm_add_epi32(pixels3, prev2);
				pixels2 = _mm_add_epi32(pixels2, prev3);
				pixels4 = _mm_add_epi32(pixels4, prev4);
				prevsum1 = _mm_shuffle_epi32(prev4, 0xff);

				pixels1 = _mm_add_epi32(pixels1, prevsum);
				pixels3 = _mm_add_epi32(pixels3, prevsum);
				pixels2 = _mm_add_epi32(pixels2, prevsum);
				pixels4 = _mm_add_epi32(pixels4, prevsum);
				// copy to dst (sum in words)
				_mm_store_si128(dst++, pixels1);
				_mm_store_si128(dst++, pixels3);
				_mm_store_si128(dst++, pixels2);
				_mm_store_si128(dst++, pixels4);
				prevsum = _mm_shuffle_epi32(pixels4, 0xff);
			}
		}
		pSrcImage1 += srcImageStrideInBytes;
		pchDst += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

#if 0
// keeping the implementation in case we need it in future
int HafCpu_Histogram_DATA_U8
(
	vx_uint32     dstHist[],
	vx_uint32     srcWidth,
	vx_uint32     srcHeight,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_uint32     numBins,
	vx_uint32     range,
	vx_uint32     offset,
	vx_uint32     window_size
)
{
	__m128i pixels1, pixels2;
	__m128i * src = (__m128i*)pSrcImage;
	__m128i * dst = (__m128i*)pDstImage;
	// clear histogram bins
	unsigned int *pdst = dstHist;
	memset(pdst, 0x0, numBins * sizeof(unsigned int));
	if (!offset)
	{
		if (range == 0xff)
		{
			for (int y = 0; y < srcHeight; y++)
			{
				for (int x = 0; x < srcWidth; x += 16)
				{
					pixels1 = _mm_load_si128(&src[x >> 4]);
					if (window_size > 1)
					{
						__m128 win, pel0, pel1, pel2, pel3;
						// read window size
						win = _mm_set1_ps((float)window_size);
						pixels2 = _mm_cvtepu8_epi32(pixels1);
						pel0 = _mm_cvtepi32_ps(pixels2);
						pel0 = _mm_div_ps(pel0, win);		// divide by window size
						_mm_srli_si128(pixels1, 4);
						pixels2 = _mm_cvtepu8_epi32(pixels1);
						pel1 = _mm_cvtepi32_ps(pixels2);
						pel1 = _mm_div_ps(pel1, win);		// divide by window size
						_mm_srli_si128(pixels1, 4);
						pixels2 = _mm_cvtepu8_epi32(pixels1);
						pel2 = _mm_cvtepi32_ps(pixels2);
						pel2 = _mm_div_ps(pel2, win);		// divide by window size
						_mm_srli_si128(pixels1, 4);
						pixels2 = _mm_cvtepu8_epi32(pixels1);
						pel3 = _mm_cvtepi32_ps(pixels2);
						pel3 = _mm_div_ps(pel3, win);		// divide by window size

						// convert to int and store
						pixels1 = _mm_cvtps_epi32(pel0);
						pixels2 = _mm_cvtps_epi32(pel1);
						_mm_store_si128(&dst[(x >> 2)], pixels1);
						_mm_store_si128(&dst[(x >> 2) + 1], pixels2);
						pixels1 = _mm_cvtps_epi32(pel2);
						pixels2 = _mm_cvtps_epi32(pel3);
						_mm_store_si128(&dst[(x >> 2) + 2], pixels1);
						_mm_store_si128(&dst[(x >> 2) + 3], pixels2);
					}

				}
			}
		}
	}

	return AGO_SUCCESS;
}
#endif

#define NUM_BINS	256
// special case histogram primitive : range - 255, offset: 0, NumBins: 255
int HafCpu_Histogram_DATA_U8
(
	vx_uint32     dstHist[],
	vx_uint32     srcWidth,
	vx_uint32     srcHeight,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes
)
{
	unsigned int *pdst = dstHist;
	memset(pdst, 0x0, NUM_BINS * sizeof(unsigned int));
	for (unsigned int y = 0; y < srcHeight; y++)
	{
		unsigned int * src = (unsigned int *)(pSrcImage + y*srcImageStrideInBytes);
		unsigned int * srclast = src + (srcWidth >> 2);
		while (src < srclast)
		{
			// do for 16 pixels..
			unsigned int pixel4;
			pixel4 = *src++;
			pdst[(pixel4 & 0xFF)]++;
			pdst[(pixel4 >> 8) & 0xFF]++;
			pdst[(pixel4 >> 16) & 0xFF]++;
			pdst[(pixel4 >> 24) & 0xFF]++;

			pixel4 = *src++;
			pdst[(pixel4 & 0xFF)]++;
			pdst[(pixel4 >> 8) & 0xFF]++;
			pdst[(pixel4 >> 16) & 0xFF]++;
			pdst[(pixel4 >> 24) & 0xFF]++;

			pixel4 = *src++;
			pdst[(pixel4 & 0xFF)]++;
			pdst[(pixel4 >> 8) & 0xFF]++;
			pdst[(pixel4 >> 16) & 0xFF]++;
			pdst[(pixel4 >> 24) & 0xFF]++;

			pixel4 = *src++;
			pdst[(pixel4 & 0xFF)]++;
			pdst[(pixel4 >> 8) & 0xFF]++;
			pdst[(pixel4 >> 16) & 0xFF]++;
			pdst[(pixel4 >> 24) & 0xFF]++;
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_HistogramMerge_DATA_DATA
(
	vx_uint32     dstHist[],
	vx_uint32     numPartitions,
	vx_uint32   * pPartSrcHist[]
)
{
	__m128i pixels1, pixels2;
	__m128i * dst = (__m128i*)dstHist;

	for (unsigned int n = 0; n < 256; n+=8)
	{
		__m128i sum1 = _mm_setzero_si128();
		__m128i sum2 = _mm_setzero_si128();
		for (unsigned int i = 0; i < numPartitions; i++){
			__m128i *phist = (__m128i *)&pPartSrcHist[i];
			pixels1 = _mm_load_si128(&phist[(n >> 2)]);
			pixels2 = _mm_load_si128(&phist[(n >> 2)+1]);
			sum1 = _mm_add_epi32(sum1, pixels1);
			sum2 = _mm_add_epi32(sum2, pixels2);
		}
		// copy merged
		_mm_store_si128(&dst[(n >> 2)], sum1);
		_mm_store_si128(&dst[(n >> 2) + 1], sum2);
	}
	return AGO_SUCCESS;
}

// Primitive: Histogram equalization
// first do a merge of individual histograms before doing equalization
int HafCpu_Equalize_DATA_DATA
(
vx_uint8    * pLut,
vx_uint32     numPartitions,
vx_uint32   * pPartSrcHist[]
)
{
	unsigned int cdfmin = 0, div;
	__m128i pixels1, pixels2, pixels4;
	__m128i dst_[NUM_BINS / 4], * dst = dst_;
	unsigned int * cdf = M128I(dst_[0]).m128i_u32;

	pixels4 = _mm_setzero_si128();

	for (unsigned int n = 0; n < NUM_BINS; n += 8)
	{
		__m128i sum1 = _mm_setzero_si128();
		__m128i sum2 = _mm_setzero_si128();
		for (unsigned int i = 0; i < numPartitions; i++){
			__m128i *phist = (__m128i *)&pPartSrcHist[i][n];
			pixels1 = _mm_load_si128(phist);
			pixels2 = _mm_load_si128(phist+1);
			sum1 = _mm_add_epi32(sum1, pixels1);
			sum2 = _mm_add_epi32(sum2, pixels2);
		}
		// calculate cdf
		// shift and add
		pixels1 = sum1;
		pixels2 = sum2;
		for (int i = 0; i < 4; i++)
		{
			pixels1 = _mm_slli_si128(pixels1, 4);
			pixels2 = _mm_slli_si128(pixels2, 4);
			sum1 = _mm_add_epi32(sum1, pixels1);
			sum2 = _mm_add_epi32(sum2, pixels2);
		}
		// for the second sum onwards, add to the first 
		pixels1 = _mm_shuffle_epi32(sum1, 0xff);
		sum1 = _mm_add_epi32(sum1, pixels4);
		sum2 = _mm_add_epi32(sum2, pixels4);
		sum2 = _mm_add_epi32(sum2, pixels1);
		pixels4 = _mm_shuffle_epi32(sum2, 0xff);

		// store cdf
		_mm_store_si128(dst++, sum1);
		_mm_store_si128(dst++, sum2);
	}
	// find the cdf[minv]
	for (int n = 0; n < NUM_BINS; n++){
		pLut[n] = 0;		// initialize
		if (cdf[n] || cdfmin){
			if (!cdfmin){
				cdfmin = cdf[n];
				div = cdf[NUM_BINS - 1] - cdfmin;		// range
			}
			// equalize to 0-255
			if (div){
				float p = (float)(cdf[n] - cdfmin) / (float)div;
				pLut[n] = (vx_uint8)(p*255.0f + 0.5f);
			}
			else
				pLut[n] = n;		// is this correct?
		}
	}

	return AGO_SUCCESS;
}

int HafCpu_MinMax_DATA_U8
	(
		vx_int32    * pDstMinValue,
		vx_int32    * pDstMaxValue,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * pLocalSrc_xmm;
	__m128i pixels;
	__m128i maxVal_xmm = _mm_setzero_si128();
	__m128i minVal_xmm = _mm_set1_epi8((char) 0xFF);

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - prefixWidth - postfixWidth;
	unsigned char maxVal = 0, minVal = 255;
	unsigned char * pLocalSrc;

	int height = (int)srcHeight;
	while (height)
	{
		pLocalSrc = (unsigned char *)pSrcImage;

		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			maxVal = max(maxVal, *pLocalSrc);
			minVal = min(minVal, *pLocalSrc);
		}

		pLocalSrc_xmm = (__m128i *) pLocalSrc;
		int width = (int)(alignedWidth >> 4);									// 16 pixels processed at a time
		while (width)
		{
			pixels = _mm_load_si128(pLocalSrc_xmm++);
			maxVal_xmm = _mm_max_epu8(maxVal_xmm, pixels);
			minVal_xmm = _mm_min_epu8(minVal_xmm, pixels);

			width--;
		}

		pLocalSrc = (unsigned char *)pLocalSrc_xmm;
		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			maxVal = max(maxVal, *pLocalSrc);
			minVal = min(minVal, *pLocalSrc);
		}

		pSrcImage += srcImageStrideInBytes;
		height--;
	}

	// Compute the max value out of the max at 16 individual places
	for (int i = 0; i < 16; i++)
	{
		maxVal = max(maxVal, M128I(maxVal_xmm).m128i_u8[i]);
		minVal = min(minVal, M128I(minVal_xmm).m128i_u8[i]);
	}

	*pDstMinValue = (vx_int32) minVal;
	*pDstMaxValue = (vx_int32) maxVal;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min values in the source image
	__m128i minVal = _mm_set1_epi8((unsigned char)globalMin);
	__m128i pixels;
	int minCount = 0;
	unsigned char * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask;

			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			pixels = _mm_cmpeq_epi8(pixels, minVal);
			minMask = _mm_movemask_epi8(pixels);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
						minCount++;
					minMask >>= 1;
				}
			}
			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			width++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
	}

	*pMinLocCount = (vx_int32)minCount;
	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min values in the source image
	__m128i maxVal = _mm_set1_epi8((unsigned char)globalMax);
	__m128i pixels;
	int maxCount = 0;
	unsigned char * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int maxMask;

			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			pixels = _mm_cmpeq_epi8(pixels, maxVal);
			maxMask = _mm_movemask_epi8(pixels);

			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
						maxCount++;
					maxMask >>= 1;
				}
			}
			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
	}

	*pMaxLocCount = (vx_int32)maxCount;
	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi8((unsigned char)globalMin);
	__m128i maxVal = _mm_set1_epi8((unsigned char)globalMax);
	__m128i pixels;
	int minCount = 0, maxCount = 0;
	unsigned char * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			if (*pLocalSrc == globalMax)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask, maxMask;

			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			__m128i temp = _mm_cmpeq_epi8(pixels, minVal);
			minMask = _mm_movemask_epi8(temp);

			temp = _mm_cmpeq_epi8(pixels, maxVal);
			maxMask = _mm_movemask_epi8(temp);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
						minCount++;
					minMask >>= 1;
				}
			}
			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
						maxCount++;
					maxMask >>= 1;
				}
			}
			
			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			if (*pLocalSrc == globalMax)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
	}

	*pMinLocCount = (vx_int32)minCount;
	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi8((unsigned char)globalMin);
	__m128i pixels;
	int minCount = 0;
	unsigned char * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool minListNotFull = (minCount < (int)capacityOfMinLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask;

			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			pixels = _mm_cmpeq_epi8(pixels, minVal);
			minMask = _mm_movemask_epi8(pixels);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
					{
						if (minListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							minLocList[minCount] = loc;
						}
						minCount++;
						minListNotFull = (minCount < (int)capacityOfMinLocList);
					}
					minMask >>= 1;
				}
			}
			
			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			width++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
	}

	*pMinLocCount = (vx_int32)minCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi8((unsigned char)globalMin);
	__m128i maxVal = _mm_set1_epi8((unsigned char)globalMax);
	__m128i pixels;
	int minCount = 0, maxCount = 0;
	unsigned char * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool minListNotFull = (minCount < (int)capacityOfMinLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			if (*pLocalSrc == globalMax)
				maxCount++;

			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask, maxMask;

			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			__m128i temp = _mm_cmpeq_epi8(pixels, minVal);
			minMask = _mm_movemask_epi8(temp);

			temp = _mm_cmpeq_epi8(pixels, maxVal);
			maxMask = _mm_movemask_epi8(temp);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
					{
						if (minListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							minLocList[minCount] = loc;
						}
						minCount++;
						minListNotFull = (minCount < (int)capacityOfMinLocList);
					}
					minMask >>= 1;
				}
			}
			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
						maxCount++;
					maxMask >>= 1;
				}
			}

			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			if (*pLocalSrc == globalMax)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
	}

	*pMinLocCount = (vx_int32)minCount;
	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i maxVal = _mm_set1_epi8((unsigned char)globalMax);
	__m128i pixels;
	int maxCount = 0;
	unsigned char * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}

			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int maxMask;

			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			
			pixels = _mm_cmpeq_epi8(pixels, maxVal);
			maxMask = _mm_movemask_epi8(pixels);

			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
					{
						if (maxListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							maxLocList[maxCount] = loc;
						}
						maxCount++;
						maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
					}
					maxMask >>= 1;
				}
			}

			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}
			width++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
	}

	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi8((unsigned char)globalMin);
	__m128i maxVal = _mm_set1_epi8((unsigned char)globalMax);
	__m128i pixels;
	int minCount = 0, maxCount = 0;
	unsigned char * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}

			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask, maxMask;

			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			__m128i temp = _mm_cmpeq_epi8(pixels, minVal);
			minMask = _mm_movemask_epi8(temp);

			temp = _mm_cmpeq_epi8(pixels, maxVal);
			maxMask = _mm_movemask_epi8(temp);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
						minCount++;
					minMask >>= 1;
				}
			}
			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
					{
						if (maxListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							maxLocList[maxCount] = loc;
						}
						maxCount++;
						maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
					}
					maxMask >>= 1;
				}
			}

			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}
			width++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
	}

	*pMinLocCount = (vx_int32)minCount;
	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi8((unsigned char) globalMin);
	__m128i maxVal = _mm_set1_epi8((unsigned char) globalMax);
	__m128i pixels;
	int minCount = 0, maxCount = 0;
	unsigned char * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool minListNotFull = (minCount < (int) capacityOfMinLocList);
	bool maxListNotFull = (maxCount < (int) capacityOfMaxLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (unsigned char *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}
							
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask, maxMask;
			
			pixels = _mm_load_si128((__m128i *) pLocalSrc);
			__m128i temp = _mm_cmpeq_epi8(pixels, minVal);
			minMask = _mm_movemask_epi8(temp);

			temp = _mm_cmpeq_epi8(pixels, maxVal);
			maxMask = _mm_movemask_epi8(temp);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
					{
						if (minListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							minLocList[minCount] = loc;
						}
						minCount++;
						minListNotFull = (minCount < (int)capacityOfMinLocList);
					}
					minMask >>= 1;
				}
			}
			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
					{
						if (maxListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							maxLocList[maxCount] = loc;
						}
						maxCount++;
						maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
					}
					maxMask >>= 1;
				}
			}
			
			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int) srcWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}
			width++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;		
	}

	*pMinLocCount = (vx_int32)minCount;
	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

int HafCpu_MinMax_DATA_S16
	(
		vx_int32    * pDstMinValue,
		vx_int32    * pDstMaxValue,
		vx_uint32     srcWidth,
		vx_uint32     srcHeight,
		vx_int16    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * pLocalSrc_xmm;
	__m128i pixels;
	
	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - prefixWidth - postfixWidth;
	short maxVal = SHRT_MIN, minVal = SHRT_MAX;
	short * pLocalSrc;

	__m128i maxVal_xmm = _mm_set1_epi16(maxVal);
	__m128i minVal_xmm = _mm_set1_epi16(minVal);

	int height = (int)srcHeight;
	while (height)
	{
		pLocalSrc = (short *)pSrcImage;
		for (int x = 0; x < prefixWidth; x++, pLocalSrc++)
		{
			maxVal = max(maxVal, *pLocalSrc);
			minVal = min(minVal, *pLocalSrc);
		}

		pLocalSrc_xmm = (__m128i *) pLocalSrc;
		int width = (int)(alignedWidth >> 3);									// 8 pixels processed at a time
		while (width)
		{
			pixels = _mm_load_si128(pLocalSrc_xmm++);
			maxVal_xmm = _mm_max_epi16(maxVal_xmm, pixels);
			minVal_xmm = _mm_min_epi16(minVal_xmm, pixels);

			width--;
		}

		pLocalSrc = (short *)pLocalSrc_xmm;
		for (int x = 0; x < postfixWidth; x++, pLocalSrc++)
		{
			maxVal = max(maxVal, *pLocalSrc);
			minVal = min(minVal, *pLocalSrc);
		}
		
		pSrcImage += (srcImageStrideInBytes >> 1);
		height--;
	}

	// Compute the max value out of the max at 16 individual places
	for (int i = 0; i < 8; i++)
	{
		maxVal = max(maxVal, M128I(maxVal_xmm).m128i_i16[i]);
		minVal = min(minVal, M128I(minVal_xmm).m128i_i16[i]);
	}

	*pDstMinValue = (vx_int32) minVal;
	*pDstMaxValue = (vx_int32) maxVal;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min values in the source image
	__m128i minVal = _mm_set1_epi16((short)globalMin);
	__m128i pixelsH, pixelsL;
	int minCount = 0;
	short * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask;

			pixelsL = _mm_load_si128((__m128i *) pLocalSrc);
			pixelsH = _mm_load_si128((__m128i *) (pLocalSrc + 8));

			pixelsH = _mm_cmpeq_epi16(pixelsH, minVal);
			pixelsL = _mm_cmpeq_epi16(pixelsL, minVal);
			pixelsL = _mm_packs_epi16(pixelsL, pixelsH);
			minMask = _mm_movemask_epi8(pixelsL);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
						minCount++;
					minMask >>= 1;
				}
			}
			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			width++;
			pLocalSrc++;
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
	}

	*pMinLocCount = (vx_int32)minCount;
	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min values in the source image
	__m128i maxVal = _mm_set1_epi16((short)globalMax);
	__m128i pixelsH, pixelsL;
	int maxCount = 0;
	short * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int maxMask;

			pixelsL = _mm_load_si128((__m128i *) pLocalSrc);
			pixelsH = _mm_load_si128((__m128i *) (pLocalSrc + 8));
			
			pixelsH = _mm_cmpeq_epi16(pixelsH, maxVal);
			pixelsL = _mm_cmpeq_epi16(pixelsL, maxVal);
			pixelsL = _mm_packs_epi16(pixelsL, pixelsH);
			maxMask = _mm_movemask_epi8(pixelsL);

			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
						maxCount++;
					maxMask >>= 1;
				}
			}
			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
	}

	*pMaxLocCount = (vx_int32)maxCount;
	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi16((short)globalMin);
	__m128i maxVal = _mm_set1_epi16((short)globalMax);
	__m128i pixelsL, pixelsH;
	int minCount = 0, maxCount = 0;
	short * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			if (*pLocalSrc == globalMax)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask, maxMask;

			pixelsL = _mm_load_si128((__m128i *) pLocalSrc);
			pixelsH = _mm_load_si128((__m128i *) (pLocalSrc + 8));
			
			__m128i temp1 = _mm_cmpeq_epi16(pixelsH, minVal);
			__m128i temp0 = _mm_cmpeq_epi16(pixelsL, minVal);
			temp0 = _mm_packs_epi16(temp0, temp1);
			minMask = _mm_movemask_epi8(temp0);

			pixelsH = _mm_cmpeq_epi16(pixelsH, maxVal);
			pixelsL = _mm_cmpeq_epi16(pixelsL, maxVal);
			temp1 = _mm_packs_epi16(pixelsL, pixelsH);
			maxMask = _mm_movemask_epi8(temp1);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
						minCount++;
					minMask >>= 1;
				}
			}
			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
						maxCount++;
					maxMask >>= 1;
				}
			}

			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			if (*pLocalSrc == globalMax)
				maxCount++;
			width++;
			pLocalSrc++;
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
	}

	*pMinLocCount = (vx_int32)minCount;
	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi16((short)globalMin);
	__m128i pixelsL, pixelsH;
	int minCount = 0;
	short * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool minListNotFull = (minCount < (int)capacityOfMinLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask;

			pixelsL = _mm_load_si128((__m128i *) pLocalSrc);
			pixelsH = _mm_load_si128((__m128i *) (pLocalSrc + 8));

			pixelsH = _mm_cmpeq_epi16(pixelsH, minVal);
			pixelsL = _mm_cmpeq_epi16(pixelsL, minVal);
			pixelsL = _mm_packs_epi16(pixelsL, pixelsH);
			minMask = _mm_movemask_epi8(pixelsL);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
					{
						if (minListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							minLocList[minCount] = loc;
						}
						minCount++;
						minListNotFull = (minCount < (int)capacityOfMinLocList);
					}
					minMask >>= 1;
				}
			}
			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			width++;
			pLocalSrc++;
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
	}

	*pMinLocCount = (vx_int32)minCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi16((short)globalMin);
	__m128i maxVal = _mm_set1_epi16((short)globalMax);
	__m128i pixelsL, pixelsH;
	int minCount = 0, maxCount = 0;
	short * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool minListNotFull = (minCount < (int)capacityOfMinLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			if (*pLocalSrc == globalMax)
				maxCount++;

			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask, maxMask;

			pixelsL = _mm_load_si128((__m128i *) pLocalSrc);
			pixelsH = _mm_load_si128((__m128i *) (pLocalSrc + 8));

			__m128i temp1 = _mm_cmpeq_epi16(pixelsH, minVal);
			__m128i temp0 = _mm_cmpeq_epi16(pixelsL, minVal);
			temp0 = _mm_packs_epi16(temp0, temp1);
			minMask = _mm_movemask_epi8(temp0);

			pixelsH = _mm_cmpeq_epi16(pixelsH, maxVal);
			pixelsL = _mm_cmpeq_epi16(pixelsL, maxVal);
			temp1 = _mm_packs_epi16(pixelsL, pixelsH);
			maxMask = _mm_movemask_epi8(temp1);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
					{
						if (minListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							minLocList[minCount] = loc;
						}
						minCount++;
						minListNotFull = (minCount < (int)capacityOfMinLocList);
					}
					minMask >>= 1;
				}
			}
			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
						maxCount++;
					maxMask >>= 1;
				}
			}

			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			if (*pLocalSrc == globalMax)
				maxCount++;
			
			width++;
			pLocalSrc++;
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
	}

	*pMinLocCount = (vx_int32)minCount;
	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i maxVal = _mm_set1_epi16((short)globalMax);
	__m128i pixelsL, pixelsH;
	int minCount = 0, maxCount = 0;
	short * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}

			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int maxMask;

			pixelsL = _mm_load_si128((__m128i *) pLocalSrc);
			pixelsH = _mm_load_si128((__m128i *) (pLocalSrc + 8));

			pixelsH = _mm_cmpeq_epi16(pixelsH, maxVal);
			pixelsL = _mm_cmpeq_epi16(pixelsL, maxVal);
			pixelsL = _mm_packs_epi16(pixelsL, pixelsH);
			maxMask = _mm_movemask_epi8(pixelsL);

			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
					{
						if (maxListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							maxLocList[maxCount] = loc;
						}
						maxCount++;
						maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
					}
					maxMask >>= 1;
				}
			}

			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}
			width++;
			pLocalSrc++;
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
	}

	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi16((short)globalMin);
	__m128i maxVal = _mm_set1_epi16((short)globalMax);
	__m128i pixelsL, pixelsH;
	int minCount = 0, maxCount = 0;
	short * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}

			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask, maxMask;

			pixelsL = _mm_load_si128((__m128i *) pLocalSrc);
			pixelsH = _mm_load_si128((__m128i *) (pLocalSrc + 8));

			__m128i temp1 = _mm_cmpeq_epi16(pixelsH, minVal);
			__m128i temp0 = _mm_cmpeq_epi16(pixelsL, minVal);
			temp0 = _mm_packs_epi16(temp0, temp1);
			minMask = _mm_movemask_epi8(temp0);

			pixelsH = _mm_cmpeq_epi16(pixelsH, maxVal);
			pixelsL = _mm_cmpeq_epi16(pixelsL, maxVal);
			temp1 = _mm_packs_epi16(pixelsL, pixelsH);
			maxMask = _mm_movemask_epi8(temp1);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
						minCount++;
					minMask >>= 1;
				}
			}
			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
					{
						if (maxListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							maxLocList[maxCount] = loc;
						}
						maxCount++;
						maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
					}
					maxMask >>= 1;
				}
			}

			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
				minCount++;
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}
			width++;
			pLocalSrc++;
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
	}

	*pMinLocCount = (vx_int32)minCount;
	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

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
	)
{
	// Compute the global minima and maxima
	vx_int32 globalMin, globalMax;
	HafCpu_MinMaxMerge_DATA_DATA(&globalMin, &globalMax, numDataPartitions, srcMinValue, srcMaxValue);

	*pDstMinValue = globalMin;
	*pDstMaxValue = globalMax;

	// Search for the min and the max values in the source image
	__m128i minVal = _mm_set1_epi16((short)globalMin);
	__m128i maxVal = _mm_set1_epi16((short)globalMax);
	__m128i pixelsL, pixelsH;
	int minCount = 0, maxCount = 0;
	short * pLocalSrc;

	int prefixWidth = intptr_t(pSrcImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	prefixWidth >>= 1;														// 2 bytes = 1 pixel
	int postfixWidth = ((int)srcWidth - prefixWidth) & 15;
	int alignedWidth = (int)srcWidth - postfixWidth;

	bool minListNotFull = (minCount < (int)capacityOfMinLocList);
	bool maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
	vx_coordinates2d_t loc;

	for (int height = 0; height < (int)srcHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		int width = 0;
		while (width < prefixWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}

			width++;
			pLocalSrc++;
		}

		while (width < alignedWidth)
		{
			int minMask, maxMask;

			pixelsL = _mm_load_si128((__m128i *) pLocalSrc);
			pixelsH = _mm_load_si128((__m128i *) (pLocalSrc + 8));
			
			__m128i temp1 = _mm_cmpeq_epi16(pixelsH, minVal);
			__m128i temp0 = _mm_cmpeq_epi16(pixelsL, minVal);
			temp0 = _mm_packs_epi16(temp0, temp1);
			minMask = _mm_movemask_epi8(temp0);

			pixelsH = _mm_cmpeq_epi16(pixelsH, maxVal);
			pixelsL = _mm_cmpeq_epi16(pixelsL, maxVal);
			temp1 = _mm_packs_epi16(pixelsL, pixelsH);
			maxMask = _mm_movemask_epi8(temp1);

			if (minMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (minMask & 1)
					{
						if (minListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							minLocList[minCount] = loc;
						}
						minCount++;
						minListNotFull = (minCount < (int)capacityOfMinLocList);
					}
					minMask >>= 1;
				}
			}
			if (maxMask)
			{
				for (int i = 0; i < 16; i++)
				{
					if (maxMask & 1)
					{
						if (maxListNotFull)
						{
							loc.y = height;
							loc.x = width + i;
							maxLocList[maxCount] = loc;
						}
						maxCount++;
						maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
					}
					maxMask >>= 1;
				}
			}

			width += 16;
			pLocalSrc += 16;
		}

		while (width < (int)srcWidth)
		{
			if (*pLocalSrc == globalMin)
			{
				if (minListNotFull)
				{
					loc.x = width;
					loc.y = height;
					minLocList[minCount] = loc;
				}
				minCount++;
				minListNotFull = (minCount < (int)capacityOfMinLocList);
			}
			if (*pLocalSrc == globalMax)
			{
				if (maxListNotFull)
				{
					loc.x = width;
					loc.y = height;
					maxLocList[maxCount] = loc;
				}
				maxCount++;
				maxListNotFull = (maxCount < (int)capacityOfMaxLocList);
			}
			width++;
			pLocalSrc++;
		}

		pSrcImage += (srcImageStrideInBytes >> 1);
	}

	*pMinLocCount = (vx_int32)minCount;
	*pMaxLocCount = (vx_int32)maxCount;

	return AGO_SUCCESS;
}

int HafCpu_MinMaxMerge_DATA_DATA
	(
		vx_int32    * pDstMinValue,
		vx_int32    * pDstMaxValue,
		vx_uint32     numDataPartitions,
		vx_int32      srcMinValue[],
		vx_int32      srcMaxValue[]
	)
{
	vx_int32 minVal, maxVal;

	minVal = srcMinValue[0];
	maxVal = srcMaxValue[0];

	for (int i = 1; i < (int) numDataPartitions; i++)
	{
		minVal = min(minVal, srcMinValue[i]);
		maxVal = min(minVal, srcMaxValue[i]);
	}

	*pDstMinValue = minVal;
	*pDstMaxValue = maxVal;

	return AGO_SUCCESS;
}


int HafCpu_MinMaxLocMerge_DATA_DATA
	(
		vx_uint32          * pDstLocCount,
		vx_uint32            capacityOfDstLocList,
		vx_coordinates2d_t   dstLocList[],
		vx_uint32            numDataPartitions,
		vx_uint32            partLocCount[],
		vx_coordinates2d_t * partLocList[]
	)
{
	int dstCount = 0;
	int srcCount;
	vx_coordinates2d_t * srcList;
	
	for (int i = 0; i < (int)numDataPartitions; i++)
	{
		srcList = partLocList[i];
		srcCount = partLocCount[i];

		while (srcCount)
		{
			*dstLocList++ = *srcList++;
			dstCount++;
			srcCount--;
			if (dstCount > (int) capacityOfDstLocList)
			{
				*pDstLocCount = (vx_uint32)(dstCount - 1);
				return AGO_SUCCESS;
			}
		}
	}
	return AGO_SUCCESS;
}

float HafCpu_FastAtan2_rad
(
	vx_int16	  Gx,
	vx_int16      Gy
)
{
	vx_uint16 ax, ay;
	ax = std::abs(Gx), ay = std::abs(Gy);
	float a, c, c2;
	if (ax >= ay)
	{
		c = (float)ay / ((float)ax + (float)DBL_EPSILON);
		c2 = c*c;
		a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	else
	{
		c = (float)ax / ((float)ay + (float)DBL_EPSILON);
		c2 = c*c;
		a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	if (Gx < 0)
		a = 180.f - a;
	if (Gy < 0)
		a = 360.f - a;
	return (a*(PI/180));
}

float HafCpu_FastAtan2_deg
(
vx_int16	  Gx,
vx_int16      Gy
)
{
	vx_uint16 ax, ay;
	ax = std::abs(Gx), ay = std::abs(Gy);
	float a, c, c2;
	if (ax >= ay)
	{
		c = (float)ay / ((float)ax + (float)DBL_EPSILON);
		c2 = c*c;
		a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	else
	{
		c = (float)ax / ((float)ay + (float)DBL_EPSILON);
		c2 = c*c;
		a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	if (Gx < 0)
		a = 180.f - a;
	if (Gy < 0)
		a = 360.f - a;
	return a;
}

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
)
{
	unsigned int y = 0;
	// do the plain vanilla version with atan2
	while (y < dstHeight)
	{
		vx_uint8 *pdst = pPhaseImage;
		vx_int16 *pGx = pGxImage;
		vx_int16 *pGy = pGyImage;

		for (unsigned int x = 0; x < dstWidth; x++)
		{
#if 0
			float arct = atan2((float)pGy[x], (float)pGx[x]);
			if (arct < 0.f)
			{
				arct += TWOPI;
			}
			// normalize and copy to dst
			*pdst++ = (vx_uint8)((vx_uint32)((float)(arct / PI) * 128 + 0.5) & 0xFF);
#else
			float scale = (float)128 / 180.f;
			float arct = HafCpu_FastAtan2_deg(pGx[x], pGy[x]);
			// normalize and copy to dst
			*pdst++ = (vx_uint8)((vx_uint32)(arct*scale + 0.5) & 0xFF);
#endif
		}
		pPhaseImage += phaseImageStrideInBytes;
		pGxImage += (gxImageStrideInBytes>>1);
		pGyImage += (gyImageStrideInBytes>>1);
		y++;
	}
	return AGO_SUCCESS;
}


int HafCpu_FastAtan2_Canny
(
vx_int16	  Gx,
vx_int16      Gy
)
{
	unsigned int ret;
	vx_uint16 ax, ay;
	ax = std::abs(Gx), ay = std::abs(Gy);	// todo:: check if math.h function is faster
	float d1 = (float)ax*0.4142135623730950488016887242097f;
	float d2 = (float)ax*2.4142135623730950488016887242097f;
	ret = (Gx*Gy) < 0 ? 3 : 1;
	if (ay <= d1)
		ret = 0;
	if (ay >= d2)
		ret = 2;
	return ret;
}
