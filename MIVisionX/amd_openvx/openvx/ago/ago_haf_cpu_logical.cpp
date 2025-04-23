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

vx_uint32 dataConvertU1ToU8_4bytes[16] = { 0x00000000, 0x000000FF, 0x0000FF00, 0x0000FFFF,
									       0x00FF0000, 0x00FF00FF, 0x00FFFF00, 0x00FFFFFF,
									       0xFF000000, 0xFF0000FF, 0xFF00FF00, 0xFF00FFFF,
									       0xFFFF0000, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF };

int HafCpu_Not_U8_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	bool useAligned = ((((intptr_t)pSrcImage | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;			// Check if src and dst buffers are 16 byte aligned

	__m128i *pLocalSrc_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc, *pLocalDst;
	
	int height = (int)dstHeight;
	int alignedWidth = (int)(dstWidth & ~63);
	int postfixWidth = dstWidth - alignedWidth;

	__m128i ones = _mm_setzero_si128();
	ones = _mm_cmpeq_epi32(ones, ones);

	if (useAligned)
	{
		while (height > 0)
		{

			pLocalSrc_xmm = (__m128i*) pSrcImage;
			pLocalDst_xmm = (__m128i*) pDstImage;
			int width = alignedWidth >> 6;

			while (width > 0)
			{
				__m128i pixels0 = _mm_load_si128(pLocalSrc_xmm++);

				__m128i pixels1 = _mm_load_si128(pLocalSrc_xmm++);
				pixels0 = _mm_andnot_si128(pixels0, ones);

				__m128i pixels2 = _mm_load_si128(pLocalSrc_xmm++);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels0);

				__m128i pixels3 = _mm_load_si128(pLocalSrc_xmm++);
				pixels2 = _mm_andnot_si128(pixels2, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels1);

				pixels3 = _mm_andnot_si128(pixels3, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels2);

				_mm_store_si128(pLocalDst_xmm++, pixels3);

				width--;
			}

			width = postfixWidth;
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			while (width > 0)
			{
				*pLocalDst++ = ~(*pLocalSrc++);
				width--;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
			height--;
		}
	}
	else							// Unaligned access
	{
		while (height > 0)
		{

			pLocalSrc_xmm = (__m128i*) pSrcImage;
			pLocalDst_xmm = (__m128i*) pDstImage;
			int width = alignedWidth >> 6;

			while (width > 0)
			{
				__m128i pixels0 = _mm_loadu_si128(pLocalSrc_xmm++);

				__m128i pixels1 = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels0 = _mm_andnot_si128(pixels0, ones);

				__m128i pixels2 = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels0);

				__m128i pixels3 = _mm_loadu_si128(pLocalSrc_xmm++);
				pixels2 = _mm_andnot_si128(pixels2, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);

				pixels3 = _mm_andnot_si128(pixels3, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels2);

				_mm_storeu_si128(pLocalDst_xmm++, pixels3);

				width--;
		}

			width = postfixWidth;
			pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			while (width > 0)
			{
				*pLocalDst++ = ~(*pLocalSrc++);
				width--;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
			height--;
		}
	}
	
#if 0
	__m128i *pLocalSrc, *pLocalDst;

	int height = (int)dstHeight, width = (int)(dstWidth >> 7);

	__m128i ones = _mm_setzero_si128();
	ones = _mm_cmpeq_epi32(ones, ones);

	while (height > 0)
	{
		pLocalSrc = (__m128i*) pSrcImage;
		pLocalDst = (__m128i*) pDstImage;
		while (width > 0)
		{
			__m128i pixels0 = _mm_load_si128(pLocalSrc++);
			__m128i pixels1 = _mm_load_si128(pLocalSrc++);
			__m128i pixels2 = _mm_load_si128(pLocalSrc++);
			__m128i pixels3 = _mm_load_si128(pLocalSrc++);
			__m128i pixels4 = _mm_load_si128(pLocalSrc++);
			__m128i pixels5 = _mm_load_si128(pLocalSrc++);
			__m128i pixels6 = _mm_load_si128(pLocalSrc++);
			__m128i pixels7 = _mm_load_si128(pLocalSrc++);

			pixels0 = _mm_andnot_si128(pixels0, ones);
			pixels1 = _mm_andnot_si128(pixels1, ones);
			pixels2 = _mm_andnot_si128(pixels2, ones);
			pixels3 = _mm_andnot_si128(pixels3, ones);
			pixels4 = _mm_andnot_si128(pixels4, ones);
			pixels5 = _mm_andnot_si128(pixels5, ones);
			pixels6 = _mm_andnot_si128(pixels6, ones);
			pixels7 = _mm_andnot_si128(pixels7, ones);

			_mm_store_si128(pLocalDst++, pixels0);
			_mm_store_si128(pLocalDst++, pixels1);
			_mm_store_si128(pLocalDst++, pixels2);
			_mm_store_si128(pLocalDst++, pixels3);
			_mm_store_si128(pLocalDst++, pixels4);
			_mm_store_si128(pLocalDst++, pixels5);
			_mm_store_si128(pLocalDst++, pixels6);
			_mm_store_si128(pLocalDst++, pixels7);

			width--;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
		width = (int)(dstWidth >> 7);
	}
#endif
#if 0
	_asm {
		mov			ebx, dstHeight
			mov			ecx, pDstImage
			mov			edx, pSrcImage

			pxor		xmm0, xmm0
			pcmpeqd		xmm0, xmm0

		OUTERLOOP:
		mov			eax, dstWidth
		INNERLOOP:
		movdqa		xmm1, [edx]
			movdqa		xmm2, [edx + 10h]
			movdqa		xmm3, [edx + 20h]
			movdqa		xmm4, [edx + 30h]

			pandn		xmm1, xmm0
			pandn		xmm2, xmm0
			pandn		xmm3, xmm0
			pandn		xmm4, xmm0

			movdqa		[ecx], xmm1
			movdqa		[ecx + 10h], xmm2
			movdqa		[ecx + 20h], xmm3
			movdqa		[ecx + 30h], xmm4

			add			edx, 40h
			add			ecx, 40h
			sub			eax, 40h
			jnz			INNERLOOP

			dec			ebx
			jnz			OUTERLOOP
	}
#endif
	return AGO_SUCCESS;
}

#if USE_BMI2
/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
int HafCpu_Not_U8_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	__m128i zeromask = _mm_setzero_si128();

	uint64_t maskConv = 0x0101010101010101;						// Getting LSB out of each byte
	__declspec(align(16)) uint64_t pixels_u64[2];

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels_u64[0] = (uint64_t)(*pSrcImage);
			pixels_u64[1] = (uint64_t)(*(pSrcImage + 8));

			// Convert U1 to U8
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif

			pixels = _mm_load_si128((__m128i*) pixels_u64);
			pixels = _mm_cmpgt_epi8(pixels, zeromask);				// Convert 0x01 to 0xFF
			
			pixels = _mm_andnot_si128(pixels, ones);
			_mm_store_si128(&dst[width >> 4], pixels);
		}
		pSrcImage += srcImageStrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
int HafCpu_Not_U1_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * src = (__m128i*)pSrcImage;
	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels = _mm_load_si128(&src[width >> 4]);
			pixels = _mm_andnot_si128(pixels, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src += (srcImageStrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function processes the pixels in a width which is the next highest multiple of 2 bytes after dstWidth */
int HafCpu_Not_U1_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 2)
		{
			pixels_u64[0] = (uint64_t)(*pSrcImage);
			pixels_u64[1] = (uint64_t)(*(pSrcImage + 8));

			// Convert U1 to U8
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels = _mm_load_si128((__m128i*) pixels_u64);
			pixels = _mm_andnot_si128(pixels, ones);						// Only LSB of each byte counts, because of extract and deposit

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 1)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#else
/* The function processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
int HafCpu_Not_U8_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	vx_int16 inputPixels;
	vx_int16 * pLocalSrc;
	int *pLocalDst;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc = (vx_int16 *)pSrcImage;
		pLocalDst = (int *)pDstImage;
		for (int width = 0; width < alignedWidth; width += 16)
		{
			inputPixels = *pLocalSrc++;
			inputPixels = ~inputPixels;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[inputPixels & 0xF];
			inputPixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[inputPixels & 0xF];
			inputPixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[inputPixels & 0xF];
			inputPixels >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[inputPixels & 0xF];
		}

		if (postfixWidth)
		{
			vx_uint8 pix = *((vx_uint8 *)pLocalSrc);
			pix = ~pix;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pix & 0xF];
			pix >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pix & 0xF];
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels */
int HafCpu_Not_U1_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * pLocalSrc_xmm;
	vx_int16 * pLocalDst_16;

	__m128i pixels;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	int pixelmask;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc_xmm = (__m128i*)pSrcImage;
		pLocalDst_16 = (vx_int16 *)pDstImage;
		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels = _mm_loadu_si128(pLocalSrc_xmm++);
			pixels = _mm_andnot_si128(pixels, ones);
			pixelmask = _mm_movemask_epi8(pixels);

			*pLocalDst_16++ = (vx_int16)(pixelmask & 0xFFFF);
		}

		if (postfixWidth)
		{
			vx_uint8 * pLocalSrc = (vx_uint8 *)pLocalSrc_xmm;
			vx_uint8 * pLocalDst = (vx_uint8 *)pLocalDst_16;

			vx_uint8 temp = 0;
			for (int i = 0; i < 8; i++)
			{
				temp |= (*pLocalSrc++ >> 7) & 1;
				temp <<= 1;
				temp = (temp > 127) ? 255 : temp;
			}
			*pLocalDst = ~temp;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the width is a multiple of 8 pixels */
int HafCpu_Not_U1_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	vx_int16 *pLocalSrc, *pLocalDst;
	vx_int16 pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc = (short *)pSrcImage;
		pLocalDst = (short *)pDstImage;

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels = *pLocalSrc++;
			pixels = ~pixels;
			*pLocalDst++ = pixels;
		}

		if (postfixWidth)
		{
			*((vx_uint8*)pLocalDst) = ~(*((vx_uint8*)pLocalSrc));
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

#endif

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
	)
{
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
				pixels1 = _mm_and_si128(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc1++ & *pLocalSrc2++;
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
				pixels1 = _mm_and_si128(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc1++ & *pLocalSrc2++;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	
	return AGO_SUCCESS;
}

#if USE_BMI2
/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_cmpgt_epi8(pixels2, zeromask);				// Convert 0x01 to 0xFF
			pixels1 = _mm_and_si128(pixels1, pixels2);
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_and_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_cmpgt_epi8(pixels1, zeromask);						// Convert 0x01 to 0xFF
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function assumes that the source image pointers are 16 byte aligned, and the source strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * src2 = (__m128i*)pSrcImage2;
	__m128i pixels1, pixels2;

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);
			pixels2 = _mm_load_si128(&src2[width >> 4]);
			pixels1 = _mm_and_si128(pixels1, pixels2);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		src2 += (srcImage2StrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels1 = _mm_and_si128(pixels1, pixels2);
			
			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_and_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			
			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#else

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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalDst;
	vx_int16 *pLocalSrc2;
	__m128i pixels1, pixels2;
	vx_int16 U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;
	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_and_si128(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				*pLocalDst = (temp & 1) * (vx_uint8)(*pLocalSrc1);
				temp >>= 1;
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
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_and_si128(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				*pLocalDst = (temp & 1) * (vx_uint8)(*pLocalSrc1);
				temp >>= 1;
			}
			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	
	return AGO_SUCCESS;
}

/* The function assumes that the width is a multiple of 8 pixels */
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
	)
{
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int32 * pLocalDst;
	vx_uint8 pixels1, pixels2;
	
	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_uint8 *)pSrcImage1;
		pLocalSrc2 = (vx_uint8 *)pSrcImage2;
		pLocalDst = (vx_int32 *)pDstImage;

		for (int width = 0; width < (int)dstWidth; width += 8)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 & pixels2;

			// U1 to U8
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
			pixels1 >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels*/
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i * pLocalSrc1_xmm, *pLocalSrc2_xmm;
	__m128i pixels1, pixels2;
	int U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_and_si128(pixels1, pixels2);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= ((*pLocalSrc1++ & *pLocalSrc2++) >> 7) & 1;		// the signed bit has the information
					temp <<= 1;
					temp = (temp > 127) ? 255 : temp;				
				}
				*pLocalDst++ = temp;
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
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_and_si128(pixels1, pixels2);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= ((*pLocalSrc1++ & *pLocalSrc2++) >> 7) & 1;
					temp <<= 1;
				}
				*pLocalDst++ = temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	
	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels*/
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
	)
{
	__m128i * pLocalSrc1_xmm;

	__m128i pixels;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1_xmm = (__m128i *) pSrcImage1;
		vx_int16 * pLocalSrc2_16 = (vx_int16 *)pSrcImage2;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels = _mm_loadu_si128(pLocalSrc1_xmm++);
			pixels1 = (vx_int16)(_mm_movemask_epi8(pixels) & 0xFFFF);
			pixels2 = *pLocalSrc2_16++;

			pixels1 = pixels1 & pixels2;
			*pLocalDst_16++ = pixels1;
		}

		if (postfixWidth)
		{
			vx_uint8 * pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			vx_uint8 * pLocalDst = (vx_uint8 *)pLocalDst_16;
			vx_uint8 pix = *((vx_uint8 *)pLocalSrc2_16);
			vx_uint8 temp = 0;
			for (int i = 0; i < 8; i++)
			{
				temp |= ((*pLocalSrc1++) >> 7) & 1;
				temp <<= 1;
				temp = (temp > 127) ? 255 : temp;
			}
			*pLocalDst++ = temp & pix;
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the widths are a multiple of 8 pixels*/
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
	)
{
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_int16 *)pSrcImage1;
		pLocalSrc2 = (vx_int16 *)pSrcImage2;
		pLocalDst = (vx_int16 *)pDstImage;
		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 & pixels2;
			*pLocalDst++ = pixels1;
		}

		if (postfixWidth)
		{
			*((vx_uint8*)pLocalDst) = *((vx_uint8*)pLocalSrc1) & *((vx_uint8*)pLocalSrc2);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#endif

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
	)
{
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
				pixels1 = _mm_or_si128(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc1++ | *pLocalSrc2++;
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
				pixels1 = _mm_or_si128(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc1++ | *pLocalSrc2++;
			}
			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

#if USE_BMI2
/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_cmpgt_epi8(pixels2, zeromask);				// Convert 0x01 to 0xFF
			pixels1 = _mm_or_si128(pixels1, pixels2);
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function assumes that the destination image pointer is 16 byte aligned, and the destination stride as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_or_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_cmpgt_epi8(pixels1, zeromask);						// Convert 0x01 to 0xFF
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function assumes that the source image pointers are 16 byte aligned, and the source strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * src2 = (__m128i*)pSrcImage2;
	__m128i pixels1, pixels2;

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);
			pixels2 = _mm_load_si128(&src2[width >> 4]);
			pixels1 = _mm_or_si128(pixels1, pixels2);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		src2 += (srcImage2StrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels1 = _mm_or_si128(pixels1, pixels2);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_or_si128(pixels1, pixels2);							// Only the LSB here has the AND value

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#else
/* The function assumes that the widths are a multiple of 8 pixels*/
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalDst;
	vx_int16 *pLocalSrc2;
	__m128i pixels1, pixels2;
	vx_int16 U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;
	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_or_si128(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				*pLocalDst = (temp & 1) ? (vx_uint8)255 : (vx_uint8)(*pLocalSrc1);
				temp >>= 1;
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
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_or_si128(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				*pLocalDst = (temp & 1) ? (vx_uint8)255 : (vx_uint8)(*pLocalSrc1);
				temp >>= 1;
			}
			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the width is a multiple of 8 pixels */
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
	)
{
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int32 * pLocalDst;
	vx_uint8 pixels1, pixels2;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_uint8 *)pSrcImage1;
		pLocalSrc2 = (vx_uint8 *)pSrcImage2;
		pLocalDst = (vx_int32 *)pDstImage;

		for (int width = 0; width < (int)dstWidth; width += 8)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 | pixels2;

			// U1 to U8
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
			pixels1 >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels*/
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i * pLocalSrc1_xmm, *pLocalSrc2_xmm;
	__m128i pixels1, pixels2;
	int U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_or_si128(pixels1, pixels2);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= ((*pLocalSrc1++ | *pLocalSrc2++) >> 7) & 1;		// the signed bit has the information
					temp <<= 1;
					temp = (temp > 127) ? 255 : temp;
				}
				*pLocalDst++ = temp;
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
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_or_si128(pixels1, pixels2);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= ((*pLocalSrc1++ | *pLocalSrc2++) >> 7) & 1;
					temp <<= 1;
				}
				*pLocalDst++ = temp;
			}
			
			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels*/
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
	)
{
	__m128i * pLocalSrc1_xmm;

	__m128i pixels;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1_xmm = (__m128i *) pSrcImage1;
		vx_int16 * pLocalSrc2_16 = (vx_int16 *)pSrcImage2;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels = _mm_loadu_si128(pLocalSrc1_xmm++);
			pixels1 = (vx_int16)(_mm_movemask_epi8(pixels) & 0xFFFF);
			pixels2 = *pLocalSrc2_16++;

			pixels1 = pixels1 | pixels2;
			*pLocalDst_16++ = pixels1;
		}

		if (postfixWidth)
		{
			vx_uint8 * pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			vx_uint8 * pLocalDst = (vx_uint8 *)pLocalDst_16;
			vx_uint8 pix = *((vx_uint8 *)pLocalSrc2_16);
			vx_uint8 temp = 0;
			for (int i = 0; i < 8; i++)
			{
				temp |= ((*pLocalSrc1++) >> 7) & 1;
				temp <<= 1;
				temp = (temp > 127) ? 255 : temp;
			}
			*pLocalDst++ = temp | pix;
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the widths are a multiple of 8 pixels */
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
	)
{
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_int16 *)pSrcImage1;
		pLocalSrc2 = (vx_int16 *)pSrcImage2;
		pLocalDst = (vx_int16 *)pDstImage;
		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 | pixels2;
			*pLocalDst++ = pixels1;
		}

		if (postfixWidth)
		{
			*((vx_uint8*)pLocalDst) = *((vx_uint8*)pLocalSrc1) | *((vx_uint8*)pLocalSrc2);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#endif

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
)
{
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
				pixels1 = _mm_xor_si128(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc1++ ^ *pLocalSrc2++;
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
				pixels1 = _mm_xor_si128(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc1++ ^ *pLocalSrc2++;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

#if USE_BMI2
/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_cmpgt_epi8(pixels2, zeromask);				// Convert 0x01 to 0xFF
			pixels1 = _mm_xor_si128(pixels1, pixels2);
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_xor_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_cmpgt_epi8(pixels1, zeromask);						// Convert 0x01 to 0xFF
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/* The function assumes that the source image pointers are 16 byte aligned, and the source strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * src2 = (__m128i*)pSrcImage2;
	__m128i pixels1, pixels2;

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);
			pixels2 = _mm_load_si128(&src2[width >> 4]);
			pixels1 = _mm_xor_si128(pixels1, pixels2);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		src2 += (srcImage2StrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

/* The function assumes that the image pointers are 16 byte aligned, and the source and destination strides as well
It processes the pixels in a width which is the next highest multiple of 16 after dstWidth */
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			
			pixels1 = _mm_xor_si128(pixels1, pixels2);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_xor_si128(pixels1, pixels2);							// Only the LSB here has the AND value

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#else
/* The function assumes that the image widths are a multiple of 8 pixels */
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalDst;
	vx_int16 *pLocalSrc2;
	__m128i pixels1, pixels2;
	vx_int16 U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;
	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_xor_si128(pixels1, pixels2);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			vx_uint8 pix;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				pix = (temp & 1) ? (vx_uint8)255 : 0;
				*pLocalDst = pix ^ ((vx_uint8)(*pLocalSrc1));
				temp >>= 1;
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
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_xor_si128(pixels1, pixels2);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			vx_uint8 pix;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				pix = (temp & 1) ? (vx_uint8)255 : 0;
				*pLocalDst = pix ^ ((vx_uint8)(*pLocalSrc1));
				temp >>= 1;
			}
			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the width is a multiple of 8 pixels */
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
	)
{
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int32 * pLocalDst;
	vx_uint8 pixels1, pixels2;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_uint8 *)pSrcImage1;
		pLocalSrc2 = (vx_uint8 *)pSrcImage2;
		pLocalDst = (vx_int32 *)pDstImage;

		for (int width = 0; width < (int)dstWidth; width += 8)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 ^ pixels2;

			// U1 to U8
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
			pixels1 >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels */
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i * pLocalSrc1_xmm, *pLocalSrc2_xmm;
	__m128i pixels1, pixels2;
	int U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_xor_si128(pixels1, pixels2);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= ((*pLocalSrc1++ ^ *pLocalSrc2++) >> 7) & 1;		// the signed bit has the information
					temp <<= 1;
					temp = (temp > 127) ? 255 : temp;
				}
				*pLocalDst++ = temp;
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
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_xor_si128(pixels1, pixels2);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= ((*pLocalSrc1++ ^ *pLocalSrc2++) >> 7) & 1;
					temp <<= 1;
					temp = (temp > 127) ? 255 : temp;
				}
				*pLocalDst++ = temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels*/
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
	)
{
	__m128i * pLocalSrc1_xmm;

	__m128i pixels;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1_xmm = (__m128i *) pSrcImage1;
		vx_int16 * pLocalSrc2_16 = (vx_int16 *)pSrcImage2;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels = _mm_loadu_si128(pLocalSrc1_xmm++);
			pixels1 = (vx_int16)(_mm_movemask_epi8(pixels) & 0xFFFF);
			pixels2 = *pLocalSrc2_16++;

			pixels1 = pixels1 ^ pixels2;
			*pLocalDst_16++ = pixels1;
		}

		if (postfixWidth)
		{
			vx_uint8 * pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			vx_uint8 * pLocalDst = (vx_uint8 *)pLocalDst_16;
			vx_uint8 pix = *((vx_uint8 *)pLocalSrc2_16);
			vx_uint8 temp = 0;
			for (int i = 0; i < 8; i++)
			{
				temp |= ((*pLocalSrc1++) >> 7) & 1;
				temp <<= 1;
				temp = (temp > 127) ? 255 : temp;
			}
			*pLocalDst++ = temp ^ pix;
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the widths are a multiple of 8 pixels*/
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
	)
{
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_int16 *)pSrcImage1;
		pLocalSrc2 = (vx_int16 *)pSrcImage2;
		pLocalDst = (vx_int16 *)pDstImage;
		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 ^ pixels2;
			*pLocalDst++ = pixels1;
		}

		if (postfixWidth)
		{
			*((vx_uint8*)pLocalDst) = *((vx_uint8*)pLocalSrc1) ^ *((vx_uint8*)pLocalSrc2);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#endif

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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m128i pixels1, pixels2;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

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
				pixels1 = _mm_and_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = ~(*pLocalSrc1++ & *pLocalSrc2++);
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
				pixels1 = _mm_and_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = ~(*pLocalSrc1++ & *pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}
	
	return AGO_SUCCESS;
}

#if USE_BMI2
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_cmpgt_epi8(pixels2, zeromask);				// Convert 0x01 to 0xFF
			pixels1 = _mm_and_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_and_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_cmpgt_epi8(pixels1, zeromask);						// Convert 0x01 to 0xFF
			pixels1 = _mm_andnot_si128(pixels1, ones);
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;

}

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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * src2 = (__m128i*)pSrcImage2;
	__m128i pixels1, pixels2;

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);
			pixels2 = _mm_load_si128(&src2[width >> 4]);
			pixels1 = _mm_and_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		src2 += (srcImage2StrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels1 = _mm_and_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_and_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_andnot_si128(pixels1, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#else
/* The function assumes that the image widths are a multiple of 8 pixels */
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalDst_xmm;
	__m128i ones = _mm_set1_epi32(0xFFFFFFFF);

	vx_uint8 *pLocalSrc1, *pLocalDst;
	vx_int16 *pLocalSrc2;
	__m128i pixels1, pixels2;
	vx_int16 U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;
	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_and_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				*pLocalDst = ~((temp & 1) * (vx_uint8)(*pLocalSrc1));
				temp >>= 1;
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
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_and_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				*pLocalDst = ~((temp & 1) * (vx_uint8)(*pLocalSrc1));
				temp >>= 1;
			}
			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the width is a multiple of 8 pixels */
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
	)
{
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int32 * pLocalDst;
	vx_uint8 pixels1, pixels2;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_uint8 *)pSrcImage1;
		pLocalSrc2 = (vx_uint8 *)pSrcImage2;
		pLocalDst = (vx_int32 *)pDstImage;

		for (int width = 0; width < (int)dstWidth; width += 8)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = ~(pixels1 & pixels2);

			// U1 to U8
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
			pixels1 >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels */
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i * pLocalSrc1_xmm, *pLocalSrc2_xmm;
	__m128i pixels1, pixels2;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	int U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_and_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= ((*pLocalSrc1++ & *pLocalSrc2++) >> 7) & 1;		// the signed bit has the information
					temp <<= 1;
				}
				*pLocalDst++ = ~temp;
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
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_and_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= ((*pLocalSrc1++ & *pLocalSrc2++) >> 7) & 1;
					temp <<= 1;
				}
				*pLocalDst++ = ~temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels */
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
	)
{
	__m128i * pLocalSrc1_xmm;

	__m128i pixels;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1_xmm = (__m128i *) pSrcImage1;
		vx_int16 * pLocalSrc2_16 = (vx_int16 *)pSrcImage2;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels = _mm_loadu_si128(pLocalSrc1_xmm++);
			pixels1 = (vx_int16)(_mm_movemask_epi8(pixels) & 0xFFFF);
			pixels2 = *pLocalSrc2_16++;

			pixels1 = pixels1 & pixels2;
			*pLocalDst_16++ = ~pixels1;
		}

		if (postfixWidth)
		{
			vx_uint8 * pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			vx_uint8 * pLocalDst = (vx_uint8 *)pLocalDst_16;
			vx_uint8 pix = *((vx_uint8 *)pLocalSrc2_16);
			vx_uint8 temp = 0;
			for (int i = 0; i < 8; i++)
			{
				temp |= ((*pLocalSrc1++) >> 7) & 1;
				temp <<= 1;
			}
			*pLocalDst++ = ~(temp & pix);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_int16 *)pSrcImage1;
		pLocalSrc2 = (vx_int16 *)pSrcImage2;
		pLocalDst = (vx_int16 *)pDstImage;
		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 & pixels2;
			*pLocalDst++ = ~pixels1;
		}

		if (postfixWidth)
		{
			*((vx_uint8*)pLocalDst) = ~(*((vx_uint8*)pLocalSrc1) & *((vx_uint8*)pLocalSrc2));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#endif

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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m128i pixels1, pixels2;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

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
				pixels1 = _mm_or_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = ~(*pLocalSrc1++ | *pLocalSrc2++);
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
				pixels1 = _mm_or_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = ~(*pLocalSrc1++ | *pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}
#if USE_BMI2
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_cmpgt_epi8(pixels2, zeromask);				// Convert 0x01 to 0xFF
			pixels1 = _mm_or_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_or_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_cmpgt_epi8(pixels1, zeromask);						// Convert 0x01 to 0xFF
			pixels1 = _mm_andnot_si128(pixels1, ones);
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * src2 = (__m128i*)pSrcImage2;
	__m128i pixels1, pixels2;

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);
			pixels2 = _mm_load_si128(&src2[width >> 4]);
			pixels1 = _mm_or_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		src2 += (srcImage2StrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels1 = _mm_or_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
		}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_or_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_andnot_si128(pixels1, ones);
			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#else
/* The function assumes that the widths are a multiple of 8 pixels */
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalDst;
	vx_int16 *pLocalSrc2;
	__m128i pixels1, pixels2;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	vx_int16 U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;
	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_or_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				*pLocalDst = (temp & 1) ? (vx_uint8)(*pLocalSrc1) : (vx_uint8)255;
				temp >>= 1;
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
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_or_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				*pLocalDst = (temp & 1) ? (vx_uint8)(*pLocalSrc1) : (vx_uint8)255;
				temp >>= 1;
			}
			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the width is a multiple of 8 pixels */
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
	)
{
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int32 * pLocalDst;
	vx_uint8 pixels1, pixels2;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_uint8 *)pSrcImage1;
		pLocalSrc2 = (vx_uint8 *)pSrcImage2;
		pLocalDst = (vx_int32 *)pDstImage;

		for (int width = 0; width < (int)dstWidth; width += 8)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = ~(pixels1 | pixels2);

			// U1 to U8
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
			pixels1 >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels */
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i * pLocalSrc1_xmm, *pLocalSrc2_xmm;
	__m128i pixels1, pixels2;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	int U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_or_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= (~((*pLocalSrc1++ | *pLocalSrc2++) >> 7)) & 1;		// the signed bit has the information
					temp <<= 1;
				}
				*pLocalDst++ = temp;
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
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_or_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= (~((*pLocalSrc1++ | *pLocalSrc2++) >> 7)) & 1;
					temp <<= 1;
				}
				*pLocalDst++ = temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels */
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
	)
{
	__m128i * pLocalSrc1_xmm;

	__m128i pixels;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1_xmm = (__m128i *) pSrcImage1;
		vx_int16 * pLocalSrc2_16 = (vx_int16 *)pSrcImage2;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels = _mm_loadu_si128(pLocalSrc1_xmm++);
			pixels1 = (vx_int16)(_mm_movemask_epi8(pixels) & 0xFFFF);
			pixels2 = *pLocalSrc2_16++;

			pixels1 = pixels1 | pixels2;
			*pLocalDst_16++ = ~pixels1;
		}

		if (postfixWidth)
		{
			vx_uint8 * pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			vx_uint8 * pLocalDst = (vx_uint8 *)pLocalDst_16;
			vx_uint8 pix = *((vx_uint8 *)pLocalSrc2_16);
			vx_uint8 temp = 0;
			for (int i = 0; i < 8; i++)
			{
				temp |= ((*pLocalSrc1++) >> 7) & 1;
				temp <<= 1;
			}
			*pLocalDst++ = ~(temp | pix);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the widths are a multiple of 8 pixels */
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
	)
{
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_int16 *)pSrcImage1;
		pLocalSrc2 = (vx_int16 *)pSrcImage2;
		pLocalDst = (vx_int16 *)pDstImage;
		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 | pixels2;
			*pLocalDst++ = ~pixels1;
		}

		if (postfixWidth)
		{
			*((vx_uint8*)pLocalDst) = ~(*((vx_uint8*)pLocalSrc1) | *((vx_uint8*)pLocalSrc2));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#endif

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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2 | (intptr_t)pDstImage) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalSrc2_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	__m128i pixels1, pixels2;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

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
				pixels1 = _mm_xor_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = ~(*pLocalSrc1++ ^ *pLocalSrc2++);
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
				pixels1 = _mm_xor_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}

			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalSrc2 = (vx_uint8 *)pLocalSrc2_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			for (int width = 0; width < postfixWidth; width++)
			{
				*pLocalDst++ = ~(*pLocalSrc1++ ^ *pLocalSrc2++);
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

#if USE_BMI2
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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_cmpgt_epi8(pixels2, zeromask);				// Convert 0x01 to 0xFF
			pixels1 = _mm_xor_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i * dst = (__m128i*)pDstImage;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_xor_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_andnot_si128(pixels1, ones);
			pixels1 = _mm_cmpgt_epi8(pixels1, zeromask);						// Convert 0x01 to 0xFF
			_mm_store_si128(&dst[width >> 4], pixels1);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i * src2 = (__m128i*)pSrcImage2;
	__m128i pixels1, pixels2;

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);
			pixels2 = _mm_load_si128(&src2[width >> 4]);
			pixels1 = _mm_xor_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		src2 += (srcImage2StrideInBytes >> 4);
		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

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
	)
{
	__m128i * src1 = (__m128i*)pSrcImage1;
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			pixels1 = _mm_load_si128(&src1[width >> 4]);

			// Read the U1 values
			pixels_u64[0] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels2 = _mm_load_si128((__m128i*) pixels_u64);
			pixels1 = _mm_xor_si128(pixels1, pixels2);
			pixels1 = _mm_andnot_si128(pixels1, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		src1 += (srcImage1StrideInBytes >> 4);
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

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
	)
{
	__m128i pixels1, pixels2;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage1 + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage1 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage2 + (width >> 3)));
			pixels_u64[3] = (uint64_t)(*(pSrcImage2 + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			pixels1 = _mm_load_si128((__m128i*) pixels_u64);
			pixels2 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			pixels1 = _mm_xor_si128(pixels1, pixels2);							// Only the LSB here has the AND value
			pixels1 = _mm_andnot_si128(pixels1, ones);

			// Convert U8 to U1
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(pixels1.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(pixels1.m128i_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			*((unsigned short *)pDstImage + (width >> 4)) = (unsigned short)(((pixels_u64[1] & 0xFF) << 8) | (pixels_u64[0] & 0xFF));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#else
/* The function assumes that the image widths are a multiple of 8 pixels */
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i *pLocalSrc1_xmm, *pLocalDst_xmm;
	vx_uint8 *pLocalSrc1, *pLocalDst;
	vx_int16 *pLocalSrc2;
	__m128i pixels1, pixels2;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);
	vx_int16 U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;
	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_xor_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_store_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			vx_uint8 pix;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				pix = (temp & 1) ? (vx_uint8)255 : 0;
				*pLocalDst = ~(pix ^ ((vx_uint8)(*pLocalSrc1)));
				temp >>= 1;
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
			pLocalSrc1_xmm = (__m128i *)pSrcImage1;
			pLocalSrc2 = (vx_int16 *)pSrcImage2;
			pLocalDst_xmm = (__m128i *)pDstImage;
			int width;
			for (width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);

				U1pixels = *pLocalSrc2++;
				M128I(pixels2).m128i_i32[0] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[1] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[2] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];
				U1pixels >>= 4;
				M128I(pixels2).m128i_i32[3] = dataConvertU1ToU8_4bytes[U1pixels & 0xF];

				pixels1 = _mm_xor_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);
				_mm_storeu_si128(pLocalDst_xmm++, pixels1);
			}
			pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			pLocalDst = (vx_uint8 *)pLocalDst_xmm;

			width = 0;
			vx_int16 temp = *pLocalSrc2++;
			vx_uint8 pix;
			for (int width = 0; width < postfixWidth; width++, pLocalSrc1++, pLocalDst++)
			{
				pix = (temp & 1) ? (vx_uint8)255 : 0;
				*pLocalDst = ~(pix ^ ((vx_uint8)(*pLocalSrc1)));
				temp >>= 1;
			}
			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the width is a multiple of 8 pixels */
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
	)
{
	vx_uint8 *pLocalSrc1, *pLocalSrc2;
	vx_int32 * pLocalDst;
	vx_uint8 pixels1, pixels2;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_uint8 *)pSrcImage1;
		pLocalSrc2 = (vx_uint8 *)pSrcImage2;
		pLocalDst = (vx_int32 *)pDstImage;

		for (int width = 0; width < (int)dstWidth; width += 8)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = ~(pixels1 ^ pixels2);

			// U1 to U8
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
			pixels1 >>= 4;
			*pLocalDst++ = dataConvertU1ToU8_4bytes[pixels1 & 0xF];
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels */
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
	)
{
	bool useAligned = ((((intptr_t)pSrcImage1 | (intptr_t)pSrcImage2) & 0xF) == 0) ? true : false;

	__m128i * pLocalSrc1_xmm, *pLocalSrc2_xmm;
	__m128i pixels1, pixels2;
	__m128i ones = _mm_set1_epi16((short)0xFFFF);

	int U1pixels;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	if (useAligned)
	{
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_load_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_load_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_xor_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= (~((*pLocalSrc1++ ^ *pLocalSrc2++) >> 7)) & 1;		// the signed bit has the information
					temp <<= 1;
				}
				*pLocalDst++ = temp;
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
			pLocalSrc1_xmm = (__m128i *) pSrcImage1;
			pLocalSrc2_xmm = (__m128i *) pSrcImage2;
			vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

			for (int width = 0; width < alignedWidth; width += 16)
			{
				pixels1 = _mm_loadu_si128(pLocalSrc1_xmm++);
				pixels2 = _mm_loadu_si128(pLocalSrc2_xmm++);
				pixels1 = _mm_xor_si128(pixels1, pixels2);
				pixels1 = _mm_andnot_si128(pixels1, ones);

				U1pixels = _mm_movemask_epi8(pixels1);
				*pLocalDst_16++ = (vx_int16)(U1pixels & 0xFFFF);
			}

			if (postfixWidth)
			{
				vx_uint8 * pLocalSrc1 = (vx_uint8*)pLocalSrc1_xmm;
				vx_uint8 * pLocalSrc2 = (vx_uint8*)pLocalSrc2_xmm;
				vx_uint8 * pLocalDst = (vx_uint8*)pLocalDst_16;
				vx_uint8 temp = 0;
				for (int i = 0; i < 8; i++)
				{
					temp |= (~((*pLocalSrc1++ ^ *pLocalSrc2++) >> 7)) & 1;
					temp <<= 1;
				}
				*pLocalDst++ = temp;
			}

			pSrcImage1 += srcImage1StrideInBytes;
			pSrcImage2 += srcImage2StrideInBytes;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

/* The function assumes that the input widths are a multiple of 8 pixels */
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
	)
{
	__m128i * pLocalSrc1_xmm;

	__m128i pixels;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1_xmm = (__m128i *) pSrcImage1;
		vx_int16 * pLocalSrc2_16 = (vx_int16 *)pSrcImage2;
		vx_int16 * pLocalDst_16 = (vx_int16 *)pDstImage;

		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels = _mm_loadu_si128(pLocalSrc1_xmm++);
			pixels1 = (vx_int16)(_mm_movemask_epi8(pixels) & 0xFFFF);
			pixels2 = *pLocalSrc2_16++;

			pixels1 = pixels1 ^ pixels2;
			*pLocalDst_16++ = ~pixels1;
		}

		if (postfixWidth)
		{
			vx_uint8 * pLocalSrc1 = (vx_uint8 *)pLocalSrc1_xmm;
			vx_uint8 * pLocalDst = (vx_uint8 *)pLocalDst_16;
			vx_uint8 pix = *((vx_uint8 *)pLocalSrc2_16);
			vx_uint8 temp = 0;
			for (int i = 0; i < 8; i++)
			{
				temp |= ((*pLocalSrc1++) >> 7) & 1;
				temp <<= 1;
			}
			*pLocalDst++ = ~(temp ^ pix);
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/* The function assumes that the widths are a multiple of 8 pixels */
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
	)
{
	vx_int16 *pLocalSrc1, *pLocalSrc2, *pLocalDst;
	vx_int16 pixels1, pixels2;

	int alignedWidth = dstWidth & ~15;
	int postfixWidth = dstWidth - alignedWidth;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		pLocalSrc1 = (vx_int16 *)pSrcImage1;
		pLocalSrc2 = (vx_int16 *)pSrcImage2;
		pLocalDst = (vx_int16 *)pDstImage;
		for (int width = 0; width < alignedWidth; width += 16)
		{
			pixels1 = *pLocalSrc1++;
			pixels2 = *pLocalSrc2++;
			pixels1 = pixels1 ^ pixels2;
			*pLocalDst++ = ~pixels1;
		}

		if (postfixWidth)
		{
			*((vx_uint8*)pLocalDst) = ~(*((vx_uint8*)pLocalSrc1) ^ *((vx_uint8*)pLocalSrc2));
		}
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
#endif