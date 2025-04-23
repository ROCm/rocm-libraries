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

DECL_ALIGN(16) unsigned char dataChannelExtract[16 * 29] ATTR_ALIGN(16) = { 
	  0,   2,   4,   6,   8,  10,  12,  14, 255, 255, 255, 255, 255, 255, 255, 255,		// Lower 8 bytes pos0 for U8_U16
	255, 255, 255, 255, 255, 255, 255, 255,   0,   2,   4,   6,   8,  10,  12,  14,		// Upper 8 bytes pos0 for U8_U16
	  1,   3,   5,   7,   9,  11,  13,  15, 255, 255, 255, 255, 255, 255, 255, 255,		// Lower 8 bytes pos1 for U8_U16
	255, 255, 255, 255, 255, 255, 255, 255,   1,   3,   5,   7,   9,  11,  13,  15,		// Upper 8 bytes pos1 for U8_U16
	  0,   3,   6,   9,  12,  15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// Lower 6 bytes pos0 for U8_U24
	255, 255, 255, 255, 255, 255,   2,   5,   8,  11,  14, 255, 255, 255, 255, 255,		// Mid	 5 bytes pos0 for U8_U24 
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  1,   4,   7,  10,  13,		// Upper 5 bytes pos0 for U8_U24 
	  1,   4,   7,  10,  13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// Lower 5 bytes pos1 for U8_U24
	255, 255, 255, 255, 255,   0,   3,   6,   9,  12,  15, 255, 255, 255, 255, 255,		// Mid	 6 bytes pos1 for U8_U24
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   2,   5,   8,  11,  14, 		// Upper 5 bytes pos1 for U8_U24
	  2,   5,   8,  11,  14, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  	// Lower 5 bytes pos2 for U8_U24
	255, 255, 255, 255, 255,   1,   4,   7,  10,  13, 255, 255, 255, 255, 255, 255,		// Mid	 5 bytes pos2 for U8_U24
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   3,   6,   9,  12,  15,		// Upper 6 bytes pos2 for U8_U24
	  0,   4,   8,  12, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// Low	 4 bytes pos0 for U8_U32
	255, 255, 255, 255,   0,   4,   8,  12, 255, 255, 255, 255, 255, 255, 255, 255,		// Next	 4 bytes pos0 for U8_U32
	255, 255, 255, 255, 255, 255, 255, 255,   0,   4,   8,  12, 255, 255, 255, 255,		// Next	 4 bytes pos0 for U8_U32
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   4,   8,  12,		// Upper 4 bytes pos0 for U8_U32	
	  1,   5,   9,  13, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// Low	 4 bytes pos1 for U8_U32
	255, 255, 255, 255,   1,   5,   9,  13, 255, 255, 255, 255, 255, 255, 255, 255,		// Next	 4 bytes pos1 for U8_U32
	255, 255, 255, 255, 255, 255, 255, 255,   1,   5,   9,  13, 255, 255, 255, 255,		// Next	 4 bytes pos1 for U8_U32
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   1,   5,   9,  13,		// Upper 4 bytes pos1 for U8_U32	
	  2,   6,  10,  14, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// Low	 4 bytes pos2 for U8_U32
	255, 255, 255, 255,   2,   6,  10,  14, 255, 255, 255, 255, 255, 255, 255, 255,		// Next	 4 bytes pos2 for U8_U32
	255, 255, 255, 255, 255, 255, 255, 255,   2,   6,  10,  14, 255, 255, 255, 255,		// Next	 4 bytes pos2 for U8_U32
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   2,   6,  10,  14,		// Upper 4 bytes pos2 for U8_U32	
	  3,   7,  11,  15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,		// Low	 4 bytes pos3 for U8_U32
	255, 255, 255, 255,   3,   7,  11,  15, 255, 255, 255, 255, 255, 255, 255, 255,		// Next	 4 bytes pos3 for U8_U32
	255, 255, 255, 255, 255, 255, 255, 255,   3,   7,  11,  15, 255, 255, 255, 255,		// Next	 4 bytes pos3 for U8_U32
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   3,   7,  11,  15		// Upper 4 bytes pos3 for U8_U32
};

DECL_ALIGN(16) unsigned char dataChannelCombine[16 * 15] ATTR_ALIGN(16) = { 
	  0, 255, 255,   1, 255, 255,   2, 255, 255,   3, 255, 255,   4, 255, 255,   5,		// R into first  16 bytes for RGB
	255, 255,   6, 255, 255,   7, 255, 255,   8, 255, 255,   9, 255, 255,  10, 255,		// R into second 16 bytes for RGB
	255,  11, 255, 255,  12, 255, 255,  13, 255, 255,  14, 255, 255,  15, 255, 255,	 	// R into third  16 bytes for RGB
	255,   0, 255, 255,   1, 255, 255,   2, 255, 255,   3, 255, 255,   4, 255, 255,		// G into first  16 bytes for RGB
	  5, 255, 255,   6, 255, 255,   7, 255, 255,   8, 255, 255,   9, 255, 255,  10,		// G into second 16 bytes for RGB
	255, 255,  11, 255, 255,  12, 255, 255,  13, 255, 255,  14, 255, 255,  15, 255,		// G into third  16 bytes for RGB
	255, 255,   0, 255, 255,   1, 255, 255,   2, 255, 255,   3, 255, 255,   4, 255,		// B into first  16 bytes for RGB
	255,   5, 255, 255,   6, 255, 255,   7, 255, 255,   8, 255, 255,   9, 255, 255,		// B into second 16 bytes for RGB
	 10, 255, 255,  11, 255, 255,  12, 255, 255,  13, 255, 255,  14, 255, 255,  15,		// B into third  16 bytes for RGB
	255,   0, 255,   1, 255,   2, 255,   3, 255,   4, 255,   5, 255,   6, 255,	 7,		// Y into UYVY
	  0, 255, 255, 255,   1, 255, 255, 255,   2, 255, 255, 255,   3, 255, 255, 255,		// U into UYVY
	255, 255,   0, 255, 255, 255,   1, 255, 255, 255,   2, 255, 255, 255,   3, 255,		// V into UYVY
	  0, 255,   1, 255,   2, 255,   3, 255,   4, 255,   5, 255,   6, 255,   7, 255,		// Y into YUYV
	255,   0, 255, 255, 255,   1, 255, 255, 255,   2, 255, 255, 255,   3, 255, 255,		// U into YUYV
	255, 255, 255,   0, 255, 255, 255,   1, 255, 255, 255,   2, 255, 255, 255,	 3,		// V into YUYV
};

extern vx_uint32 dataConvertU1ToU8_4bytes[16];

/* This function assumes that the pixelSizeinBytes is equal to the srcStrideX*/
int HafCpu_BufferCopyDisperseInDst
	(
		vx_uint32	  dstWidth,
		vx_uint32	  dstHeight,
		vx_uint32	  pixelSizeInBytes,
		vx_uint8	* pDstImage,
		vx_uint32	  dstImageStrideYInBytes,
		vx_uint32	  dstImageStrideXInBytes,
		vx_uint8	* pSrcImage,
		vx_uint32	  srcImageStrideYInBytes
	)
{
	if (pixelSizeInBytes == 1)					// 8 bits per pixel
	{
		vx_uint8 *pLocalSrc, *pLocalDst;
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc = pSrcImage;
			pLocalDst = pDstImage;

			for (int width = 0; width < (int)dstWidth; width++)
			{
				*pLocalDst = *pLocalSrc++;
				pLocalDst += dstImageStrideXInBytes;
			}
			pSrcImage += srcImageStrideYInBytes;
			pDstImage += dstImageStrideYInBytes;
		}
	}
	else if (pixelSizeInBytes == 2)				// 16 bits per pixel
	{
		vx_int16 *pLocalSrc, *pLocalDst;
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc = (vx_int16 *) pSrcImage;
			pLocalDst = (vx_int16 *) pDstImage;

			int xStride = dstImageStrideXInBytes >> 1;
			for (int width = 0; width < (int)dstWidth; width++)
			{
				*pLocalDst = *pLocalSrc++;
				pLocalDst += xStride;
			}
			pSrcImage += srcImageStrideYInBytes;
			pDstImage += dstImageStrideYInBytes;
		}
	}
	else if (pixelSizeInBytes == 4)				// 32 bits per pixel
	{
		vx_int32 *pLocalSrc, *pLocalDst;
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc = (vx_int32 *)pSrcImage;
			pLocalDst = (vx_int32 *)pDstImage;

			int xStride = dstImageStrideXInBytes >> 2;
			for (int width = 0; width < (int)dstWidth; width++)
			{
				*pLocalDst = *pLocalSrc++;
				pLocalDst += xStride;
			}
			pSrcImage += srcImageStrideYInBytes;
			pDstImage += dstImageStrideYInBytes;
		}
	}
	else										// General case
	{
		vx_uint8 *pLocalSrc, *pLocalDst;
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc = pSrcImage;
			pLocalDst = pDstImage;

			for (int width = 0; width < (int)dstWidth; width++)
			{
				for (int byte = 0; byte < (int)pixelSizeInBytes; byte++)
					*pLocalDst++ = *pLocalSrc++;
				pLocalDst += dstImageStrideXInBytes;
			}
			pSrcImage += srcImageStrideYInBytes;
			pDstImage += dstImageStrideYInBytes;
		}
	}
	
	return AGO_SUCCESS;
}

/* This function assumes that the pixelSizeinBytes is equal to the dstStrideX*/
int HafCpu_BufferCopyDisperseInSrc
	(
		vx_uint32	  dstWidth,
		vx_uint32	  dstHeight,
		vx_uint32	  pixelSizeInBytes,
		vx_uint8	* pDstImage,
		vx_uint32	  dstImageStrideYInBytes,
		vx_uint8	* pSrcImage,
		vx_uint32	  srcImageStrideYInBytes,
		vx_uint32	  srcImageStrideXInBytes
	)
{
	if (pixelSizeInBytes == 1)					// 8 bits per pixel
	{
		vx_uint8 *pLocalSrc, *pLocalDst;
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc = pSrcImage;
			pLocalDst = pDstImage;

			for (int width = 0; width < (int)dstWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc;
				pLocalSrc += srcImageStrideXInBytes;
			}
			pSrcImage += srcImageStrideYInBytes;
			pDstImage += dstImageStrideYInBytes;
		}
	}
	else if (pixelSizeInBytes == 2)				// 16 bits per pixel
	{
		vx_int16 *pLocalSrc, *pLocalDst;
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc = (vx_int16 *)pSrcImage;
			pLocalDst = (vx_int16 *)pDstImage;

			int xStride = srcImageStrideXInBytes >> 1;
			for (int width = 0; width < (int)dstWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc;
				pLocalSrc += xStride;
			}
			pSrcImage += srcImageStrideYInBytes;
			pDstImage += dstImageStrideYInBytes;
		}
	}
	else if (pixelSizeInBytes == 4)				// 32 bits per pixel
	{
		vx_int32 *pLocalSrc, *pLocalDst;
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc = (vx_int32 *)pSrcImage;
			pLocalDst = (vx_int32 *)pDstImage;

			int xStride = srcImageStrideXInBytes >> 2;
			for (int width = 0; width < (int)dstWidth; width++)
			{
				*pLocalDst++ = *pLocalSrc;
				pLocalSrc += xStride;
			}
			pSrcImage += srcImageStrideYInBytes;
			pDstImage += dstImageStrideYInBytes;
		}
	}
	else										// General case
	{
		vx_uint8 *pLocalSrc, *pLocalDst;
		for (int height = 0; height < (int)dstHeight; height++)
		{
			pLocalSrc = pSrcImage;
			pLocalDst = pDstImage;

			for (int width = 0; width < (int)dstWidth; width++)
			{
				for (int byte = 0; byte < (int)pixelSizeInBytes; byte++)
					*pLocalDst++ = *pLocalSrc++;
				pLocalSrc += srcImageStrideXInBytes;
			}
			pSrcImage += srcImageStrideYInBytes;
			pDstImage += dstImageStrideYInBytes;
		}
	}

	return AGO_SUCCESS;
}

int HafCpu_BinaryCopy_U8_U8
	(
		vx_size       size,
		vx_uint8    * pDstBuf,
		vx_uint8    * pSrcBuf
	)
{
	if ((intptr_t(pSrcBuf) & 15) | (intptr_t(pDstBuf) & 15))
		memcpy(pDstBuf, pSrcBuf, size);
	else
	{
		__m128i * src = (__m128i*) pSrcBuf;
		__m128i * dst = (__m128i*) pDstBuf;
		__m128i r0, r1, r2, r3;

		vx_size prefixBytes = intptr_t(pDstBuf) & 15;
		vx_size sizeAligned = size & ~63;

		for (unsigned int i = 0; i < sizeAligned; i += 64)
		{
			r0 = _mm_loadu_si128(src++);
			r1 = _mm_loadu_si128(src++);
			r2 = _mm_loadu_si128(src++);
			r3 = _mm_loadu_si128(src++);
			_mm_store_si128(dst++, r0);
			_mm_store_si128(dst++, r1);
			_mm_store_si128(dst++, r2);
			_mm_store_si128(dst++, r3);
		}
		for (vx_size i = sizeAligned; i < size; i++) {
			pDstBuf[i] = pSrcBuf[i];
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelCopy_U8_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	if ((srcImageStrideInBytes | dstImageStrideInBytes) & 15)
	{
		int height = (int)dstHeight;
		while (height)
		{
			unsigned char * pLocalSrc = (unsigned char *)pSrcImage;
			unsigned char * pLocalDst = (unsigned char *)pDstImage;
			int width = (int)dstWidth;
			while (width)
			{
				*pLocalDst++ = *pLocalSrc++;
				width--;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
			height--;
		}
	}
	else
	{
		__m128i r0, r1;
		unsigned char *pLocalSrc, *pLocalDst;
		__m128i *pLocalSrc_xmm, *pLocalDst_xmm;

		int prefixWidth = intptr_t(pDstImage) & 15;
		prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
		int postfixWidth = ((int)dstWidth - prefixWidth) & 31;					// 32 pixels processed at a time in SSE loop
		int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDst = (unsigned char *)pDstImage;

			for (int x = 0; x < prefixWidth; x++)
				*pLocalDst++ = *pLocalSrc++;

			int width = alignedWidth >> 5;									// 32 pixels copied at a time
			pLocalSrc_xmm = (__m128i *) pLocalSrc;
			pLocalDst_xmm = (__m128i *) pLocalDst;
			while (width)
			{
				r0 = _mm_loadu_si128(pLocalSrc_xmm++);
				_mm_store_si128(pLocalDst_xmm++, r0);
				r1 = _mm_loadu_si128(pLocalSrc_xmm++);
				_mm_store_si128(pLocalDst_xmm++, r1);

				width--;
			}

			pLocalSrc = (unsigned char *)pLocalSrc_xmm;
			pLocalDst = (unsigned char *)pLocalDst_xmm;
			for (int x = 0; x < postfixWidth; x++)
				*pLocalDst++ = *pLocalSrc++;

			pSrcImage += srcImageStrideInBytes;
			pDstImage += dstImageStrideInBytes;
			height--;
		}
	}
	return AGO_SUCCESS;
}

#if USE_BMI2
/*The function assumes that the destination pointer is 16 byte aligned and the destination stride as well.
Also, the width is a multiple of 32, if not then number of pixels copies would be the next largest multiple of 32 after dstWidth. 
The LSB of every byte is copies, therefore 0 -> 0 and non zero -> 1*/
int HafCpu_ChannelCopy_U8_U1
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
	__m128i r0, r1;
	__m128i zeromask = _mm_setzero_si128();

	__declspec(align(16)) uint64_t pixels_u64[4];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 32)
		{
			// Read the U1 values from src1
			pixels_u64[0] = (uint64_t)(*(pSrcImage + (width >> 3)));
			pixels_u64[1] = (uint64_t)(*(pSrcImage + (width >> 3) + 1));
#ifdef _WIN64
			pixels_u64[0] = _pdep_u64(pixels_u64[0], maskConv);
			pixels_u64[1] = _pdep_u64(pixels_u64[1], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			// Read the U1 values from src2
			pixels_u64[2] = (uint64_t)(*(pSrcImage + (width >> 3) + 2));
			pixels_u64[3] = (uint64_t)(*(pSrcImage + (width >> 3) + 3));
#ifdef _WIN64
			pixels_u64[2] = _pdep_u64(pixels_u64[2], maskConv);
			pixels_u64[3] = _pdep_u64(pixels_u64[3], maskConv);
#else
#pragma message("Warning: TBD: need a 32-bit implementation using _pext_u32")
#endif
			r0 = _mm_load_si128((__m128i*) pixels_u64);
			r1 = _mm_load_si128((__m128i*) (pixels_u64 + 2));

			// Convert U1 to U8	- Thresholded
			r0 = _mm_cmpgt_epi8(r0, zeromask);
			r1 = _mm_cmpgt_epi8(r1, zeromask);

			_mm_store_si128(&dst[width >> 4], r0);
			_mm_store_si128(&dst[(width >> 4) + 1], r1);
		}
		pSrcImage += srcImageStrideInBytes;
		dst += (dstImageStrideInBytes >> 4);
	}
	return AGO_SUCCESS;
}

/*The function assumes that the source pointer is 16 byte aligned and the source stride as well.
Also, the width is a multiple of 32, if not then number of pixels copies would be the next largest multiple of 16 after dstWidth.
The LSB of every byte is copies, therefore 0 -> 0 and non zero -> 1*/
int HafCpu_ChannelCopy_U1_U8
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
	__m128i r0;

	__declspec(align(16)) uint64_t pixels_u64[2];
	uint64_t maskConv = 0x0101010101010101;

	for (unsigned int height = 0; height < dstHeight; height++)
	{
		for (unsigned int width = 0; width < dstWidth; width += 16)
		{
			r0 = _mm_load_si128(&src[width >> 4]);
			
			// Convert U8 to U1	- Extract LSB
#ifdef _WIN64
			pixels_u64[0] = _pext_u64(r0.m128i_u64[0], maskConv);
			pixels_u64[1] = _pext_u64(r0.m128i_u64[1], maskConv);
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
#else

int HafCpu_ChannelCopy_U8_U1
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

	int height = (int) dstHeight;
	short inputPixels;
	int outputPixels[4];

	int postfixWidth = dstWidth & 15;			// The input has to be a multiple of 16 or 8 pixels (U1 = 8 pixels in a byte)
	int alignedWidth = dstWidth >> 4;
	while (height > 0)
	{
		pLocalSrc = (short *) pSrcImage;
		pLocalDst = (int *) pDstImage;
		int width = alignedWidth;				// Each inner loop processess 4 output ints = 4*4 = 16 bytes

		while (width > 0)
		{
			inputPixels = *pLocalSrc++;
			outputPixels[0] = dataConvertU1ToU8_4bytes[inputPixels & 15];
			inputPixels >>= 4;
			outputPixels[1] = dataConvertU1ToU8_4bytes[inputPixels & 15];
			inputPixels >>= 4;
			outputPixels[2] = dataConvertU1ToU8_4bytes[inputPixels & 15];
			inputPixels >>= 4;
			outputPixels[3] = dataConvertU1ToU8_4bytes[inputPixels & 15];
			*pLocalDst++ = outputPixels[0];
			*pLocalDst++ = outputPixels[1];
			*pLocalDst++ = outputPixels[2];
			*pLocalDst++ = outputPixels[3];

			width--;
		}

		width = postfixWidth;
		while (width > 0)
		{
			inputPixels = *((vx_uint8 *)pLocalSrc - 1);
			outputPixels[0] = dataConvertU1ToU8_4bytes[inputPixels & 15];
			inputPixels >>= 4;
			outputPixels[1] = dataConvertU1ToU8_4bytes[inputPixels & 15];
			*pLocalDst++ = outputPixels[0];
			*pLocalDst++ = outputPixels[1];
			width = 0;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;

		height--;
	}
	return AGO_SUCCESS;
}

/*The function assumes that the source pointer is 16 byte aligned and the source stride as well.
Also, the width is a multiple of 16, if not then number of pixels copies would be the next largest multiple of 16 after dstWidth.
The function also assumes that the input is either 0x00 or 0xFF. Only the MSB of the pixelvalues of input is used*/
int HafCpu_ChannelCopy_U1_U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	__m128i * pLocalSrc;
	short * pLocalDst;

	__m128i pixels;
	int pixelmask;
	int height = (int) dstHeight;

	while (height > 0)
	{
		pLocalSrc = (__m128i*) pSrcImage;
		pLocalDst = (short *) pDstImage;
		int width = (int) (dstWidth >> 4);		// 16 pixels (bits) are processed at a time in the inner loop
		while (width > 0)
		{
			pixels = _mm_load_si128(pLocalSrc++);
			pixelmask = _mm_movemask_epi8(pixels);
			*pLocalDst++ = (short) (pixelmask & 0xFFFF);
			width--;
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	
	return AGO_SUCCESS;
}
#endif

/* This function assumes that the width is a multiple of 8, if not, then the number of pixels copied is the next highest multiple of 8 after dstWidth*/
int HafCpu_ChannelCopy_U1_U1
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	for (unsigned int y = 0; y < dstHeight; y++)
	{
		for (unsigned int x = 0; x < (dstWidth >> 3); x++)
		{
			pDstImage[x] = pSrcImage[x];
		}
		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

/*The function assumes that the data pointers are 16 byte aligned, and size is a multiple of 16, otherwise it is taken to be the multiple of 16 largest after size*/
int HafCpu_MemSet_U8
	(
		vx_size       count,
		vx_uint8    * pDstBuf,
		vx_uint8      value
	)
{
	__m128i val = _mm_set1_epi8((char)value);
	__m128i * buf = (__m128i *) pDstBuf;
	__m128i * buf_end = buf + (count >> 4);
	for (; buf != buf_end; buf++)
		_mm_store_si128(buf, val);
	return AGO_SUCCESS;
}

/*The function assumes that the data pointers are 16 byte aligned, and size is a multiple of 16, otherwise it is taken to be the multiple of 16 largest after size*/
int HafCpu_MemSet_U16
	(
		vx_size       count,
		vx_uint16   * pDstBuf,
		vx_uint16     value
	)
{
	__m128i val = _mm_set1_epi16((short)value);
	__m128i * buf = (__m128i *) pDstBuf;
	__m128i * buf_end = buf + (count >> 3);
	for (; buf != buf_end; buf++)
		_mm_store_si128(buf, val);
	return AGO_SUCCESS;
}

/*The function assumes that the data pointers are 16 byte aligned, and size is a multiple of 48, otherwise it is taken to be the multiple of 48 largest after size*/
int HafCpu_MemSet_U24
	(
		vx_size       count,
		vx_uint8	* pDstBuf,
		vx_uint32     value
	)
{
	char val_R = (char)(value & 0xFF);
	char val_G = (char)((value >> 8) & 0xFF);
	char val_B = (char)((value >> 16) & 0xFF);
	__m128i val1 = _mm_set_epi8(val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R);
	__m128i val2 = _mm_set_epi8(val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G);
	__m128i val3 = _mm_set_epi8(val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B, val_G, val_R, val_B);
	__m128i * buf = (__m128i *) pDstBuf;
	__m128i * buf_end = buf + ((count*3) >> 4);
	for (; buf < buf_end;) {
		_mm_store_si128(buf++, val1);
		_mm_store_si128(buf++, val2);
		_mm_store_si128(buf++, val3);
	}
	return AGO_SUCCESS;
}

/*The function assumes that the data pointers are 16 byte aligned, and size is a multiple of 16, otherwise it is taken to be the multiple of 16 largest after size*/
int HafCpu_MemSet_U32
	(
		vx_size       count,
		vx_uint32   * pDstBuf,
		vx_uint32     value
	)
{
	__m128i val = _mm_set1_epi32((int)value);
	__m128i * buf = (__m128i *) pDstBuf;
	__m128i * buf_end = buf + (count >> 2);
	for (; buf != buf_end; buf++)
		_mm_store_si128(buf, val);
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U16_Pos0
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1;
	__m128i mask1 = _mm_load_si128(tbl);
	__m128i mask2 = _mm_load_si128(tbl + 1);
	
	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r0 = _mm_or_si128(r0, r1);
			_mm_storeu_si128((__m128i *) pLocalDst, r0);

			pLocalSrc += 32;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			*pLocalDst++ = *pLocalSrc++;
			pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U16_Pos1
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1;
	__m128i mask1 = _mm_load_si128(tbl + 2);
	__m128i mask2 = _mm_load_si128(tbl + 3);

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r0 = _mm_or_si128(r0, r1);
			_mm_storeu_si128((__m128i *) pLocalDst, r0);

			pLocalSrc += 32;
			pLocalDst += 16;
		}
		
		for (int width = 0; width < postfixWidth; width++)
		{
			pLocalSrc++;
			*pLocalDst++ = *pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U24_Pos0
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1, r2;
	__m128i mask1 = _mm_load_si128(tbl + 4);
	__m128i mask2 = _mm_load_si128(tbl + 5);
	__m128i mask3 = _mm_load_si128(tbl + 6);

	int height = (int)dstHeight;
	while (height)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;
		int width = alignedWidth >> 4;

		while (width)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));

			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r2 = _mm_shuffle_epi8(r2, mask3);
			r0 = _mm_or_si128(r0, r1);
			r0 = _mm_or_si128(r0, r2);

			_mm_storeu_si128((__m128i *) pLocalDst, r0);
			width--;
			pLocalSrc += 48;
			pLocalDst += 16;
		}

		width = postfixWidth;
		while (width)
		{
			*pLocalDst++ = *pLocalSrc;
			pLocalSrc += 3;
			width--;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U24_Pos1
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1, r2;
	__m128i mask1 = _mm_load_si128(tbl + 7);
	__m128i mask2 = _mm_load_si128(tbl + 8);
	__m128i mask3 = _mm_load_si128(tbl + 9);

	int height = (int)dstHeight;
	while (height)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;
		int width = alignedWidth >> 4;

		while (width)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));

			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r2 = _mm_shuffle_epi8(r2, mask3);
			r0 = _mm_or_si128(r0, r1);
			r0 = _mm_or_si128(r0, r2);

			_mm_storeu_si128((__m128i *) pLocalDst, r0);
			width--;
			pLocalSrc += 48;
			pLocalDst += 16;
		}

		width = postfixWidth;
		while (width)
		{
			*pLocalDst++ = *++pLocalSrc;
			pLocalSrc += 2;
			width--;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U24_Pos2
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1, r2;
	__m128i mask1 = _mm_load_si128(tbl + 10);
	__m128i mask2 = _mm_load_si128(tbl + 11);
	__m128i mask3 = _mm_load_si128(tbl + 12);

	int height = (int)dstHeight;
	while (height)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;
		int width = alignedWidth >> 4;

		while (width)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));

			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r2 = _mm_shuffle_epi8(r2, mask3);
			r0 = _mm_or_si128(r0, r1);
			r0 = _mm_or_si128(r0, r2);

			_mm_storeu_si128((__m128i *) pLocalDst, r0);
			width--;
			pLocalSrc += 48;
			pLocalDst += 16;
		}

		width = postfixWidth;
		while (width)
		{
			pLocalSrc += 2;
			*pLocalDst++ = *pLocalSrc++;
			width--;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U32_Pos0
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1, r2, r3;
	__m128i mask1 = _mm_load_si128(tbl + 13);
	__m128i mask2 = _mm_load_si128(tbl + 14);
	__m128i mask3 = _mm_load_si128(tbl + 15);
	__m128i mask4 = _mm_load_si128(tbl + 16);

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));
			r3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 48));
			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r2 = _mm_shuffle_epi8(r2, mask3);
			r3 = _mm_shuffle_epi8(r3, mask4);
			r0 = _mm_or_si128(r0, r1);
			r0 = _mm_or_si128(r0, r2);
			r0 = _mm_or_si128(r0, r3);
			_mm_storeu_si128((__m128i *)pLocalDst, r0);

			pLocalSrc += 64;
			pLocalDst += 16;
		}
		
		for (int width = 0; width < postfixWidth; width++)
		{
			*pLocalDst++ = *pLocalSrc;
			pLocalSrc += 4;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U32_Pos1
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1, r2, r3;
	__m128i mask1 = _mm_load_si128(tbl + 17);
	__m128i mask2 = _mm_load_si128(tbl + 18);
	__m128i mask3 = _mm_load_si128(tbl + 19);
	__m128i mask4 = _mm_load_si128(tbl + 20);

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));
			r3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 48));
			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r2 = _mm_shuffle_epi8(r2, mask3);
			r3 = _mm_shuffle_epi8(r3, mask4);
			r0 = _mm_or_si128(r0, r1);
			r0 = _mm_or_si128(r0, r2);
			r0 = _mm_or_si128(r0, r3);
			_mm_storeu_si128((__m128i *)pLocalDst, r0);

			pLocalSrc += 64;
			pLocalDst += 16;
		}
		
		for (int width = 0; width < postfixWidth; width++)
		{
			*pLocalDst++ = *++pLocalSrc;
			pLocalSrc += 3;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U32_Pos2
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1, r2, r3;
	__m128i mask1 = _mm_load_si128(tbl + 21);
	__m128i mask2 = _mm_load_si128(tbl + 22);
	__m128i mask3 = _mm_load_si128(tbl + 23);
	__m128i mask4 = _mm_load_si128(tbl + 24);

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));
			r3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 48));
			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r2 = _mm_shuffle_epi8(r2, mask3);
			r3 = _mm_shuffle_epi8(r3, mask4);
			r0 = _mm_or_si128(r0, r1);
			r0 = _mm_or_si128(r0, r2);
			r0 = _mm_or_si128(r0, r3);
			_mm_storeu_si128((__m128i *)pLocalDst, r0);

			pLocalSrc += 64;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			pLocalSrc += 2;
			*pLocalDst++ = *pLocalSrc;
			pLocalSrc += 2;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8_U32_Pos3
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

	__m128i * tbl = (__m128i*) dataChannelExtract;
	__m128i r0, r1, r2, r3;
	__m128i mask1 = _mm_load_si128(tbl + 25);
	__m128i mask2 = _mm_load_si128(tbl + 26);
	__m128i mask3 = _mm_load_si128(tbl + 27);
	__m128i mask4 = _mm_load_si128(tbl + 28);

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc = pSrcImage;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 4); width++)
		{
			r0 = _mm_loadu_si128((__m128i *)pLocalSrc);
			r1 = _mm_loadu_si128((__m128i *)(pLocalSrc + 16));
			r2 = _mm_loadu_si128((__m128i *)(pLocalSrc + 32));
			r3 = _mm_loadu_si128((__m128i *)(pLocalSrc + 48));
			r0 = _mm_shuffle_epi8(r0, mask1);
			r1 = _mm_shuffle_epi8(r1, mask2);
			r2 = _mm_shuffle_epi8(r2, mask3);
			r3 = _mm_shuffle_epi8(r3, mask4);
			r0 = _mm_or_si128(r0, r1);
			r0 = _mm_or_si128(r0, r2);
			r0 = _mm_or_si128(r0, r3);
			_mm_storeu_si128((__m128i *)pLocalDst, r0);

			pLocalSrc += 64;
			pLocalDst += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			pLocalSrc += 3;
			*pLocalDst++ = *pLocalSrc++;
		}

		pSrcImage += srcImageStrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelCombine_U16_U8U8
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes
	)
{
	unsigned char *pLocalSrc0, *pLocalSrc1, *pLocalDst;
	__m128i r0, r1, resultL, resultH;
	__m128i *pLocalSrc0_xmm, *pLocalSrc1_xmm, *pLocalDst_xmm;

	int prefixWidth = intptr_t(pDstImage) & 15;
	prefixWidth = (prefixWidth == 0) ? 0 : (16 - prefixWidth);
	int postfixWidth = ((int)dstWidth - prefixWidth) & 31;					// 32 pixels processed at a time in SSE loop
	int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

	int height = (int)dstHeight;
	while (height)
	{
		pLocalSrc0 = (unsigned char *) pSrcImage0;
		pLocalSrc1 = (unsigned char *) pSrcImage1;
		pLocalDst = (unsigned char *) pDstImage;

		for (int x = 0; x < prefixWidth; x++)
		{
			*pLocalDst++ = *pLocalSrc0++;
			*pLocalDst++ = *pLocalSrc1++;
		}


		int width = (int)(dstWidth >> 4);									// 16 byte pairs copied into dst at once
		pLocalSrc0_xmm = (__m128i *) pLocalSrc0;
		pLocalSrc1_xmm = (__m128i *) pLocalSrc1;
		pLocalDst_xmm = (__m128i *) pLocalDst;
		while (width)
		{
			r0 = _mm_load_si128(pLocalSrc0_xmm++);
			r1 = _mm_load_si128(pLocalSrc1_xmm++);
			resultL = _mm_unpacklo_epi8(r0, r1);
			resultH = _mm_unpackhi_epi8(r0, r1);
			_mm_store_si128(pLocalDst_xmm++, resultL);
			_mm_store_si128(pLocalDst_xmm++, resultH);
			width--;
		}

		pLocalSrc0 = (unsigned char *) pLocalSrc0_xmm;
		pLocalSrc1 = (unsigned char *) pLocalSrc1_xmm;
		pLocalDst = (unsigned char *) pLocalDst_xmm;
		for (int x = 0; x < postfixWidth; x++)
		{
			*pLocalDst++ = *pLocalSrc0++;
			*pLocalDst++ = *pLocalSrc1++;
		}

		pSrcImage0 += srcImage0StrideInBytes;
		pSrcImage1 += srcImage1StrideInBytes;
		pDstImage += dstImageStrideInBytes;
		height--;
	}

	return AGO_SUCCESS;
}

int HafCpu_ChannelCombine_U24_U8U8U8_RGB
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataChannelCombine;
	__m128i r, g, b, result1, result2, result3;
	__m128i maskR1 = _mm_load_si128(tbl);
	__m128i maskR2 = _mm_load_si128(tbl + 1);
	__m128i maskR3 = _mm_load_si128(tbl + 2);
	__m128i maskG1 = _mm_load_si128(tbl + 3);
	__m128i maskG2 = _mm_load_si128(tbl + 4);
	__m128i maskG3 = _mm_load_si128(tbl + 5);
	__m128i maskB1 = _mm_load_si128(tbl + 6);
	__m128i maskB2 = _mm_load_si128(tbl + 7);
	__m128i maskB3 = _mm_load_si128(tbl + 8);

	int height = (int) dstHeight;
	while (height)
	{
		vx_uint8 * pLocalSrc0 = pSrcImage0;
		vx_uint8 * pLocalSrc1 = pSrcImage1;
		vx_uint8 * pLocalSrc2 = pSrcImage2;
		vx_uint8 * pLocalDst = pDstImage;
		
		int width = (int) (dstWidth >> 4);
		while (width)
		{
			r = _mm_loadu_si128((__m128i *) pLocalSrc0);
			g = _mm_loadu_si128((__m128i *) pLocalSrc1);
			b = _mm_loadu_si128((__m128i *) pLocalSrc2);

			
			result1 = _mm_shuffle_epi8(r, maskR1);					// Extract and place R in first 16 bytes
			result2 = _mm_shuffle_epi8(g, maskG1);					// Extract and place G in first 16 bytes
			result3 = _mm_shuffle_epi8(b, maskB1);					// Extract and place B in first 16 bytes
			result1 = _mm_or_si128(result1, result2);
			result1 = _mm_or_si128(result1, result3);

			result2 = _mm_shuffle_epi8(r, maskR2);					// Extract and place R in second 16 bytes
			result3 = _mm_shuffle_epi8(g, maskG2);					// Extract and place G in second 16 bytes
			result2 = _mm_or_si128(result2, result3);
			result3 = _mm_shuffle_epi8(b, maskB2);					// Extract and place B in second 16 bytes
			result2 = _mm_or_si128(result2, result3);

			result3 = _mm_shuffle_epi8(r, maskR3);					// Extract and place R in third 16 bytes
			r = _mm_shuffle_epi8(g, maskG3);						// Extract and place G in third 16 bytes
			g = _mm_shuffle_epi8(b, maskB3);						// Extract and place B in third 16 bytes
			result3 = _mm_or_si128(result3, r);
			result3 = _mm_or_si128(result3, g);

			_mm_storeu_si128((__m128i *) pLocalDst, result1);
			_mm_storeu_si128((__m128i *) (pLocalDst + 16), result2);
			_mm_storeu_si128((__m128i *) (pLocalDst + 32), result3);

			width--;
			pLocalSrc0 += 16;
			pLocalSrc1 += 16;
			pLocalSrc2 += 16;
			pLocalDst += 48;
		}

		for (width = 0; width < postfixWidth; width++)
		{
			*pLocalDst++ = *pLocalSrc0++;
			*pLocalDst++ = *pLocalSrc1++;
			*pLocalDst++ = *pLocalSrc2++;
		}

		pSrcImage0 += srcImage0StrideInBytes;
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
		
		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelCombine_U32_U8U8U8_UYVY
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	)
{
	int alignedWidth = dstWidth & ~31;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataChannelCombine;
	__m128i Y0, Y1, U, V;
	__m128i maskY = _mm_load_si128(tbl + 9);
	__m128i maskU = _mm_load_si128(tbl + 10);
	__m128i maskV = _mm_load_si128(tbl + 11);
	__m128i result1, result2;

	for (int height = 0; height < (int) dstHeight; height++)
	{
		vx_uint8 * pLocalSrc0 = pSrcImage0;
		vx_uint8 * pLocalSrc1 = pSrcImage1;
		vx_uint8 * pLocalSrc2 = pSrcImage2;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 5); width++)
		{
			Y0 = _mm_loadu_si128((__m128i *) pLocalSrc0);
			Y1 = _mm_loadu_si128((__m128i *) (pLocalSrc0 + 16));
			U = _mm_loadu_si128((__m128i *) pLocalSrc1);
			V = _mm_loadu_si128((__m128i *) pLocalSrc2);

			result1 = _mm_shuffle_epi8(Y0, maskY);			// Y
			result2 = _mm_shuffle_epi8(U, maskU);			// U
			result1 = _mm_or_si128(result1, result2);		// U Y _ Y
			result2 = _mm_shuffle_epi8(V, maskV);			// V
			result1 = _mm_or_si128(result1, result2);		// U Y V Y	- first 16 bytes

			Y0 = _mm_srli_si128(Y0, 8);
			U = _mm_srli_si128(U, 4);
			V = _mm_srli_si128(V, 4);
			result2 = _mm_shuffle_epi8(Y0, maskY);			// Y
			Y0 = _mm_shuffle_epi8(U, maskU);				// U
			result2 = _mm_or_si128(result2, Y0);			// U Y _ Y
			Y0 = _mm_shuffle_epi8(V, maskV);				// V
			result2 = _mm_or_si128(result2, Y0);			// U Y V Y - next 16 bytes

			_mm_storeu_si128((__m128i *)pLocalDst, result1);
			_mm_storeu_si128((__m128i *)(pLocalDst + 16), result2);

			U = _mm_srli_si128(U, 4);
			V = _mm_srli_si128(V, 4);
			result1 = _mm_shuffle_epi8(Y1, maskY);			// Y
			result2 = _mm_shuffle_epi8(U, maskU);			// U
			result1 = _mm_or_si128(result1, result2);		// U Y _ Y
			result2 = _mm_shuffle_epi8(V, maskV);			// V
			result1 = _mm_or_si128(result1, result2);		// U Y V Y	- next 16 bytes

			Y1 = _mm_srli_si128(Y1, 8);
			U = _mm_srli_si128(U, 4);
			V = _mm_srli_si128(V, 4);
			result2 = _mm_shuffle_epi8(Y1, maskY);			// Y
			Y1 = _mm_shuffle_epi8(U, maskU);				// U
			result2 = _mm_or_si128(result2, Y1);			// U Y _ Y
			Y1 = _mm_shuffle_epi8(V, maskV);				// V
			result2 = _mm_or_si128(result2, Y1);			// U Y V Y - last 16 bytes

			_mm_storeu_si128((__m128i *)(pLocalDst + 32), result1);
			_mm_storeu_si128((__m128i *)(pLocalDst + 48), result2);

			pLocalSrc0 += 32;
			pLocalSrc1 += 16;
			pLocalSrc2 += 16;
			pLocalDst += 64;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			*pLocalDst++ = *pLocalSrc1++;			// U
			*pLocalDst++ = *pLocalSrc0++;			// Y
			*pLocalDst++ = *pLocalSrc2++;			// V
			*pLocalDst++ = *pLocalSrc0++;			// Y
		}

		pSrcImage0 += srcImage0StrideInBytes;
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelCombine_U32_U8U8U8_YUYV
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes
	)
{
	int alignedWidth = dstWidth & ~31;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i * tbl = (__m128i*) dataChannelCombine;
	__m128i Y0, Y1, U, V;
	__m128i maskY = _mm_load_si128(tbl + 12);
	__m128i maskU = _mm_load_si128(tbl + 13);
	__m128i maskV = _mm_load_si128(tbl + 14);
	__m128i result1, result2;

	for (int height = 0; height < (int)dstHeight; height++)
	{
		vx_uint8 * pLocalSrc0 = pSrcImage0;
		vx_uint8 * pLocalSrc1 = pSrcImage1;
		vx_uint8 * pLocalSrc2 = pSrcImage2;
		vx_uint8 * pLocalDst = pDstImage;

		for (int width = 0; width < (alignedWidth >> 5); width++)
		{
			Y0 = _mm_loadu_si128((__m128i *) pLocalSrc0);
			Y1 = _mm_loadu_si128((__m128i *) (pLocalSrc0 + 16));
			U = _mm_loadu_si128((__m128i *) pLocalSrc1);
			V = _mm_loadu_si128((__m128i *) pLocalSrc2);

			result1 = _mm_shuffle_epi8(Y0, maskY);			// Y
			result2 = _mm_shuffle_epi8(U, maskU);			// U
			result1 = _mm_or_si128(result1, result2);		// Y U Y _
			result2 = _mm_shuffle_epi8(V, maskV);			// V
			result1 = _mm_or_si128(result1, result2);		// Y U Y V	- first 16 bytes

			Y0 = _mm_srli_si128(Y0, 8);
			U = _mm_srli_si128(U, 4);
			V = _mm_srli_si128(V, 4);
			result2 = _mm_shuffle_epi8(Y0, maskY);			// Y
			Y0 = _mm_shuffle_epi8(U, maskU);				// U
			result2 = _mm_or_si128(result2, Y0);			// Y U Y _
			Y0 = _mm_shuffle_epi8(V, maskV);				// V
			result2 = _mm_or_si128(result2, Y0);			// Y U Y V - next 16 bytes

			_mm_storeu_si128((__m128i *)pLocalDst, result1);
			_mm_storeu_si128((__m128i *)(pLocalDst + 16), result2);

			U = _mm_srli_si128(U, 4);
			V = _mm_srli_si128(V, 4);
			result1 = _mm_shuffle_epi8(Y1, maskY);			// Y
			result2 = _mm_shuffle_epi8(U, maskU);			// U
			result1 = _mm_or_si128(result1, result2);		// Y U Y _
			result2 = _mm_shuffle_epi8(V, maskV);			// V
			result1 = _mm_or_si128(result1, result2);		// Y U Y V	- next 16 bytes

			Y1 = _mm_srli_si128(Y1, 8);
			U = _mm_srli_si128(U, 4);
			V = _mm_srli_si128(V, 4);
			result2 = _mm_shuffle_epi8(Y1, maskY);			// Y
			Y1 = _mm_shuffle_epi8(U, maskU);				// U
			result2 = _mm_or_si128(result2, Y1);			// Y U Y _
			Y1 = _mm_shuffle_epi8(V, maskV);				// V
			result2 = _mm_or_si128(result2, Y1);			// Y U Y V - last 16 bytes

			_mm_storeu_si128((__m128i *)(pLocalDst + 32), result1);
			_mm_storeu_si128((__m128i *)(pLocalDst + 48), result2);

			pLocalSrc0 += 32;
			pLocalSrc1 += 16;
			pLocalSrc2 += 16;
			pLocalDst += 64;
		}
		
		for (int width = 0; width < postfixWidth; width++)
		{
			*pLocalDst++ = *pLocalSrc0++;			// Y
			*pLocalDst++ = *pLocalSrc1++;			// U
			*pLocalDst++ = *pLocalSrc0++;			// Y
			*pLocalDst++ = *pLocalSrc2++;			// V
		}

		pSrcImage0 += srcImage0StrideInBytes;
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelCombine_U32_U8U8U8U8_RGBX
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage0,
		vx_uint32     srcImage0StrideInBytes,
		vx_uint8    * pSrcImage1,
		vx_uint32     srcImage1StrideInBytes,
		vx_uint8    * pSrcImage2,
		vx_uint32     srcImage2StrideInBytes,
		vx_uint8    * pSrcImage3,
		vx_uint32     srcImage3StrideInBytes
	)
{
	int alignedWidth = dstWidth & ~15;
	int postfixWidth = (int)dstWidth - alignedWidth;

	__m128i r, g, b, x, pixels0, pixels1, pixels2;

	int height = (int) dstHeight;

	while (height)
	{
		vx_uint8 * pLocalSrc0 = pSrcImage0;
		vx_uint8 * pLocalSrc1 = pSrcImage1;
		vx_uint8 * pLocalSrc2 = pSrcImage2;
		vx_uint8 * pLocalSrc3 = pSrcImage3;
		vx_uint8 * pLocalDst = pDstImage;

		int width = (int)(dstWidth >> 4);					// Inner loop processess 16 pixels at a time
		while (width)
		{
			r = _mm_loadu_si128((__m128i *) pLocalSrc0);
			g = _mm_loadu_si128((__m128i *) pLocalSrc1);
			b = _mm_loadu_si128((__m128i *) pLocalSrc2);
			x = _mm_loadu_si128((__m128i *) pLocalSrc3);

			pixels0 = _mm_unpacklo_epi8(r, g);				// r0 g0 r1 g1 r2 g2 r3 g3 r4 g4 r5 g5 r6 g6 r7 g7
			pixels1 = _mm_unpacklo_epi8(b, x);				// b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 b5 x5 b6 x6 b7 x7
			pixels2 = _mm_unpacklo_epi16(pixels0, pixels1);	// r0 g0 b0 x0 r1 g1 b1 x1 r2 g2 b2 x2 r3 g3 b3 x3
			_mm_storeu_si128((__m128i *)pLocalDst, pixels2);
			pLocalDst += 16;

			pixels2 = _mm_unpackhi_epi16(pixels0, pixels1);	// r4 g4 b4 x4 r5 g5 b5 x5 r6 g6 b6 x6 r7 g7 b7 x7
			_mm_storeu_si128((__m128i *)pLocalDst, pixels2);
			pLocalDst += 16;

			pixels0 = _mm_unpackhi_epi8(r, g);				// r8 g8 r9 g9 r10 g10 r11 g11 r12 g12 r13 g13 r14 g14 r15 g15
			pixels1 = _mm_unpackhi_epi8(b, x);				// b8 x8 b9 x9 b10 x10 b11 x11 b12 x12 b13 x13 b14 x14 b15 x15
			pixels2 = _mm_unpacklo_epi16(pixels0, pixels1);	// r8 g8 b8 x8 r9 g9 b9 x9 r10 g10 b10 x10 r11 g11 b11 x11
			_mm_storeu_si128((__m128i *)pLocalDst, pixels2);
			pLocalDst += 16;

			pixels2 = _mm_unpackhi_epi16(pixels0, pixels1);	// r12 g12 b12 x12 r13 g13 b13 x13 r14 g14 b14 x14 r15 g15 b15 x15
			_mm_storeu_si128((__m128i *)pLocalDst, pixels2);
			pLocalDst += 16;
			
			width--;
			pLocalSrc0 += 16;
			pLocalSrc1 += 16;
			pLocalSrc2 += 16;
			pLocalSrc3 += 16;
		}

		for (int width = 0; width < postfixWidth; width++)
		{
			*pLocalDst++ = *pLocalSrc0++;
			*pLocalDst++ = *pLocalSrc1++;
			*pLocalDst++ = *pLocalSrc2++;
			*pLocalDst++ = *pLocalSrc3++;
		}

		pSrcImage0 += srcImage0StrideInBytes;
		pSrcImage1 += srcImage1StrideInBytes;
		pSrcImage2 += srcImage2StrideInBytes;
		pSrcImage3 += srcImage3StrideInBytes;
		pDstImage += dstImageStrideInBytes;

		height--;
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8U8U8_U24
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage0,
		vx_uint8    * pDstImage1,
		vx_uint8    * pDstImage2,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	// Check for output buffer alignment
	intptr_t prealignBytes = (intptr_t(pDstImage0) & intptr_t(pDstImage1) & intptr_t(pDstImage2)) & 15;
	bool isAligned = (prealignBytes == ((intptr_t(pDstImage0) | intptr_t(pDstImage1) | intptr_t(pDstImage2)) & 15));	// True if all three buffers have the same alignment


	unsigned char *pLocalSrc, *pLocalDst0, *pLocalDst1, *pLocalDst2;
	__m128i * tbl = (__m128i *) dataChannelExtract;
	__m128i pixels0, pixels1, pixels2, pixels_R, pixels_G;

	__m128i mask_r0 = _mm_load_si128(tbl + 4);
	__m128i mask_r1 = _mm_load_si128(tbl + 5);
	__m128i mask_r2 = _mm_load_si128(tbl + 6);
	__m128i mask_g0 = _mm_load_si128(tbl + 7);
	__m128i mask_g1 = _mm_load_si128(tbl + 8);
	__m128i mask_g2 = _mm_load_si128(tbl + 9);
	__m128i mask_b0 = _mm_load_si128(tbl + 10);
	__m128i mask_b1 = _mm_load_si128(tbl + 11);
	__m128i mask_b2 = _mm_load_si128(tbl + 12);

	if (isAligned)
	{
		int prefixWidth = (int)((prealignBytes == 0) ? 0 : (16 - prealignBytes));
		int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
		int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *) pSrcImage;
			pLocalDst0 = (unsigned char *) pDstImage0;
			pLocalDst1 = (unsigned char *) pDstImage1;
			pLocalDst2 = (unsigned char *) pDstImage2;

			for (int x = 0; x < prefixWidth; x++)
			{
				*pLocalDst0++ = *pSrcImage++;
				*pLocalDst1++ = *pSrcImage++;
				*pLocalDst2++ = *pSrcImage++;
			}

			int width = (int)(alignedWidth >> 4);											// 16 bytes at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels2 = _mm_loadu_si128((__m128i *) (pLocalSrc + 32));

				pixels_R = _mm_shuffle_epi8(pixels0, mask_r0);
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels1, mask_r1));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels2, mask_r2));
				_mm_store_si128((__m128i *)pLocalDst0, pixels_R);

				pixels_G = _mm_shuffle_epi8(pixels0, mask_g0);
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels1, mask_g1));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels2, mask_g2));
				_mm_store_si128((__m128i *)pLocalDst1, pixels_G);

				pixels0 = _mm_shuffle_epi8(pixels0, mask_b0);
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels1, mask_b1));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels2, mask_b2));
				_mm_store_si128((__m128i *)pLocalDst2, pixels0);

				pLocalSrc += 48;
				pLocalDst0 += 16;
				pLocalDst1 += 16;
				pLocalDst2 += 16;
				width--;
			}

			for (int x = 0; x < postfixWidth; x++)
			{
				*pLocalDst0++ = *pSrcImage++;
				*pLocalDst1++ = *pSrcImage++;
				*pLocalDst2++ = *pSrcImage++;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage0 += dstImageStrideInBytes;
			pDstImage1 += dstImageStrideInBytes;
			pDstImage2 += dstImageStrideInBytes;
			height--;
		}
	}
	else
	{
		int postfixWidth = dstWidth & 15;
		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDst0 = (unsigned char *)pDstImage0;
			pLocalDst1 = (unsigned char *)pDstImage1;
			pLocalDst2 = (unsigned char *)pDstImage2;

			int width = (int)(dstWidth >> 4);											// 16 bytes at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels2 = _mm_loadu_si128((__m128i *) (pLocalSrc + 32));

				pixels_R = _mm_shuffle_epi8(pixels0, mask_r0);
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels1, mask_r1));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels2, mask_r2));
				_mm_storeu_si128((__m128i *)pLocalDst0, pixels_R);

				pixels_G = _mm_shuffle_epi8(pixels0, mask_g0);
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels1, mask_g1));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels2, mask_g2));
				_mm_storeu_si128((__m128i *)pLocalDst1, pixels_G);

				pixels0 = _mm_shuffle_epi8(pixels0, mask_b0);
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels1, mask_b1));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels2, mask_b2));
				_mm_storeu_si128((__m128i *)pLocalDst2, pixels0);

				pLocalSrc += 48;
				pLocalDst0 += 16;
				pLocalDst1 += 16;
				pLocalDst2 += 16;
				width--;
			}

			for (int x = 0; x < postfixWidth; x++)
			{
				*pLocalDst0++ = *pSrcImage++;
				*pLocalDst1++ = *pSrcImage++;
				*pLocalDst2++ = *pSrcImage++;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage0 += dstImageStrideInBytes;
			pDstImage1 += dstImageStrideInBytes;
			pDstImage2 += dstImageStrideInBytes;
			height--;
		}
	}
	
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8U8U8_U32
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage0,
		vx_uint8    * pDstImage1,
		vx_uint8    * pDstImage2,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	// Check for output buffer alignment
	intptr_t prealignBytes = (intptr_t(pDstImage0) & intptr_t(pDstImage1) & intptr_t(pDstImage2)) & 15;
	bool isAligned = (prealignBytes == ((intptr_t(pDstImage0) | intptr_t(pDstImage1) | intptr_t(pDstImage2)) & 15));	// True if all three buffers have the same alignment
	unsigned char *pLocalSrc, *pLocalDst0, *pLocalDst1, *pLocalDst2;
	__m128i * tbl = (__m128i *) dataChannelExtract;
	__m128i pixels0, pixels1, pixels2, pixels3, pixels_R, pixels_G;

	__m128i mask0 = _mm_load_si128(tbl + 13);
	__m128i mask1 = _mm_load_si128(tbl + 14);
	__m128i mask2 = _mm_load_si128(tbl + 15);
	__m128i mask3 = _mm_load_si128(tbl + 16);

	if (isAligned)
	{
		int prefixWidth = (int)((prealignBytes == 0) ? 0 : (16 - prealignBytes));
		int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
		int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDst0 = (unsigned char *)pDstImage0;
			pLocalDst1 = (unsigned char *)pDstImage1;
			pLocalDst2 = (unsigned char *)pDstImage2;

			for (int x = 0; x < prefixWidth; x++)
			{
				*pLocalDst0++ = *pLocalSrc++;
				*pLocalDst1++ = *pLocalSrc++;
				*pLocalDst2++ = *pLocalSrc++;
				pLocalSrc++;;
			}

			int width = (int)(alignedWidth >> 4);
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels2 = _mm_loadu_si128((__m128i *) (pLocalSrc + 32));
				pixels3 = _mm_loadu_si128((__m128i *) (pLocalSrc + 48));

				pixels_R = _mm_shuffle_epi8(pixels0, mask0);
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels1, mask1));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels2, mask2));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels3, mask3));
				_mm_store_si128((__m128i *)pLocalDst0, pixels_R);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels_G = _mm_shuffle_epi8(pixels0, mask0);
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels1, mask1));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels2, mask2));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels3, mask3));
				_mm_store_si128((__m128i *)pLocalDst1, pixels_G);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels0 = _mm_shuffle_epi8(pixels0, mask0);
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels1, mask1));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels2, mask2));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels3, mask3));
				_mm_store_si128((__m128i *)pLocalDst2, pixels0);

				pLocalSrc += 64;
				pLocalDst0 += 16;
				pLocalDst1 += 16;
				pLocalDst2 += 16;

				width--;
			}

			for (int x = 0; x < postfixWidth; x++)
			{
				*pLocalDst0++ = *pLocalSrc++;
				*pLocalDst1++ = *pLocalSrc++;
				*pLocalDst2++ = *pLocalSrc++;
				pLocalSrc++;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage0 += dstImageStrideInBytes;
			pDstImage1 += dstImageStrideInBytes;
			pDstImage2 += dstImageStrideInBytes;
			height--;
		}
	}
	else
	{
		int postfixWidth = dstWidth & 15;
		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDst0 = (unsigned char *)pDstImage0;
			pLocalDst1 = (unsigned char *)pDstImage1;
			pLocalDst2 = (unsigned char *)pDstImage2;

			int width = (int)(dstWidth >> 4);											// 16 bytes at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels2 = _mm_loadu_si128((__m128i *) (pLocalSrc + 32));
				pixels3 = _mm_loadu_si128((__m128i *) (pLocalSrc + 48));

				pixels_R = _mm_shuffle_epi8(pixels0, mask0);
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels1, mask1));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels2, mask2));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels3, mask3));
				_mm_storeu_si128((__m128i *)pLocalDst0, pixels_R);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels_G = _mm_shuffle_epi8(pixels0, mask0);
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels1, mask1));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels2, mask2));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels3, mask3));
				_mm_storeu_si128((__m128i *)pLocalDst1, pixels_G);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels0 = _mm_shuffle_epi8(pixels0, mask0);
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels1, mask1));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels2, mask2));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels3, mask3));
				_mm_storeu_si128((__m128i *)pLocalDst2, pixels0);

				pLocalSrc += 64;
				pLocalDst0 += 16;
				pLocalDst1 += 16;
				pLocalDst2 += 16;
				width--;
			}

			for (int x = 0; x < postfixWidth; x++)
			{
				*pLocalDst0++ = *pLocalSrc++;
				*pLocalDst1++ = *pLocalSrc++;
				*pLocalDst2++ = *pLocalSrc++;
				pLocalSrc++;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage0 += dstImageStrideInBytes;
			pDstImage1 += dstImageStrideInBytes;
			pDstImage2 += dstImageStrideInBytes;
			height--;
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_ChannelExtract_U8U8U8U8_U32
	(
		vx_uint32     dstWidth,
		vx_uint32     dstHeight,
		vx_uint8    * pDstImage0,
		vx_uint8    * pDstImage1,
		vx_uint8    * pDstImage2,
		vx_uint8    * pDstImage3,
		vx_uint32     dstImageStrideInBytes,
		vx_uint8    * pSrcImage,
		vx_uint32     srcImageStrideInBytes
	)
{
	// Check for output buffer alignment
	intptr_t prealignBytes = (intptr_t(pDstImage0) & intptr_t(pDstImage1) & intptr_t(pDstImage2) & intptr_t(pDstImage3)) & 15;
	bool isAligned = (prealignBytes == ((intptr_t(pDstImage0) | intptr_t(pDstImage1) | intptr_t(pDstImage2) | intptr_t(pDstImage3)) & 15));	// True if all three buffers have the same alignment

	unsigned char *pLocalSrc, *pLocalDst0, *pLocalDst1, *pLocalDst2, *pLocalDst3;
	__m128i * tbl = (__m128i *) dataChannelExtract;
	__m128i pixels0, pixels1, pixels2, pixels3, pixels_R, pixels_G, pixels_B;

	__m128i mask0 = _mm_load_si128(tbl + 13);
	__m128i mask1 = _mm_load_si128(tbl + 14);
	__m128i mask2 = _mm_load_si128(tbl + 15);
	__m128i mask3 = _mm_load_si128(tbl + 16);

	if (isAligned)
	{
		int prefixWidth = (int)((prealignBytes == 0) ? 0 : (16 - prealignBytes));
		int postfixWidth = ((int)dstWidth - prefixWidth) & 15;
		int alignedWidth = (int)dstWidth - prefixWidth - postfixWidth;

		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDst0 = (unsigned char *)pDstImage0;
			pLocalDst1 = (unsigned char *)pDstImage1;
			pLocalDst2 = (unsigned char *)pDstImage2;
			pLocalDst3 = (unsigned char *)pDstImage3;

			for (int x = 0; x < prefixWidth; x++)
			{
				*pLocalDst0++ = *pSrcImage++;
				*pLocalDst1++ = *pSrcImage++;
				*pLocalDst2++ = *pSrcImage++;
				*pLocalDst3++ = *pSrcImage++;
			}

			int width = (int)(alignedWidth >> 4);
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels2 = _mm_loadu_si128((__m128i *) (pLocalSrc + 32));
				pixels3 = _mm_loadu_si128((__m128i *) (pLocalSrc + 48));

				pixels_R = _mm_shuffle_epi8(pixels0, mask0);
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels1, mask1));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels2, mask2));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels3, mask3));
				_mm_store_si128((__m128i *)pLocalDst0, pixels_R);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels_G = _mm_shuffle_epi8(pixels0, mask0);
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels1, mask1));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels2, mask2));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels3, mask3));
				_mm_store_si128((__m128i *)pLocalDst1, pixels_G);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels_B = _mm_shuffle_epi8(pixels0, mask0);
				pixels_B = _mm_or_si128(pixels_B, _mm_shuffle_epi8(pixels1, mask1));
				pixels_B = _mm_or_si128(pixels_B, _mm_shuffle_epi8(pixels2, mask2));
				pixels_B = _mm_or_si128(pixels_B, _mm_shuffle_epi8(pixels3, mask3));
				_mm_store_si128((__m128i *)pLocalDst2, pixels_B);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels0 = _mm_shuffle_epi8(pixels0, mask0);
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels1, mask1));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels2, mask2));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels3, mask3));
				_mm_store_si128((__m128i *)pLocalDst3, pixels0);

				pLocalSrc += 64;
				pLocalDst0 += 16;
				pLocalDst1 += 16;
				pLocalDst2 += 16;
				pLocalDst3 += 16;

				width--;
			}

			for (int x = 0; x < postfixWidth; x++)
			{
				*pLocalDst0++ = *pSrcImage++;
				*pLocalDst1++ = *pSrcImage++;
				*pLocalDst2++ = *pSrcImage++;
				*pLocalDst3++ = *pSrcImage++;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage0 += dstImageStrideInBytes;
			pDstImage1 += dstImageStrideInBytes;
			pDstImage2 += dstImageStrideInBytes;
			pDstImage3 += dstImageStrideInBytes;
			height--;
		}
	}
	else
	{
		int postfixWidth = dstWidth & 15;
		int height = (int)dstHeight;
		while (height)
		{
			pLocalSrc = (unsigned char *)pSrcImage;
			pLocalDst0 = (unsigned char *)pDstImage0;
			pLocalDst1 = (unsigned char *)pDstImage1;
			pLocalDst2 = (unsigned char *)pDstImage2;
			pLocalDst3 = (unsigned char *)pDstImage3;

			int width = (int)(dstWidth >> 4);											// 16 bytes at a time
			while (width)
			{
				pixels0 = _mm_loadu_si128((__m128i *) pLocalSrc);
				pixels1 = _mm_loadu_si128((__m128i *) (pLocalSrc + 16));
				pixels2 = _mm_loadu_si128((__m128i *) (pLocalSrc + 32));
				pixels3 = _mm_loadu_si128((__m128i *) (pLocalSrc + 48));

				pixels_R = _mm_shuffle_epi8(pixels0, mask0);
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels1, mask1));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels2, mask2));
				pixels_R = _mm_or_si128(pixels_R, _mm_shuffle_epi8(pixels3, mask3));
				_mm_storeu_si128((__m128i *)pLocalDst0, pixels_R);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels_G = _mm_shuffle_epi8(pixels0, mask0);
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels1, mask1));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels2, mask2));
				pixels_G = _mm_or_si128(pixels_G, _mm_shuffle_epi8(pixels3, mask3));
				_mm_storeu_si128((__m128i *)pLocalDst1, pixels_G);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels_B = _mm_shuffle_epi8(pixels0, mask0);
				pixels_B = _mm_or_si128(pixels_B, _mm_shuffle_epi8(pixels1, mask1));
				pixels_B = _mm_or_si128(pixels_B, _mm_shuffle_epi8(pixels2, mask2));
				pixels_B = _mm_or_si128(pixels_B, _mm_shuffle_epi8(pixels3, mask3));
				_mm_storeu_si128((__m128i *)pLocalDst2, pixels_B);

				pixels0 = _mm_srli_si128(pixels0, 1);
				pixels1 = _mm_srli_si128(pixels1, 1);
				pixels2 = _mm_srli_si128(pixels2, 1);
				pixels3 = _mm_srli_si128(pixels3, 1);

				pixels0 = _mm_shuffle_epi8(pixels0, mask0);
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels1, mask1));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels2, mask2));
				pixels0 = _mm_or_si128(pixels0, _mm_shuffle_epi8(pixels3, mask3));
				_mm_storeu_si128((__m128i *)pLocalDst3, pixels0);

				pLocalSrc += 64;
				pLocalDst0 += 16;
				pLocalDst1 += 16;
				pLocalDst2 += 16;
				pLocalDst3 += 16;
				width--;
			}

			for (int x = 0; x < postfixWidth; x++)
			{
				*pLocalDst0++ = *pSrcImage++;
				*pLocalDst1++ = *pSrcImage++;
				*pLocalDst2++ = *pSrcImage++;
				*pLocalDst3++ = *pSrcImage++;
			}

			pSrcImage += srcImageStrideInBytes;
			pDstImage0 += dstImageStrideInBytes;
			pDstImage1 += dstImageStrideInBytes;
			pDstImage2 += dstImageStrideInBytes;
			pDstImage3 += dstImageStrideInBytes;
			height--;
		}
	}
	return AGO_SUCCESS;
}
