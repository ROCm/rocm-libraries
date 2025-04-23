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

//=======================================
//				Types
//=======================================
typedef signed char			S8;
typedef unsigned char		U8;
typedef unsigned short		U16;
typedef unsigned int		U32;
typedef unsigned long long	U64;
typedef signed short		S16;
typedef signed int			S32;
typedef signed long long	S64;

union DECL_ALIGN(16) XMM128
{
	__m128	f;
	__m128d	d;
	__m128i	i;
	__m64	m64[2];
	double	f64[2];
	U64		u64[2];
	S64		s64[2];
	float	f32[4];
	S32		s32[4];
	U32		u32[4];
	S16		s16[8];
	U16		u16[8];
	U8		u8[16];
	S8      s8[16];
} ATTR_ALIGN(16);

#define FP_BITS		18
#define FP_MUL		(1<<FP_BITS)
#define FP_ROUND    (1<<17)

// Image remapping primitive
/*
Remap with nearest neighbor interpolation type
The map table has 16 bit values out of which 13 bits are used for integer position and 3 bit for fractional.
Assumption: the value of 0xffff in map table corresponds to border and the border values will be substituted by 1.
The BORDER policy is not specified
*/
static const __m128i CONST_7		= _mm_set1_epi16((short) 7);
static const __m128i CONST_3		= _mm_set1_epi16((short) 3);
static const __m128i CONST_FFFF		= _mm_set1_epi16((short) 0xFFFF);
static const __m128i CONST_0000FFFF = _mm_set1_epi32((int) 0x0000FFFF);

int HafCpu_Remap_U8_U8_Nearest
(
	vx_uint32              dstWidth,
	vx_uint32              dstHeight,
	vx_uint8             * pDstImage,
	vx_uint32              dstImageStrideInBytes,
	vx_uint32              srcWidth,
	vx_uint32              srcHeight,
	vx_uint8             * pSrcImage,
	vx_uint32              srcImageStrideInBytes,
	ago_coord2d_ushort_t  * pMap,
	vx_uint32              mapStrideInBytes
)
{
	__m128i zeromask = _mm_setzero_si128();
	__m128i mapxy, mapfrac;

	const __m128i sstride = _mm_set1_epi32(srcImageStrideInBytes);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pchMap = (unsigned char *)pMap;
	vx_uint32 extra_pixels = dstWidth&3;

	while (pchDst < pchDstlast)
	{
		ago_coord2d_short_t *pMapY_X = (ago_coord2d_short_t *)pchMap;
		unsigned int *pdst = (unsigned int *)pchDst;
		unsigned int *pdstLast = pdst + (dstWidth >> 2);
		while (pdst < pdstLast)
		{
			__m128i temp0;
			// read remap table location for (x,y)
			mapxy = _mm_loadu_si128((__m128i *)pMapY_X );		// mapped table [src_y4,src_x4 .....src_y2,src_x1,src_y0,src_x0]
			// check for boundary values: will be substituted by 1.
			temp0 = _mm_cmpeq_epi16(mapxy, CONST_FFFF);
			mapxy = _mm_andnot_si128(temp0, mapxy);
			temp0 = _mm_and_si128(temp0, CONST_7);		// frac= 7 will be rounded to 1
			mapxy = _mm_or_si128(mapxy, temp0);			// combined result

			// get the fractional part for rounding
			mapfrac = _mm_and_si128(mapxy, CONST_7);
			mapxy = _mm_srli_epi16(mapxy, 3);			// mapxy is the int part.

			// check if the fractional part if >3, then round to next location
			mapfrac = _mm_cmpgt_epi16(mapfrac, CONST_3);
			mapfrac = _mm_and_si128(mapfrac, _mm_set1_epi16((short)1) );
			// add rounding
			mapxy = _mm_add_epi16(mapxy, mapfrac);

			// getPixel from src at mapxy position
			// calculate (mapxy.y*srcImageStrideInBytes + mapxy.x)
			temp0 = _mm_srli_epi32(mapxy, 16);					//[0000src_y4......0000src_y0]
			mapxy = _mm_and_si128(mapxy, CONST_0000FFFF);		// [0000src_x4......0000src_x0]
			temp0 = _mm_mullo_epi32(temp0, sstride);				// temp0 = src_y*stride;
			mapxy = _mm_add_epi32(mapxy, temp0);				// mapxy = src_y*stride + src_x;

			// read each src pixel from mapped position and copy to dst
			*pdst++ = pSrcImage[M128I(mapxy).m128i_i32[0]] | (pSrcImage[M128I(mapxy).m128i_i32[1]] << 8) |
				(pSrcImage[M128I(mapxy).m128i_i32[2]] << 16) | (pSrcImage[M128I(mapxy).m128i_i32[3]] << 24);
			pMapY_X += 4;
		}
		// process extra pixels if any
		if (extra_pixels){
			unsigned char *pd = (unsigned char *)pdst;
			for (unsigned int i = 0; i < extra_pixels; i++, pMapY_X++){
				int x = (pMapY_X->x != -1) ? (pMapY_X->x >> 3) + ((pMapY_X->x&7)>>2): 0;
				int y = (pMapY_X->y != -1) ? (pMapY_X->y >> 3) + ((pMapY_X->y&7)>>2) : 0;
				pd[i] = pSrcImage[y*srcImageStrideInBytes + x];
			}
		}
		pchDst += dstImageStrideInBytes;
		pchMap += mapStrideInBytes;
	}

	return AGO_SUCCESS;
}

/*
Remap with nearest neighbor interpolation type
The map table has 16 bit values out of which 13 bits are used for integer position and 3 bit for fractional.
Assumption: the value of 0xffff in map table corresponds to border and border.
The BORDER policy is constant
*/

int HafCpu_Remap_U8_U8_Nearest_Constant
(
	vx_uint32              dstWidth,
	vx_uint32              dstHeight,
	vx_uint8             * pDstImage,
	vx_uint32              dstImageStrideInBytes,
	vx_uint32              srcWidth,
	vx_uint32              srcHeight,
	vx_uint8             * pSrcImage,
	vx_uint32              srcImageStrideInBytes,
	ago_coord2d_ushort_t  * pMap,
	vx_uint32              mapStrideInBytes,
	vx_uint8               border
)
{
	__m128i zeromask = _mm_setzero_si128();
	__m128i mapxy, mapfrac, sstride;

	const __m128i srcb = _mm_set1_epi32((srcHeight*srcImageStrideInBytes) - 1);
	sstride = _mm_set1_epi32(srcImageStrideInBytes);
	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pchMap = (unsigned char *)pMap;
	vx_uint32 extra_pixels = dstWidth&3;

	while (pchDst < pchDstlast)
	{
		ago_coord2d_short_t *pMapY_X = (ago_coord2d_short_t *)pchMap;
		unsigned int *pdst = (unsigned int *)pchDst;
		unsigned int *pdstLast = pdst + (dstWidth>> 2);
		while (pdst < pdstLast)
		{
			__m128i temp0, temp1;
			int mask;
			// read remap table location for (x,y)
			mapxy = _mm_loadu_si128((__m128i *)pMapY_X);		// mapped table [src_y4,src_x4 .....src_y2,src_x1,src_y0,src_x0]
			// check for boundary values: will be substituted by 1.
			temp1 = _mm_cmpeq_epi16(mapxy, CONST_FFFF);
			mapxy = _mm_andnot_si128(temp1, mapxy);
			temp0 = _mm_and_si128(temp1, zeromask);
			mapxy = _mm_or_si128(mapxy, temp0);			// combined result

			// get the fractional part for rounding
			mapfrac = _mm_and_si128(mapxy, CONST_7);
			mapxy = _mm_srli_epi16(mapxy, 3);
			// check if the fractional part if >3, then round to next location
			mapfrac = _mm_cmpgt_epi16(mapfrac, CONST_3);
			mapfrac = _mm_and_si128(mapfrac, _mm_set1_epi16((short)1));
			// add rounding
			mapxy = _mm_add_epi16(mapxy, mapfrac);
			// getPixel from src at mapxy position
			// calculate (mapxy.y*srcImageStrideInBytes + mapxy.x)
			temp0 = _mm_srli_epi32(mapxy, 16);					//[0000src_y4......0000src_y0]
			mapxy = _mm_and_si128(mapxy, CONST_0000FFFF);		// [0000src_x4......0000src_x0]
			temp0 = _mm_mullo_epi32(temp0, sstride);				// temp0 = src_y*stride;
			mapxy = _mm_add_epi32(mapxy, temp0);				// mapxy = src_y*stride + src_x;
			// check if pixels exceed boundary
			temp0 = _mm_cmpgt_epi32(mapxy, srcb);
			temp1 = _mm_or_si128(temp1, temp0);
			mask  = _mm_movemask_epi8(temp1);

			// read each src pixel from mapped position and copy to dst
			if (!mask){
				*pdst++ = pSrcImage[M128I(mapxy).m128i_i32[0]] | (pSrcImage[M128I(mapxy).m128i_i32[1]] << 8) |
					(pSrcImage[M128I(mapxy).m128i_i32[2]] << 16) | (pSrcImage[M128I(mapxy).m128i_i32[3]] << 24);
			}
			else
			{
				// copy each checking for boundary
				unsigned int dstpel = (mask & 0xf) ? border : pSrcImage[M128I(mapxy).m128i_i32[0]];
				dstpel |= (mask & 0xf0) ? (border << 8) : (pSrcImage[M128I(mapxy).m128i_i32[1]] << 8);
				dstpel |= (mask & 0xf00) ? (border << 16) : (pSrcImage[M128I(mapxy).m128i_i32[2]] << 16);
				dstpel |= (mask & 0xf000) ? (border << 24) : (pSrcImage[M128I(mapxy).m128i_i32[3]] << 24);
				*pdst++ = dstpel;
			}
			pMapY_X += 4;
		}
		// process extra pixels if any
		if (extra_pixels){
			unsigned char *pd = (unsigned char *)pdst;
			for (unsigned int i = 0; i < extra_pixels; i++, pMapY_X++){
				int x = (pMapY_X->x != -1) ? (pMapY_X->x >> 3) + ((pMapY_X->x & 7) >> 2) : border;
				int y = (pMapY_X->y != -1) ? (pMapY_X->y >> 3) + ((pMapY_X->y & 7) >> 2) : border;
				pd[i] = pSrcImage[y*srcImageStrideInBytes + x];
			}
		}
		pchDst += dstImageStrideInBytes;
		pchMap += mapStrideInBytes;
	}

	return AGO_SUCCESS;
}

int HafCpu_Remap_U8_U8_Bilinear
(
	vx_uint32              dstWidth,
	vx_uint32              dstHeight,
	vx_uint8             * pDstImage,
	vx_uint32              dstImageStrideInBytes,
	vx_uint32              srcWidth,
	vx_uint32              srcHeight,
	vx_uint8             * pSrcImage,
	vx_uint32              srcImageStrideInBytes,
	ago_coord2d_ushort_t  * pMap,
	vx_uint32              mapStrideInBytes
)
{
	__m128i zeromask = _mm_setzero_si128();
	__m128i mapxy, mapfrac;

	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pchMap = (unsigned char *)pMap;
	const __m128i sstride = _mm_set1_epi32(srcImageStrideInBytes);
	const __m128i round = _mm_set1_epi32((int)32);

	while (pchDst < pchDstlast)
	{
		ago_coord2d_short_t *pMapY_X = (ago_coord2d_short_t *)pchMap;
		unsigned int *pdst = (unsigned int *)pchDst;
		unsigned int *pdstLast = pdst + ((dstWidth+3) >> 2);
		while (pdst < pdstLast)
		{
			__m128i temp0, temp1, w_xy, oneminusxy, p12, p34;
			unsigned char *p0;
			// read remap table location for (x,y)
			mapxy = _mm_loadu_si128((__m128i *)pMapY_X);		// mapped table [src_y3,src_x3 .....src_y2,src_x1,src_y0,src_x0]
			// check for boundary values: will be substituted by 1.
			temp0 = _mm_cmpeq_epi16(mapxy, CONST_FFFF);
			mapxy = _mm_andnot_si128(temp0, mapxy);
			temp0 = _mm_and_si128(temp0, _mm_set1_epi16(0x8));
			mapxy = _mm_or_si128(mapxy, temp0);			// combined result

			// get the fractional part for rounding
			mapfrac = _mm_and_si128(mapxy, CONST_7);					// [dy3, dx3.........dy0, dx0]
			oneminusxy = _mm_sub_epi16(_mm_set1_epi16(8), mapfrac);		// [1-dy3, 1-dx3........1-dy0, 1-dx0]
			mapxy = _mm_srli_epi16(mapxy, 3);							// [y3, x3.............y0, x0]
			// calculate (mapxy.y*srcImageStrideInBytes + mapxy.x)
			temp0 = _mm_srli_epi32(mapxy, 16);					//[0000src_y4......0000src_y0]
			mapxy = _mm_and_si128(mapxy, CONST_0000FFFF);		// [0000src_x4......0000src_x0]
			temp0 = _mm_mullo_epi32(temp0, sstride);				// temp0 = src_y*stride;
			mapxy = _mm_add_epi32(mapxy, temp0);				// mapxy = src_y*stride + src_x;

			// load the pixels 2 pixels in one load
			p0 = &pSrcImage[M128I(mapxy).m128i_i32[0]];
			p12 = _mm_cvtsi32_si128(((unsigned int *)p0)[0]);
			p34 = _mm_cvtsi32_si128(((unsigned int *)(p0 + srcImageStrideInBytes))[0]);
			temp0 = _mm_unpacklo_epi16(oneminusxy, mapfrac);			// [dy1, 1-dy1, dx1, 1-dx1, dy0, 1-dy0, dx0, 1-dx0]
			temp1 = _mm_unpacklo_epi32(temp0, temp0);					// [dy0, 1-dy0, dy0, 1-dy0, dx0, 1-dx0, dx0, 1-dx0]
			temp0 = _mm_unpackhi_epi32(temp0, temp0);					// [dy1, 1-dy1, dy1, 1-dy1, dx1, 1-dx1, dx1, 1-dx1]

			w_xy	= _mm_unpacklo_epi64(temp1, temp0);					// [dx1, 1-dx1, dx1, 1-dx1 dx0, 1-dx0, dx0, 1-dx0]
			temp1 = _mm_unpackhi_epi64(temp1, temp0);					// [dy1, 1-dy1, dy1, 1-dy1, dy0, 1-dy0, dy0, 1-dy0]
			temp1 = _mm_shufflelo_epi16(temp1, 0xd8);
			temp1 = _mm_shufflehi_epi16(temp1, 0xd8);					// [dy1, dy1, 1-dy1, 1-dy1, dy0, dy0, 1-dy0, 1-dy0]
			// calculate weight 
			w_xy = _mm_mullo_epi16(w_xy, temp1);						// [w3, w2, w1, w0]	// for 2 
			p12 = _mm_unpacklo_epi16(p12, p34);
			p0 = &pSrcImage[M128I(mapxy).m128i_i32[1]];
			temp0 = _mm_cvtsi32_si128(((unsigned int *)p0)[0]);
			p34 = _mm_cvtsi32_si128(((unsigned int *)(p0 + srcImageStrideInBytes))[0]);
			temp0 = _mm_unpacklo_epi16(temp0, p34);
//			w_xy = _mm_srli_epi16(w_xy, 6);
			p12 = _mm_unpacklo_epi32(p12, temp0);
			p12 = _mm_unpacklo_epi8(p12, zeromask);				// [p2, p2, p1, p0] for 2

			// multiply add with weight
			p12 = _mm_madd_epi16(p12, w_xy);			// (w3p3+w2p2),(w0p0+w1p1) for 2
			p34 = _mm_hadd_epi32(p12, p12);				// dst 0 and 1

			// do computation for dst 2 and 3
			temp0 = _mm_unpackhi_epi16(oneminusxy, mapfrac);			// [dy3, 1-dy3, dx3, 1-dx3, dy2, 1-dy2, dx2, 1-dx2]
			temp1 = _mm_unpacklo_epi32(temp0, temp0);					// [dy2, 1-dy2, dy2, 1-dy2, dx2, 1-dx2, dx2, 1-dx2]
			temp0 = _mm_unpackhi_epi32(temp0, temp0);					// [dy3, 1-dy3, dy3, 1-dy3, dx3, 1-dx3, dx3, 1-dx3]
			w_xy = _mm_unpacklo_epi64(temp1, temp0);					// [dx3, 1-dx3, dx3, 1-dx3, dx2, 1-dx2, dx2, 1-dx2]
			temp1 = _mm_unpackhi_epi64(temp1, temp0);					// [dy3, 1-dy3, dy3, 1-dy3, dy2, 1-dy2, dy2, 1-dy2]
			temp1 = _mm_shufflelo_epi16(temp1, 0xd8);
			temp1 = _mm_shufflehi_epi16(temp1, 0xd8);					// [dy1, dy1, 1-dy1, 1-dy1, dy0, dy0, 1-dy0, 1-dy0]
			// calculate weight 
			w_xy = _mm_mullo_epi16(w_xy, temp1);						// [w3, w2, w1, w0]	// for 2 and 3 
			p0 = &pSrcImage[M128I(mapxy).m128i_i32[2]];
			p12 = _mm_cvtsi32_si128(((unsigned int *)p0)[0]);
			temp0 = _mm_cvtsi32_si128(((unsigned int *)(p0 + srcImageStrideInBytes))[0]);
			p12 = _mm_unpacklo_epi16(p12, temp0);
			p0 = &pSrcImage[M128I(mapxy).m128i_i32[3]];
			temp0 = _mm_cvtsi32_si128(((unsigned int *)p0)[0]);
			temp1 = _mm_cvtsi32_si128(((unsigned int *)(p0 + srcImageStrideInBytes))[0]);
			//w_xy = _mm_srli_epi16(w_xy, 6);
			temp0 = _mm_unpacklo_epi16(temp0, temp1);
			p12 = _mm_unpacklo_epi32(p12, temp0);

			p12 = _mm_unpacklo_epi8(p12, zeromask);				// [p2, p2, p1, p0] for 2

			// multiply add with weight
			p12 = _mm_madd_epi16(p12, w_xy);			// (w3p3+w2p2),(w0p0+w1p1) for 2
			temp0 = _mm_hadd_epi32(p12, p12);				// dst 2 and 3

			p34 = _mm_unpacklo_epi64(p34, temp0);
			p34 = _mm_add_epi32(p34, round);
			p34 = _mm_srli_epi32(p34, 6);
			// convert 32 bit to 8 bit
			p34 = _mm_packus_epi32(p34, zeromask);
			p34 = _mm_packus_epi16(p34, zeromask);

			// read each src pixel from mapped position and copy to dst
			*pdst++ = M128I(p34).m128i_i32[0];
			pMapY_X += 4;
		}
		pchDst += dstImageStrideInBytes;
		pchMap += mapStrideInBytes;

	}

	return AGO_SUCCESS;
}

int HafCpu_Remap_U8_U8_Bilinear_Constant
(
	vx_uint32              dstWidth,
	vx_uint32              dstHeight,
	vx_uint8             * pDstImage,
	vx_uint32              dstImageStrideInBytes,
	vx_uint32              srcWidth,
	vx_uint32              srcHeight,
	vx_uint8             * pSrcImage,
	vx_uint32              srcImageStrideInBytes,
	ago_coord2d_ushort_t  * pMap,
	vx_uint32              mapStrideInBytes,
	vx_uint8               border
)
{
	__m128i zeromask = _mm_setzero_si128();
	__m128i mapxy, mapfrac;

	const __m128i sstride = _mm_set1_epi32(srcImageStrideInBytes);
	const __m128i pborder = _mm_set1_epi32(border);
	const __m128i round = _mm_set1_epi32((int)32);

	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	unsigned char *pchMap = (unsigned char *)pMap;

	while (pchDst < pchDstlast)
	{
		ago_coord2d_short_t *pMapY_X = (ago_coord2d_short_t *)pchMap;
		unsigned int *pdst = (unsigned int *)pchDst;
		unsigned int *pdstLast = pdst + ((dstWidth+3) >> 2);
		while (pdst < pdstLast)
		{
			__m128i temp0, temp1, w_xy, oneminusxy, p12, p34, mask;
			unsigned char *p0;
			// read remap table location for (x,y)
			mapxy = _mm_loadu_si128((__m128i *)pMapY_X);		// mapped table [src_y3,src_x3 .....src_y2,src_x1,src_y0,src_x0]
			// check for boundary values: will be substituted by border.
			mask = _mm_cmpeq_epi16(mapxy, CONST_FFFF);
			mapxy = _mm_andnot_si128(mask, mapxy);
			temp0 = _mm_and_si128(mask, zeromask);
			mapxy = _mm_or_si128(mapxy, temp0);			// combined result
			//mask = _mm_movemask_epi8(temp1);

			// get the fractional part for rounding
			// get the fractional part for rounding
			mapfrac = _mm_and_si128(mapxy, CONST_7);					// [dy3, dx3.........dy0, dx0]
			oneminusxy = _mm_sub_epi16(_mm_set1_epi16(8), mapfrac);		// [1-dy3, 1-dx3........1-dy0, 1-dx0]
			mapxy = _mm_srli_epi16(mapxy, 3);							// [y3, x3.............y0, x0]
			// calculate (mapxy.y*srcImageStrideInBytes + mapxy.x)
			temp0 = _mm_srli_epi32(mapxy, 16);					//[0000src_y4......0000src_y0]
			mapxy = _mm_and_si128(mapxy, CONST_0000FFFF);		// [0000src_x4......0000src_x0]
			temp0 = _mm_mullo_epi32(temp0, sstride);				// temp0 = src_y*stride;
			mapxy = _mm_add_epi32(mapxy, temp0);				// mapxy = src_y*stride + src_x;

				// load the pixels 2 pixels in one load
			p0 = &pSrcImage[M128I(mapxy).m128i_i32[0]];
			p12 = _mm_cvtsi32_si128(((unsigned int *)p0)[0]);
			p34 = _mm_cvtsi32_si128(((unsigned int *)(p0 + srcImageStrideInBytes))[0]);
			temp0 = _mm_unpacklo_epi16(oneminusxy, mapfrac);			// [dy1, 1-dy1, dx1, 1-dx1, dy0, 1-dy0, dx0, 1-dx0]
			temp1 = _mm_unpacklo_epi32(temp0, temp0);					// [dy0, 1-dy0, dy0, 1-dy0, dx0, 1-dx0, dx0, 1-dx0]
			temp0 = _mm_unpackhi_epi32(temp0, temp0);					// [dy1, 1-dy1, dy1, 1-dy1, dx1, 1-dx1, dx1, 1-dx1]

			w_xy = _mm_unpacklo_epi64(temp1, temp0);					// [dx1, 1-dx1, dx1, 1-dx1 dx0, 1-dx0, dx0, 1-dx0]
			temp1 = _mm_unpackhi_epi64(temp1, temp0);					// [dy1, 1-dy1, dy1, 1-dy1, dy0, 1-dy0, dy0, 1-dy0]
			temp1 = _mm_shufflelo_epi16(temp1, 0xd8);
			temp1 = _mm_shufflehi_epi16(temp1, 0xd8);					// [dy1, dy1, 1-dy1, 1-dy1, dy0, dy0, 1-dy0, 1-dy0]

			// calculate weight 
			w_xy = _mm_mullo_epi16(w_xy, temp1);						// [w3, w2, w1, w0]	// for 2 
			p12 = _mm_unpacklo_epi16(p12, p34);
			p0 = &pSrcImage[M128I(mapxy).m128i_i32[1]];
			temp0 = _mm_cvtsi32_si128(((unsigned int *)p0)[0]);
			p34 = _mm_cvtsi32_si128(((unsigned int *)(p0 + srcImageStrideInBytes))[0]);
			temp0 = _mm_unpacklo_epi16(temp0, p34);
			p12 = _mm_unpacklo_epi32(p12, temp0);
			p12 = _mm_unpacklo_epi8(p12, zeromask);				// [p3, p2, p1, p0] for 2

			// multiply add with weight
			p12 = _mm_madd_epi16(p12, w_xy);			// (w3p3+w2p2),(w0p0+w1p1) for 2
			p34 = _mm_hadd_epi32(p12, p12);				// dst 0 and 1

			// do computation for dst 2 and 3
			temp0 = _mm_unpackhi_epi16(oneminusxy, mapfrac);			// [dy3, 1-dy3, dx3, 1-dx3, dy2, 1-dy2, dx2, 1-dx2]
			temp1 = _mm_unpacklo_epi32(temp0, temp0);					// [dy2, 1-dy2, dy2, 1-dy2, dx2, 1-dx2, dx2, 1-dx2]
			temp0 = _mm_unpackhi_epi32(temp0, temp0);					// [dy3, 1-dy3, dy3, 1-dy3, dx3, 1-dx3, dx3, 1-dx3]
			w_xy = _mm_unpacklo_epi64(temp1, temp0);					// [dx3, 1-dx3, dx3, 1-dx3, dx2, 1-dx2, dx2, 1-dx2]
			temp1 = _mm_unpackhi_epi64(temp1, temp0);					// [dy3, 1-dy3, dy3, 1-dy3, dy2, 1-dy2, dy2, 1-dy2]
			temp1 = _mm_shufflelo_epi16(temp1, 0xd8);
			temp1 = _mm_shufflehi_epi16(temp1, 0xd8);					// [dy3, dy3, 1-dy3, 1-dy3, dy0, dy2, 1-dy2, 1-dy2]

			// calculate weight 
			w_xy = _mm_mullo_epi16(w_xy, temp1);						// [w3, w2, w1, w0]	// for 2 and 3 
			p0 = &pSrcImage[M128I(mapxy).m128i_i32[2]];
			p12 = _mm_cvtsi32_si128(((unsigned int *)p0)[0]);
			temp0 = _mm_cvtsi32_si128(((unsigned int *)(p0 + srcImageStrideInBytes))[0]);
			p12 = _mm_unpacklo_epi16(p12, temp0);
			p0 = &pSrcImage[M128I(mapxy).m128i_i32[3]];
			temp0 = _mm_cvtsi32_si128(((unsigned int *)p0)[0]);
			temp1 = _mm_cvtsi32_si128(((unsigned int *)(p0 + srcImageStrideInBytes))[0]);
			temp0 = _mm_unpacklo_epi16(temp0, temp1);
			p12 = _mm_unpacklo_epi32(p12, temp0);
			//w_xy = _mm_shuffle_epi32(w_xy, 0x4e);
			p12 = _mm_unpacklo_epi8(p12, zeromask);				// [p3, p2, p1, p0] for 2

			// multiply add with weight
			p12 = _mm_madd_epi16(p12, w_xy);			// (w3p3+w2p2),(w0p0+w1p1) for 2
			temp0 = _mm_hadd_epi32(p12, p12);				// dst 0 and 1
			//p34 = _mm_shuffle_epi32(p34, 0xd8);

			//temp0 = _mm_shuffle_epi32(temp0, 0xd8);
			p34 = _mm_unpacklo_epi64(p34, temp0);
			p34 = _mm_add_epi32(p34, round);
			p34 = _mm_srli_epi32(p34, 6);

			p34 = _mm_andnot_si128(mask, p34);
			mask = _mm_and_si128(mask, pborder);
			p34 = _mm_or_si128(p34, mask);			// combined result
			// convert 32 bit to 8 bit
			p34 = _mm_packus_epi32(p34, zeromask);
			p34 = _mm_packus_epi16(p34, zeromask);

			// read each src pixel from mapped position and copy to dst
			*pdst++ = M128I(p34).m128i_i32[0];

			pMapY_X += 4;
		}
		pchDst += dstImageStrideInBytes;
		pchMap += mapStrideInBytes;
	}

	return AGO_SUCCESS;
}

// The dst pixels are nearest affine transformed (truncate towards zero rounding). Bounday_mode is not specified. 
// If the transformed location is out of bounds: 0 or max pixel will be used as substitution.
int HafCpu_WarpAffine_U8_U8_Nearest
(
vx_uint32             dstWidth,
vx_uint32             dstHeight,
vx_uint8            * pDstImage,
vx_uint32             dstImageStrideInBytes,
vx_uint32             srcWidth,
vx_uint32             srcHeight,
vx_uint8            * pSrcImage,
vx_uint32             srcImageStrideInBytes,
ago_affine_matrix_t * matrix,
vx_uint8			* pLocalData
)
{
	__m128  ymap, xmap, ydest, xdest;
	__m128i srcb, src_s;
	__m128i zeromask = _mm_setzero_si128();

	const float r00 = matrix->matrix[0][0];
	const float r10 = matrix->matrix[0][1];
	const float r01 = matrix->matrix[1][0];
	const float r11 = matrix->matrix[1][1];
	const float const1 = matrix->matrix[2][0];
	const float const2 = matrix->matrix[2][1];

	const __m128 srcbx = _mm_set1_ps((float)srcWidth);
	const __m128 srcby = _mm_set1_ps((float)srcHeight);
	const __m128 zero = _mm_set1_ps(0);
	srcb = _mm_set1_epi32((srcHeight*srcImageStrideInBytes) - 1);
	src_s = _mm_set1_epi32(srcImageStrideInBytes);

	// check if all mapped pixels are valid or not
	bool bBoder = (const1 < 0) | (const2 < 0) | (const1 >= srcWidth) | (const2 >= srcHeight);
	// check for (dstWidth, 0)
	float x1 = (r00*dstWidth + const1);
	float y1 = (r10*dstWidth + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
	// check for (0, dstHeight)
	x1 = (r01*dstHeight + const1);
	y1 = (r11*dstHeight + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
	// check for (dstWidth, dstHeight)
	x1 = (r00*dstWidth + r01*dstHeight + const1);
	y1 = (r10*dstWidth + r11*dstHeight + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);

	XMM128 mask;
	unsigned int x, y;
	float *r00_x, *r10_x;
	r00_x = (float*)pLocalData;
	r10_x = r00_x + dstWidth;
	for (x = 0; x<dstWidth; x++){
		r00_x[x] = r00 * x;
		r10_x[x] = r10 * x;
	}
	y = 0;

	if (bBoder){
		while (y < dstHeight)
		{
			// calculate (y*m[0][1] + m[0][2]) for x and y
			xdest = _mm_set1_ps(y*r01 + const1);
			ydest = _mm_set1_ps(y*r11 + const2);

			unsigned int x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128i xpels, ypels;
				// read x into xpel
				xmap = _mm_load_ps(&r00_x[x]);
				xmap = _mm_add_ps(xmap, xdest);				// xf = dst[x3, x2, x1, x0]
				ymap = _mm_load_ps(&r10_x[x]);
				ymap = _mm_add_ps(ymap, ydest);				// ymap <- r10*x + ty

				mask.f = _mm_cmpge_ps(xmap, zero);
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(xmap, srcbx));
				mask.f = _mm_and_ps(mask.f, _mm_cmpge_ps(ymap, zero));
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(ymap, srcby));

				// convert to integer with rounding towards zero
				xpels = _mm_cvttps_epi32(xmap);
				ypels = _mm_cvttps_epi32(ymap);

				// multiply ydest*srcImageStrideInBytes
				ypels = _mm_mullo_epi32(ypels, src_s);
				ypels = _mm_add_epi32(ypels, xpels);			// pixel location at src for dst image.

				// check if the values exceed boundary and clamp it to boundary :: need to do this to avoid memory access violations
				ypels = _mm_min_epi32(ypels, srcb);
				ypels = _mm_max_epi32(ypels, zeromask);

				// check if the values exceed boundary and clamp it to boundary
				xpels = _mm_set_epi32(pSrcImage[M128I(ypels).m128i_i32[3]], pSrcImage[M128I(ypels).m128i_i32[2]], pSrcImage[M128I(ypels).m128i_i32[1]], pSrcImage[M128I(ypels).m128i_i32[0]]);
				// mask for boundary: boundary pixels  will substituted with xero
				xpels = _mm_and_si128(xpels, mask.i);

				// convert to unsigned char and write to dst
				xpels = _mm_packus_epi32(xpels, zeromask);
				xpels = _mm_packus_epi16(xpels, zeromask);
				*dst++ = M128I(xpels).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		while (y < dstHeight)
		{
			unsigned int x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			// calculate (y*m[0][1] + m[0][2]) for x and y
			xdest = _mm_set1_ps(y*r01 + const1);
			ydest = _mm_set1_ps(y*r11 + const2);
			while (x < dstWidth)
			{
				__m128i xpels, ypels;
				// read x into xpel
				xmap = _mm_load_ps(&r00_x[x]);
				xmap = _mm_add_ps(xmap, xdest);				// xf = dst[x3, x2, x1, x0]
				ymap = _mm_load_ps(&r10_x[x]);
				ymap = _mm_add_ps(ymap, ydest);				// ymap <- r10*x + ty

				// convert to integer with rounding towards zero
				xpels = _mm_cvttps_epi32(xmap);
				ypels = _mm_cvttps_epi32(ymap);
				// multiply ydest*srcImageStrideInBytes
				ypels = _mm_mullo_epi32(ypels, src_s);
				ypels = _mm_add_epi32(ypels, xpels);			// pixel location at src for dst image.

				// check if the values exceed boundary and clamp it to boundary
				xpels = _mm_set_epi32(pSrcImage[M128I(ypels).m128i_i32[3]], pSrcImage[M128I(ypels).m128i_i32[2]], pSrcImage[M128I(ypels).m128i_i32[1]], pSrcImage[M128I(ypels).m128i_i32[0]]);
				// convert to unsigned char and write to dst
				xpels = _mm_packus_epi32(xpels, zeromask);
				xpels = _mm_packus_epi16(xpels, zeromask);
				*dst++ = M128I(xpels).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}


// The dst pixels are nearest affine transformed (truncate towards zero rounding). Bounday_mode is not specified. 
// If the transformed location is out of bounds: border has to be substituted.
int HafCpu_WarpAffine_U8_U8_Nearest_Constant
(
	vx_uint32             dstWidth,
	vx_uint32             dstHeight,
	vx_uint8            * pDstImage,
	vx_uint32             dstImageStrideInBytes,
	vx_uint32             srcWidth,
	vx_uint32             srcHeight,
	vx_uint8            * pSrcImage,
	vx_uint32             srcImageStrideInBytes,
	ago_affine_matrix_t * matrix,
	vx_uint8              border,
	vx_rectangle_t        rect_valid,
	vx_uint8			* pLocalData
)
{
	__m128  ymap, xmap, ydest, xdest;
	__m128i pborder, srcb, src_s;
	__m128i zeromask = _mm_setzero_si128();
	const unsigned int u32_border = border | (border << 8) | (border << 16) | (border << 24);

	const float r00 = matrix->matrix[0][0];
	const float r10 = matrix->matrix[0][1];
	const float r01 = matrix->matrix[1][0];
	const float r11 = matrix->matrix[1][1];
	const float const1 = matrix->matrix[2][0];
	const float const2 = matrix->matrix[2][1];

	const __m128 srcbx = _mm_set1_ps((float)srcWidth);
	const __m128 srcby = _mm_set1_ps((float)srcHeight);
	const __m128 zero = _mm_set1_ps(0);

	const __m128 vl = _mm_set1_ps((float)rect_valid.start_x);
	const __m128 vr = _mm_set1_ps((float)rect_valid.end_x);
	const __m128 vt = _mm_set1_ps((float)rect_valid.start_y);
	const __m128 vb = _mm_set1_ps((float)rect_valid.end_y);

	srcb = _mm_set1_epi32((srcHeight*srcImageStrideInBytes) - 1);
	src_s = _mm_set1_epi32(srcImageStrideInBytes);
	pborder = _mm_cvtsi32_si128((int)border);
	pborder = _mm_shuffle_epi32(pborder, 0);
	// check if all mapped pixels are valid or not
	bool bBoder = (const1 < 0) | (const2 < 0) | (const1 >= srcWidth) | (const2 >= srcHeight);
	// check for (dstWidth, 0)
	float x1 = (r00*dstWidth + const1);
	float y1 = (r10*dstWidth + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
	// check for (0, dstHeight)
	x1 = (r01*dstHeight + const1);
	y1 = (r11*dstHeight + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
	// check for (dstWidth, dstHeight)
	x1 = (r00*dstWidth + r01*dstHeight + const1);
	y1 = (r10*dstWidth + r11*dstHeight + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);

	XMM128 mask;
	unsigned int x, y;
	float *r00_x = (float*)pLocalData;
	float *r10_x = r00_x + dstWidth;
	for (x = 0; x<dstWidth; x++){
		r00_x[x] = r00 * x;
		r10_x[x] = r10 * x;
	}
	y = 0;

	if (bBoder){
		while (y < dstHeight)
		{
			// calculate (y*m[0][1] + m[0][2]) for x and y
			xdest = _mm_set1_ps(y*r01 + const1);
			ydest = _mm_set1_ps(y*r11 + const2);

			unsigned int x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128i xpels, ypels;
				// read x into xpel
				xmap = _mm_load_ps(&r00_x[x]);
				xmap = _mm_add_ps(xmap, xdest);				// xf = dst[x3, x2, x1, x0]
				ymap = _mm_load_ps(&r10_x[x]);
				ymap = _mm_add_ps(ymap, ydest);				// ymap <- r10*x + ty

				mask.f = _mm_cmpge_ps(xmap, zero);
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(xmap, srcbx));
				mask.f = _mm_and_ps(mask.f, _mm_cmpge_ps(ymap, zero));
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(ymap, srcby));
				
				// mask for valid region rectangle
				mask.f = _mm_and_ps(mask.f, _mm_cmpge_ps(xmap, vl));
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(xmap, vr));
				mask.f = _mm_and_ps(mask.f, _mm_cmpge_ps(ymap, vt));
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(ymap, vb));

				//int m = _mm_movemask_ps(mask.f);
				//if (m){
					// convert to integer with rounding towards zero
					xpels = _mm_cvttps_epi32(xmap);
					ypels = _mm_cvttps_epi32(ymap);
					// multiply ydest*srcImageStrideInBytes
					ypels = _mm_mullo_epi32(ypels, src_s);
					ypels = _mm_add_epi32(ypels, xpels);			// pixel location at src for dst image.

					// check if the values exceed boundary and clamp it to boundary :: need to do this to avoid memory access violations
					ypels = _mm_min_epi32(ypels, srcb);
					ypels = _mm_max_epi32(ypels, zeromask);

					// check if the values exceed boundary and clamp it to boundary
					xpels = _mm_set_epi32(pSrcImage[M128I(ypels).m128i_i32[3]], pSrcImage[M128I(ypels).m128i_i32[2]], pSrcImage[M128I(ypels).m128i_i32[1]], pSrcImage[M128I(ypels).m128i_i32[0]]);
					// mask for boundary
					xpels = _mm_and_si128(xpels, mask.i);
					xpels = _mm_or_si128(xpels, _mm_andnot_si128(mask.i, pborder));			// combined result

					// convert to unsigned char and write to dst
					xpels = _mm_packus_epi32(xpels, zeromask);
					xpels = _mm_packus_epi16(xpels, zeromask);
					*dst++ = M128I(xpels).m128i_i32[0];
				//}
				//else
				//{
				//	*dst++ = u32_border;
				//}
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		while (y < dstHeight)
		{
			// calculate (y*m[0][1] + m[0][2]) for x and y
			xdest = _mm_set1_ps(y*r01 + const1);
			ydest = _mm_set1_ps(y*r11 + const2);

			unsigned int x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128i xpels, ypels;
				// read x into xpel
				xmap = _mm_load_ps(&r00_x[x]);
				xmap = _mm_add_ps(xmap, xdest);				// xf = dst[x3, x2, x1, x0]
				ymap = _mm_load_ps(&r10_x[x]);
				ymap = _mm_add_ps(ymap, ydest);				// ymap <- r10*x + ty
				// convert to integer with rounding towards zero
				xpels = _mm_cvttps_epi32(xmap);
				ypels = _mm_cvttps_epi32(ymap);
				// multiply ydest*srcImageStrideInBytes
				ypels = _mm_mullo_epi32(ypels, src_s);
				ypels = _mm_add_epi32(ypels, xpels);			// pixel location at src for dst image.
				xpels = _mm_set_epi32(pSrcImage[M128I(ypels).m128i_i32[3]], pSrcImage[M128I(ypels).m128i_i32[2]], pSrcImage[M128I(ypels).m128i_i32[1]], pSrcImage[M128I(ypels).m128i_i32[0]]);
				// convert to unsigned char and write to dst
				xpels = _mm_packus_epi32(xpels, zeromask);
				xpels = _mm_packus_epi16(xpels, zeromask);
				*dst++ = M128I(xpels).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_WarpAffine_U8_U8_Bilinear
(
	vx_uint32             dstWidth,
	vx_uint32             dstHeight,
	vx_uint8            * pDstImage,
	vx_uint32             dstImageStrideInBytes,
	vx_uint32             srcWidth,
	vx_uint32             srcHeight,
	vx_uint8            * pSrcImage,
	vx_uint32             srcImageStrideInBytes,
	ago_affine_matrix_t * matrix,
	vx_uint8			* pLocalData
)
{
	// call the HafCpu_WarpAffine_U8_U8_Bilinear_Constant with border value 128
	__m128  ymap, xmap, ydest, xdest;
	__m128i srcb, src_s;
	const __m128i zeromask = _mm_setzero_si128();
	const __m128i one = _mm_set1_epi32(1);

	// do backward mapping to find the (x, y) locations in source corresponding to (x', y') from dest by doing inverse matrix
	const float r00 = matrix->matrix[0][0];
	const float r10 = matrix->matrix[0][1];
	const float r01 = matrix->matrix[1][0];
	const float r11 = matrix->matrix[1][1];
	const float const1 = matrix->matrix[2][0];
	const float const2 = matrix->matrix[2][1];

	const __m128 zero = _mm_set1_ps(0);
	const __m128i srcbx_i = _mm_set1_epi32(srcWidth);
	const __m128i srcby_i = _mm_set1_epi32(srcHeight);
	const __m128 srcbx = _mm_cvtepi32_ps(srcbx_i);
	const __m128 srcby = _mm_cvtepi32_ps(srcby_i);

	const __m128i p0mask = _mm_set1_epi32((int)0xFF);
	const __m128 oneFloat = _mm_set1_ps(1.0);
	srcb = _mm_set1_epi32((srcHeight*srcImageStrideInBytes) - 1);
	src_s = _mm_set1_epi32(srcImageStrideInBytes);

	XMM128 mask;
	unsigned int x, y;
	float *r00_x = (float*)pLocalData;
	float *r10_x = (float *)ALIGN16(r00_x + dstWidth);
	for (x = 0; x<dstWidth; x++){
		r00_x[x] = r00 * x;
		r10_x[x] = r10 * x;
	}
	bool bBoder = (const1 < 0) | (const2 < 0) | (const1 >= srcWidth) | (const2 >= srcHeight);
	// check for (dstWidth, 0)
	float x1 = (r00*dstWidth  + const1);
	float y1 = (r10*dstWidth  + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
	// check for (0, dstHeight)
	x1 = (r01*dstHeight + const1);
	y1 = (r11*dstHeight + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
	// check for (dstWidth, dstHeight)
	x1 = (r00*dstWidth + r01*dstHeight + const1);
	y1 = (r10*dstWidth + r11*dstHeight + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);

	y = 0;
	if (bBoder){
		__m128i srcb = _mm_set1_epi32((srcHeight-1)*srcImageStrideInBytes - 1);
		__m128i src_s = _mm_set1_epi32(srcImageStrideInBytes);

		while (y < dstHeight)
		{
			// calculate (y*m[0][1] + m[0][2]) for x and y
			xdest = _mm_set1_ps(y*r01 + const1);
			ydest = _mm_set1_ps(y*r11 + const2);

			x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128 xFraction, yFraction, one_minus_xFraction, one_minus_yFraction;
				__m128 p0_f, p1_f, p2_f, p3_f;
				__m128i p0, p1, p2, p3, xint, yint;			// pixels in src 
				unsigned char *psrc;

				// read x into xpel
				xmap = _mm_load_ps(&r00_x[x]);
				xmap = _mm_add_ps(xmap, xdest);				// xf = dst[x3, x2, x1, x0]
				ymap = _mm_load_ps(&r10_x[x]);
				ymap = _mm_add_ps(ymap, ydest);				// ymap <- r10*x + ty

				mask.f = _mm_cmpge_ps(xmap, zero);
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(xmap, srcbx));
				mask.f = _mm_and_ps(mask.f, _mm_cmpge_ps(ymap, zero));
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(ymap, srcby));
				int m = _mm_movemask_ps(mask.f);
				if (m){
					// convert to integer with rounding towards zero
					xint = _mm_cvttps_epi32(xmap);
					yint = _mm_cvttps_epi32(ymap);

					//xFraction = xmap-xint;
					//yFraction = ymap-yint;
					xFraction = _mm_cvtepi32_ps(xint);
					yFraction = _mm_cvtepi32_ps(yint);
					xFraction = _mm_sub_ps(xmap, xFraction);
					yFraction = _mm_sub_ps(ymap, yFraction);

					// clip for boundary
					yint = _mm_mullo_epi32(yint, src_s);
					yint = _mm_add_epi32(yint, xint);
					//(1-xFraction)
					//(1-yFraction)
					one_minus_xFraction = _mm_sub_ps(oneFloat, xFraction);
					one_minus_yFraction = _mm_sub_ps(oneFloat, yFraction);
					yint = _mm_min_epi32(yint, srcb);
					yint = _mm_max_epi32(yint, zeromask);

					// read pixels from src and re-arrange
					psrc = pSrcImage + M128I(yint).m128i_u32[0];
					M128I(p0).m128i_u32[0] = ((unsigned int *)psrc)[0];
					M128I(p2).m128i_u32[0] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

					psrc = pSrcImage + M128I(yint).m128i_u32[1];
					M128I(p0).m128i_u32[1] = ((unsigned int *)psrc)[0];
					M128I(p2).m128i_u32[1] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

					psrc = pSrcImage + M128I(yint).m128i_u32[2];
					M128I(p0).m128i_u32[2] = ((unsigned int *)psrc)[0];
					M128I(p2).m128i_u32[2] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

					psrc = pSrcImage + M128I(yint).m128i_u32[3];
					M128I(p0).m128i_u32[3] = ((unsigned int *)psrc)[0];
					M128I(p2).m128i_u32[3] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

					// get p0, p1, p2, p3 by masking and shifting
					p1 = p0;
					p0 = _mm_and_si128(p0, p0mask);
					p1 = _mm_srli_epi32(p1, 8);
					p3 = p2;
					p2 = _mm_and_si128(p2, p0mask);
					p3 = _mm_srli_epi32(p3, 8);
					p1 = _mm_and_si128(p1, p0mask);
					p3 = _mm_and_si128(p3, p0mask);

					p0_f = _mm_cvtepi32_ps(p0);
					p1_f = _mm_cvtepi32_ps(p1);
					p2_f = _mm_cvtepi32_ps(p2);
					p3_f = _mm_cvtepi32_ps(p3);

					p0_f = _mm_mul_ps(p0_f, one_minus_xFraction);
					p0_f = _mm_mul_ps(p0_f, one_minus_yFraction);
					p1_f = _mm_mul_ps(p1_f, xFraction);
					p1_f = _mm_mul_ps(p1_f, one_minus_yFraction);
					p2_f = _mm_mul_ps(p2_f, one_minus_xFraction);
					p2_f = _mm_mul_ps(p2_f, yFraction);
					p3_f = _mm_mul_ps(p3_f, xFraction);
					p3_f = _mm_mul_ps(p3_f, yFraction);

					p0_f = _mm_add_ps(p0_f, p1_f);
					p2_f = _mm_add_ps(p2_f, p3_f);
					p0_f = _mm_add_ps(p0_f, p2_f);
					p0 = _mm_cvtps_epi32(p0_f);
					// mask for boundary
					p0 = _mm_and_si128(mask.i, p0);

					// convert to unsigned char and write to dst
					p0 = _mm_packus_epi32(p0, zeromask);
					p0 = _mm_packus_epi16(p0, zeromask);
					*dst++ = M128I(p0).m128i_i32[0];
				}
				else
				{
					*dst++ = 0;
				}
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else{
		XMM128 xint = { 0 }, yint = { 0 };
		while (y < dstHeight)
		{
			// calculate (y*m[0][1] + m[0][2]) for x and y
			xdest = _mm_set1_ps(y*r01 + const1);
			ydest = _mm_set1_ps(y*r11 + const2);

			x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128 xFraction, yFraction, one_minus_xFraction, one_minus_yFraction;
				__m128 p0_f, p1_f, p2_f, p3_f;
				__m128i p0, p1, p2, p3;			// pixels in src 
				unsigned char *psrc;
				// read x into xpel
				xmap = _mm_load_ps(&r00_x[x]);
				xmap = _mm_add_ps(xmap, xdest);				// xf = dst[x3, x2, x1, x0]
				ymap = _mm_load_ps(&r10_x[x]);
				ymap = _mm_add_ps(ymap, ydest);				// ymap <- r10*x + ty
				// convert to integer with rounding towards zero
				xint.i = _mm_cvttps_epi32(xmap);
				yint.i = _mm_cvttps_epi32(ymap);

				//xFraction = xmap-xint;
				//yFraction = ymap-yint;
				xFraction = _mm_cvtepi32_ps(xint.i);
				yFraction = _mm_cvtepi32_ps(yint.i);
				xFraction = _mm_sub_ps(xmap, xFraction);
				yFraction = _mm_sub_ps(ymap, yFraction);

				//(1-xFraction)
				//(1-yFraction)
				one_minus_xFraction = _mm_sub_ps(oneFloat, xFraction);
				one_minus_yFraction = _mm_sub_ps(oneFloat, yFraction);

				// read pixels from src and re-arrange
				psrc = pSrcImage + (yint.s32[0] * srcImageStrideInBytes + xint.s32[0]);
				M128I(p0).m128i_u32[0] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[0] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[1] * srcImageStrideInBytes + xint.s32[1]);
				M128I(p0).m128i_u32[1] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[1] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[2] * srcImageStrideInBytes + xint.s32[2]);
				M128I(p0).m128i_u32[2] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[2] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[3] * srcImageStrideInBytes + xint.s32[3]);
				M128I(p0).m128i_u32[3] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[3] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				// get p0, p1, p2, p3 by masking and shifting
				p1 = p0;
				p0 = _mm_and_si128(p0, p0mask);
				p1 = _mm_srli_epi32(p1, 8);
				p3 = p2;
				p2 = _mm_and_si128(p2, p0mask);
				p3 = _mm_srli_epi32(p3, 8);
				p1 = _mm_and_si128(p1, p0mask);
				p3 = _mm_and_si128(p3, p0mask);

				p0_f = _mm_cvtepi32_ps(p0);
				p1_f = _mm_cvtepi32_ps(p1);
				p2_f = _mm_cvtepi32_ps(p2);
				p3_f = _mm_cvtepi32_ps(p3);

				p0_f = _mm_mul_ps(p0_f, one_minus_xFraction);
				p0_f = _mm_mul_ps(p0_f, one_minus_yFraction);
				p1_f = _mm_mul_ps(p1_f, xFraction);
				p1_f = _mm_mul_ps(p1_f, one_minus_yFraction);
				p2_f = _mm_mul_ps(p2_f, one_minus_xFraction);
				p2_f = _mm_mul_ps(p2_f, yFraction);
				p3_f = _mm_mul_ps(p3_f, xFraction);
				p3_f = _mm_mul_ps(p3_f, yFraction);

				p0_f = _mm_add_ps(p0_f, p1_f);
				p2_f = _mm_add_ps(p2_f, p3_f);
				p0_f = _mm_add_ps(p0_f, p2_f);
				p0 = _mm_cvtps_epi32(p0_f);

				// convert to unsigned char and write to dst
				p0 = _mm_packus_epi32(p0, zeromask);
				p0 = _mm_packus_epi16(p0, zeromask);
				*dst++ = M128I(p0).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}

// the implementation currently uses floating point.
// TODO:: used fixed point for bilinear interpolation. We can do 8 pixels in the innel loop.
int HafCpu_WarpAffine_U8_U8_Bilinear_Constant
(
vx_uint32             dstWidth,
vx_uint32             dstHeight,
vx_uint8            * pDstImage,
vx_uint32             dstImageStrideInBytes,
vx_uint32             srcWidth,
vx_uint32             srcHeight,
vx_uint8            * pSrcImage,
vx_uint32             srcImageStrideInBytes,
ago_affine_matrix_t * matrix,
vx_uint8              border,
vx_uint8			* pLocalData
)
{
	__m128  ymap, xmap, ydest, xdest;
	__m128i srcb, src_s;
	const unsigned int u32_border = border | (border << 8) | (border << 16) | (border << 24);
	const __m128i zeromask = _mm_setzero_si128();
	const __m128i one = _mm_set1_epi32(1);
	const __m128i pborder = _mm_set1_epi32((int)border);	

	// do backward mapping to find the (x, y) locations in source corresponding to (x', y') from dest by doing inverse matrix
	const float r00 = matrix->matrix[0][0];
	const float r10 = matrix->matrix[0][1];
	const float r01 = matrix->matrix[1][0];
	const float r11 = matrix->matrix[1][1];
	const float const1 = matrix->matrix[2][0];
	const float const2 = matrix->matrix[2][1];

	const __m128 zero = _mm_set1_ps(0);
	const __m128i srcbx_i = _mm_set1_epi32(srcWidth);
	const __m128i srcby_i = _mm_set1_epi32(srcHeight);
	const __m128 srcbx = _mm_cvtepi32_ps(srcbx_i);
	const __m128 srcby = _mm_cvtepi32_ps(srcby_i);

	const __m128i p0mask = _mm_set1_epi32((int)0xFF);
	const __m128 oneFloat = _mm_set1_ps(1.0);
	srcb = _mm_set1_epi32((srcHeight)*srcImageStrideInBytes - 1);
	src_s = _mm_set1_epi32(srcImageStrideInBytes);

	XMM128 xint = { 0 }, yint = { 0 }, mask;
	unsigned int x, y;
	float *r00_x = (float*)pLocalData;
	float *r10_x = (float *)ALIGN16(r00_x + dstWidth);
	for (x = 0; x<dstWidth; x++){
		r00_x[x] = r00 * x;
		r10_x[x] = r10 * x;
	}
	bool bBoder = (const1 < 0) | (const2 < 0) | (const1 >= srcWidth) | (const2 >= srcHeight);
	// check for (dstWidth, 0)
	float x1 = (r00*dstWidth + const1);
	float y1 = (r10*dstWidth + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
	// check for (0, dstHeight)
	x1 = (r01*dstHeight + const1);
	y1 = (r11*dstHeight + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
	// check for (dstWidth, dstHeight)
	x1 = (r00*dstWidth + r01*dstHeight + const1);
	y1 = (r10*dstWidth + r11*dstHeight + const2);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);

	y = 0;
	if (bBoder){
		while (y < dstHeight)
		{
			// calculate (y*m[0][1] + m[0][2]) for x and y
			xdest = _mm_set1_ps(y*r01 + const1);
			ydest = _mm_set1_ps(y*r11 + const2);

			unsigned int x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128 xFraction, yFraction, one_minus_xFraction, one_minus_yFraction;
				__m128 p0_f, p1_f, p2_f, p3_f;
				__m128i p0, p1, p2, p3;			// pixels in src 
				unsigned char *psrc;

				// read x into xpel
				xmap = _mm_load_ps(&r00_x[x]);
				xmap = _mm_add_ps(xmap, xdest);				// xf = dst[x3, x2, x1, x0]
				ymap = _mm_load_ps(&r10_x[x]);
				ymap = _mm_add_ps(ymap, ydest);				// ymap <- r10*x + ty

				mask.f = _mm_cmpge_ps(xmap, zero);
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(xmap, srcbx));
				mask.f = _mm_and_ps(mask.f, _mm_cmpge_ps(ymap, zero));
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(ymap, srcby));
				//int m = _mm_movemask_ps(mask.f);
				//if (m){
					// convert to integer with rounding towards zero
					xint.i = _mm_cvttps_epi32(xmap);
					yint.i = _mm_cvttps_epi32(ymap);

					//xFraction = xmap-xint;
					//yFraction = ymap-yint;
					xFraction = _mm_cvtepi32_ps(xint.i);
					yFraction = _mm_cvtepi32_ps(yint.i);
					xFraction = _mm_sub_ps(xmap, xFraction);
					yFraction = _mm_sub_ps(ymap, yFraction);
					
					yint.i = _mm_mullo_epi32(yint.i, src_s);
					yint.i = _mm_add_epi32(yint.i, xint.i);
					//(1-xFraction)
					//(1-yFraction)
					one_minus_xFraction = _mm_sub_ps(oneFloat, xFraction);
					one_minus_yFraction = _mm_sub_ps(oneFloat, yFraction);
					yint.i = _mm_min_epi32(yint.i, srcb);
					yint.i = _mm_max_epi32(yint.i, zeromask);

					// read pixels from src and re-arrange
					psrc = pSrcImage + yint.s32[0];
					M128I(p0).m128i_u32[0] = ((unsigned int *)psrc)[0];
					M128I(p2).m128i_u32[0] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

					psrc = pSrcImage + yint.s32[1];
					M128I(p0).m128i_u32[1] = ((unsigned int *)psrc)[0];
					M128I(p2).m128i_u32[1] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

					psrc = pSrcImage + yint.s32[2];
					M128I(p0).m128i_u32[2] = ((unsigned int *)psrc)[0];
					M128I(p2).m128i_u32[2] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

					psrc = pSrcImage + yint.s32[3];
					M128I(p0).m128i_u32[3] = ((unsigned int *)psrc)[0];
					M128I(p2).m128i_u32[3] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

					// get p0, p1, p2, p3 by masking and shifting
					p1 = p0;
					p0 = _mm_and_si128(p0, p0mask);
					p1 = _mm_srli_epi32(p1, 8);
					p3 = p2;
					p2 = _mm_and_si128(p2, p0mask);
					p3 = _mm_srli_epi32(p3, 8);
					p1 = _mm_and_si128(p1, p0mask);
					p3 = _mm_and_si128(p3, p0mask);

					p0_f = _mm_cvtepi32_ps(p0);
					p1_f = _mm_cvtepi32_ps(p1);
					p2_f = _mm_cvtepi32_ps(p2);
					p3_f = _mm_cvtepi32_ps(p3);

					p0_f = _mm_mul_ps(p0_f, one_minus_xFraction);
					p0_f = _mm_mul_ps(p0_f, one_minus_yFraction);
					p1_f = _mm_mul_ps(p1_f, xFraction);
					p1_f = _mm_mul_ps(p1_f, one_minus_yFraction);
					p2_f = _mm_mul_ps(p2_f, one_minus_xFraction);
					p2_f = _mm_mul_ps(p2_f, yFraction);
					p3_f = _mm_mul_ps(p3_f, xFraction);
					p3_f = _mm_mul_ps(p3_f, yFraction);

					p0_f = _mm_add_ps(p0_f, p1_f);
					p2_f = _mm_add_ps(p2_f, p3_f);
					p0_f = _mm_add_ps(p0_f, p2_f);
					p0 = _mm_cvtps_epi32(p0_f);
					// mask for boundary
					p0 = _mm_and_si128(mask.i, p0);
					p0 = _mm_or_si128(p0, _mm_andnot_si128(mask.i, pborder));			// combined result

					// convert to unsigned char and write to dst
					p0 = _mm_packus_epi32(p0, zeromask);
					p0 = _mm_packus_epi16(p0, zeromask);
					*dst++ = M128I(p0).m128i_i32[0];
				/*}
				else
				{
					*dst++ = u32_border;
				}*/
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else{
		while (y < dstHeight)
		{
			// calculate (y*m[0][1] + m[0][2]) for x and y
			xdest = _mm_set1_ps(y*r01 + const1);
			ydest = _mm_set1_ps(y*r11 + const2);

			unsigned int x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128 xFraction, yFraction, one_minus_xFraction, one_minus_yFraction;
				__m128 p0_f, p1_f, p2_f, p3_f;
				__m128i p0, p1, p2, p3;			// pixels in src 
				unsigned char *psrc;
				// read x into xpel
				xmap = _mm_load_ps(&r00_x[x]);
				xmap = _mm_add_ps(xmap, xdest);				// xf = dst[x3, x2, x1, x0]
				ymap = _mm_load_ps(&r10_x[x]);
				ymap = _mm_add_ps(ymap, ydest);				// ymap <- r10*x + ty
				// convert to integer with rounding towards zero
				xint.i = _mm_cvttps_epi32(xmap);
				yint.i = _mm_cvttps_epi32(ymap);

				//xFraction = xmap-xint;
				//yFraction = ymap-yint;
				xFraction = _mm_cvtepi32_ps(xint.i);
				yFraction = _mm_cvtepi32_ps(yint.i);
				xFraction = _mm_sub_ps(xmap, xFraction);
				yFraction = _mm_sub_ps(ymap, yFraction);

				//(1-xFraction)
				//(1-yFraction)
				one_minus_xFraction = _mm_sub_ps(oneFloat, xFraction);
				one_minus_yFraction = _mm_sub_ps(oneFloat, yFraction);

				// read pixels from src and re-arrange
				psrc = pSrcImage + (yint.s32[0] * srcImageStrideInBytes + xint.s32[0]);
				M128I(p0).m128i_u32[0] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[0] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[1] * srcImageStrideInBytes + xint.s32[1]);
				M128I(p0).m128i_u32[1] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[1] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[2] * srcImageStrideInBytes + xint.s32[2]);
				M128I(p0).m128i_u32[2] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[2] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[3] * srcImageStrideInBytes + xint.s32[3]);
				M128I(p0).m128i_u32[3] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[3] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				// get p0, p1, p2, p3 by masking and shifting
				p1 = p0;
				p0 = _mm_and_si128(p0, p0mask);
				p1 = _mm_srli_epi32(p1, 8);
				p3 = p2;
				p2 = _mm_and_si128(p2, p0mask);
				p3 = _mm_srli_epi32(p3, 8);
				p1 = _mm_and_si128(p1, p0mask);
				p3 = _mm_and_si128(p3, p0mask);

				p0_f = _mm_cvtepi32_ps(p0);
				p1_f = _mm_cvtepi32_ps(p1);
				p2_f = _mm_cvtepi32_ps(p2);
				p3_f = _mm_cvtepi32_ps(p3);

				p0_f = _mm_mul_ps(p0_f, one_minus_xFraction);
				p0_f = _mm_mul_ps(p0_f, one_minus_yFraction);
				p1_f = _mm_mul_ps(p1_f, xFraction);
				p1_f = _mm_mul_ps(p1_f, one_minus_yFraction);
				p2_f = _mm_mul_ps(p2_f, one_minus_xFraction);
				p2_f = _mm_mul_ps(p2_f, yFraction);
				p3_f = _mm_mul_ps(p3_f, xFraction);
				p3_f = _mm_mul_ps(p3_f, yFraction);

				p0_f = _mm_add_ps(p0_f, p1_f);
				p2_f = _mm_add_ps(p2_f, p3_f);
				p0_f = _mm_add_ps(p0_f, p2_f);
				p0 = _mm_cvtps_epi32(p0_f);

				// convert to unsigned char and write to dst
				p0 = _mm_packus_epi32(p0, zeromask);
				p0 = _mm_packus_epi16(p0, zeromask);
				*dst++ = M128I(p0).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

int HafCpu_WarpPerspective_U8_U8_Nearest
(
vx_uint32                  dstWidth,
vx_uint32                  dstHeight,
vx_uint8                 * pDstImage,
vx_uint32                  dstImageStrideInBytes,
vx_uint32                  srcWidth,
vx_uint32                  srcHeight,
vx_uint8                 * pSrcImage,
vx_uint32                  srcImageStrideInBytes,
ago_perspective_matrix_t * matrix,
vx_uint8				 * pLocalData
)
{
	return HafCpu_WarpPerspective_U8_U8_Nearest_Constant(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes, srcWidth,
		srcHeight, pSrcImage, srcImageStrideInBytes, matrix, (unsigned char)0, pLocalData);
}


// This alogorithm implements Constant Denominator method described in 
// " A Novel Architechture for real time sprite decoding".
// The idea is to do perpective warping along the lines of constant divisor..
// The number of floating point divisions are reduced from O(Nsqared) to O(N)
/*
	forward mapping: 
		x' = (ax+by+c)/(gx+hy+1)
		y' = (dx+ey+f)/(gx+hy+1)
	backward mapping:
		x  = ((hf-e)x'+(b-hc)y'+(ec-bf))/(eg-dh)x'+(ah-bg)y'+(db-ae))
		y  = ((d-fg)x'+(cg-a)y'+(af-dc))/(eg-dh)x'+(ah-bg)y'+(db-ae))

*/


int HafCpu_WarpPerspective_U8_U8_Nearest_Constant
(
vx_uint32                  dstWidth,
vx_uint32                  dstHeight,
vx_uint8                 * pDstImage,
vx_uint32                  dstImageStrideInBytes,
vx_uint32                  srcWidth,
vx_uint32                  srcHeight,
vx_uint8                 * pSrcImage,
vx_uint32                  srcImageStrideInBytes,
ago_perspective_matrix_t * matrix,
vx_uint8                   border,
vx_uint8				 * pLocalData
)
{
	// calculate inverse mapping coefficients for x and y
	const float a = matrix->matrix[0][0];				
	const float d = matrix->matrix[0][1];
	const float g = matrix->matrix[0][2];
	const float b = matrix->matrix[1][0];
	const float e = matrix->matrix[1][1];
	const float h = matrix->matrix[1][2];
	const float c = matrix->matrix[2][0];		
	const float f = matrix->matrix[2][1];		
	const float i = matrix->matrix[2][2];

	// can't assume if end points in the warped image is within boundary, all the warped image is within boundary
	bool bBoder = 1;

	XMM128 mask;
	__m128 xdest, ydest, zdest;
	const __m128i zeromask = _mm_setzero_si128();
	const __m128i one = _mm_set1_epi32(1);
	const __m128i pborder = _mm_set1_epi32((int)border);	
	const __m128i srcb = _mm_set1_epi32((srcHeight*srcImageStrideInBytes) - 1);
	const __m128i src_s = _mm_set1_epi32(srcImageStrideInBytes);
	const __m128 zero = _mm_set1_ps(0);
	const __m128 srcbx = _mm_set1_ps((float)srcWidth);
	const __m128 srcby = _mm_set1_ps((float)srcHeight);
	const __m128 oneFloat = _mm_set1_ps(1.0);

	unsigned int x;
	float *A_x = (float*)pLocalData;
	float *D_x = (float *)ALIGN16(A_x + dstWidth);
	float *G_x = (float *)ALIGN16(D_x + dstWidth);
	for (x = 0; x<dstWidth; x++){
		A_x[x] = a * x;
		D_x[x] = d * x;
		G_x[x] = g * x;			// (eg - dh)
	}

	unsigned int y = 0;
	// do the plain vanilla version with floating point division in inner_loop
	if (bBoder){
		while (y < dstHeight)
		{
			xdest = _mm_set1_ps(y*b + c);
			ydest = _mm_set1_ps(y*e + f);
			zdest = _mm_set1_ps(y*h + i);
			x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128 xmap, ymap, zmap;
				__m128i xpels, ypels;

				zmap = _mm_load_ps(&G_x[x]);
				xmap = _mm_load_ps(&A_x[x]);
				zmap = _mm_add_ps(zmap, zdest);
				ymap = _mm_load_ps(&D_x[x]);
				zmap = _mm_div_ps(oneFloat, zmap);
				xmap = _mm_add_ps(xmap, xdest);				
				ymap = _mm_add_ps(ymap, ydest);				
				xmap = _mm_mul_ps(xmap, zmap);
				ymap = _mm_mul_ps(ymap, zmap);

				mask.f = _mm_cmpge_ps(xmap, zero);
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(xmap, srcbx));
				mask.f = _mm_and_ps(mask.f, _mm_cmpge_ps(ymap, zero));
				mask.f = _mm_and_ps(mask.f, _mm_cmplt_ps(ymap, srcby));

				// convert to integer with rounding towards zero
				xpels = _mm_cvttps_epi32(xmap);
				ypels = _mm_cvttps_epi32(ymap);
				// multiply ydest*srcImageStrideInBytes
				ypels = _mm_mullo_epi32(ypels, src_s);
				ypels = _mm_add_epi32(ypels, xpels);			// pixel location at src for dst image.

				// check if the values exceed boundary and clamp it to boundary :: need to do this to avoid memory access violations
				ypels = _mm_min_epi32(ypels, srcb);
				ypels = _mm_max_epi32(ypels, zeromask);

				// check if the values exceed boundary and clamp it to boundary
				xpels = _mm_set_epi32(pSrcImage[M128I(ypels).m128i_i32[3]], pSrcImage[M128I(ypels).m128i_i32[2]], pSrcImage[M128I(ypels).m128i_i32[1]], pSrcImage[M128I(ypels).m128i_i32[0]]);
				// mask for boundary
				xpels = _mm_and_si128(xpels, mask.i);
				xpels = _mm_or_si128(xpels, _mm_andnot_si128(mask.i, pborder));			// combined result

				// convert to unsigned char and write to dst
				xpels = _mm_packus_epi32(xpels, zeromask);
				xpels = _mm_packus_epi16(xpels, zeromask);
				*dst++ = M128I(xpels).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	else{
		while (y < dstHeight)
		{
			xdest = _mm_set1_ps(y*b + c);
			ydest = _mm_set1_ps(y*e + f);
			zdest = _mm_set1_ps(y*h + i);
			unsigned int *dst = (unsigned int *)pDstImage;
			x = 0;
			while (x < dstWidth)
			{
				__m128 xmap, ymap, zmap;
				__m128i xpels, ypels;

				zmap = _mm_load_ps(&G_x[x]);
				xmap = _mm_load_ps(&A_x[x]);
				zmap = _mm_add_ps(zmap, zdest);
				ymap = _mm_load_ps(&D_x[x]);
				zmap = _mm_div_ps(oneFloat, zmap);
				xmap = _mm_add_ps(xmap, xdest);				
				ymap = _mm_add_ps(ymap, ydest);				
				xmap = _mm_mul_ps(xmap, zmap);
				ymap = _mm_mul_ps(ymap, zmap);

				// convert to integer with rounding towards zero
				xpels = _mm_cvttps_epi32(xmap);
				ypels = _mm_cvttps_epi32(ymap);
				// multiply ydest*srcImageStrideInBytes
				ypels = _mm_mullo_epi32(ypels, src_s);
				ypels = _mm_add_epi32(ypels, xpels);			// pixel location at src for dst image.

				// check if the values exceed boundary and clamp it to boundary
				xpels = _mm_set_epi32(pSrcImage[M128I(ypels).m128i_i32[3]], pSrcImage[M128I(ypels).m128i_i32[2]], pSrcImage[M128I(ypels).m128i_i32[1]], pSrcImage[M128I(ypels).m128i_i32[0]]);
				// convert to unsigned char and write to dst
				xpels = _mm_packus_epi32(xpels, zeromask);
				xpels = _mm_packus_epi16(xpels, zeromask);
				*dst++ = M128I(xpels).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}

	return AGO_SUCCESS;
}

int HafCpu_WarpPerspective_U8_U8_Bilinear
(
vx_uint32                  dstWidth,
vx_uint32                  dstHeight,
vx_uint8                 * pDstImage,
vx_uint32                  dstImageStrideInBytes,
vx_uint32                  srcWidth,
vx_uint32                  srcHeight,
vx_uint8                 * pSrcImage,
vx_uint32                  srcImageStrideInBytes,
ago_perspective_matrix_t * matrix,
vx_uint8				 * pLocalData
)
{
	return HafCpu_WarpPerspective_U8_U8_Bilinear_Constant(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes, srcWidth,
		srcHeight, pSrcImage, srcImageStrideInBytes, matrix, (unsigned char)0, pLocalData);
}


int HafCpu_WarpPerspective_U8_U8_Bilinear_Constant
(
vx_uint32                  dstWidth,
vx_uint32                  dstHeight,
vx_uint8                 * pDstImage,
vx_uint32                  dstImageStrideInBytes,
vx_uint32                  srcWidth,
vx_uint32                  srcHeight,
vx_uint8                 * pSrcImage,
vx_uint32                  srcImageStrideInBytes,
ago_perspective_matrix_t * matrix,
vx_uint8                   border,
vx_uint8				 * pLocalData
)
{
	// calculate inverse mapping coefficients for x and y
	const float a = matrix->matrix[0][0];
	const float d = matrix->matrix[0][1];
	const float g = matrix->matrix[0][2];
	const float b = matrix->matrix[1][0];
	const float e = matrix->matrix[1][1];
	const float h = matrix->matrix[1][2];
	const float c = matrix->matrix[2][0];
	const float f = matrix->matrix[2][1];
	const float i = matrix->matrix[2][2];

	XMM128 xint, yint, xmask, ymask;
	__m128 xdest, ydest, zdest;
	const __m128i zeromask = _mm_setzero_si128();
	const __m128i one = _mm_set1_epi32(1);
	const __m128i pborder = _mm_set1_epi32((int)border);	
	const __m128 zero = _mm_set1_ps(0);
	const __m128 oneFloat = _mm_set1_ps(1.0);
	const __m128i srcbx = _mm_set1_epi32((int)srcWidth);
	const __m128i srcby = _mm_set1_epi32((int)srcHeight);
	const __m128i p0mask = _mm_set1_epi32((int)0xFF);
	const __m128i srcb = _mm_set1_epi32((srcHeight)*srcImageStrideInBytes - 1);
	const __m128i src_s = _mm_set1_epi32(srcImageStrideInBytes);
	const __m128i srcbx1 = _mm_set1_epi32((int)(srcWidth));
	const __m128i srcby1 = _mm_set1_epi32((int)(srcHeight));
	const __m128i negone = _mm_set1_epi32((int)-1);

	unsigned int x;
	float *A_x = (float*)pLocalData;
	float *D_x = (float *)ALIGN16(A_x + dstWidth);
	float *G_x = (float *)ALIGN16(D_x + dstWidth);
	for (x = 0; x<dstWidth; x++){
		A_x[x] = a * x;
		D_x[x] = d * x;
		G_x[x] = g * x;			// (eg - dh)
	}

#if 0
	// find out if all the mapped pixels are within valid range or not
	// find out if all the mapped pixels are within valid range or not
	float z, x0, y0, x1, y1;
	z = (float)(1.0 / i);
	x0 = c * z;
	y0 = f * z;
	z = (float)(1.0 / (h * dstHeight + g*dstWidth + i));
	x1 = (a*dstWidth + b * dstHeight + c) * z;
	y1 = (d*dstWidth + e * dstHeight + f) * z;
	bool bBoder = (x0 < 0) | (y0 < 0) | (x0 >= srcWidth) | (y0 >= srcHeight);
	bBoder |= (x1 < 0) | (y1 < 0) | (x1 >= srcWidth) | (y1 >= srcHeight);
#endif
	bool bBoder = 1;

	unsigned int y = 0;
	if (bBoder){
		// do the plain vanilla version with floating point division in inner_loop
		while (y < dstHeight)
		{
			xdest = _mm_set1_ps(y*b + c);
			ydest = _mm_set1_ps(y*e + f);
			zdest = _mm_set1_ps(y*h + i);
			x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128 xmap, ymap, zmap;
				__m128 xFraction, yFraction, one_minus_xFraction, one_minus_yFraction;
				__m128 p0_f, p1_f, p2_f, p3_f;
				__m128i p0 = _mm_set1_epi8(border), p1, p2 = _mm_set1_epi8(border), p3;	
				__m128i mask, mask1; // mask for boundary checking 
				unsigned char *psrc;
				zmap = _mm_load_ps(&G_x[x]);
				xmap = _mm_load_ps(&A_x[x]);
				zmap = _mm_add_ps(zmap, zdest);
				ymap = _mm_load_ps(&D_x[x]);
				zmap = _mm_div_ps(oneFloat, zmap);
				xmap = _mm_add_ps(xmap, xdest);
				ymap = _mm_add_ps(ymap, ydest);
				xmap = _mm_mul_ps(xmap, zmap);
				ymap = _mm_mul_ps(ymap, zmap);
				xmask.f = _mm_cmplt_ps(xmap, zero);
				ymask.f = _mm_cmplt_ps(ymap, zero);
				// convert to integer with rounding towards zero
				xint.i = _mm_cvttps_epi32(xmap);
				xint.i = _mm_sub_epi32(xint.i, _mm_srli_epi32(xmask.i, 31));
				yint.i = _mm_cvttps_epi32(ymap);
				yint.i = _mm_sub_epi32(yint.i, _mm_srli_epi32(ymask.i, 31));
				mask = _mm_cmplt_epi32(xint.i, srcbx);
				mask = _mm_andnot_si128(_mm_cmplt_epi32(xint.i, zeromask), mask);
				mask = _mm_and_si128(mask, _mm_cmplt_epi32(yint.i, srcby));
				mask = _mm_andnot_si128(_mm_cmplt_epi32(yint.i, zeromask), mask);
				mask1 = _mm_cmplt_epi32(xint.i, srcbx1);	// xmap+1 < srcWidth;
				mask1 = _mm_andnot_si128(_mm_cmplt_epi32(xint.i, negone), mask1);
				mask1 = _mm_and_si128(mask1, _mm_cmplt_epi32(yint.i, srcby1));
				mask1 = _mm_andnot_si128(_mm_cmplt_epi32(yint.i, negone), mask1);

				//xFraction = xmap-xint;
				//yFraction = ymap-yint;
				xFraction = _mm_cvtepi32_ps(xint.i);
				yFraction = _mm_cvtepi32_ps(yint.i);
				xFraction = _mm_sub_ps(xmap, xFraction);
				yFraction = _mm_sub_ps(ymap, yFraction);

				// clip for boundary
				yint.i = _mm_mullo_epi32(yint.i, src_s);
				yint.i = _mm_add_epi32(yint.i, xint.i);
				//(1-xFraction)
				//(1-yFraction)
				one_minus_xFraction = _mm_sub_ps(oneFloat, xFraction);
				one_minus_yFraction = _mm_sub_ps(oneFloat, yFraction);
				yint.i = _mm_min_epi32(yint.i, srcb);
				yint.i = _mm_max_epi32(yint.i, zeromask);

				// read pixels from src and re-arrange
				psrc = pSrcImage + yint.s32[0];
				M128I(p0).m128i_u32[0] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[0] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + yint.s32[1];
				M128I(p0).m128i_u32[1] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[1] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + yint.s32[2];
				M128I(p0).m128i_u32[2] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[2] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + yint.s32[3];
				M128I(p0).m128i_u32[3] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[3] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				// get p0, p1, p2, p3 by masking and shifting
				p1 = p0;
				p0 = _mm_and_si128(p0, p0mask);
				p1 = _mm_srli_epi32(p1, 8);
				p3 = p2;
				p2 = _mm_and_si128(p2, p0mask);
				p3 = _mm_srli_epi32(p3, 8);
				p1 = _mm_and_si128(p1, p0mask);
				p3 = _mm_and_si128(p3, p0mask);
				// mask p0, p2 with border
				p0 = _mm_and_si128(p0, mask);
				p0 = _mm_or_si128(p0, _mm_andnot_si128(mask, pborder));			// combined result
				p2 = _mm_and_si128(p2, mask);
				p2 = _mm_or_si128(p2, _mm_andnot_si128(mask, pborder));			// combined result
				// mask p1 and p3 with border
				p1 = _mm_and_si128(p1, mask1);
				p1 = _mm_or_si128(p1, _mm_andnot_si128(mask1, pborder));			// combined result
				p3 = _mm_and_si128(p3, mask1);
				p3 = _mm_or_si128(p3, _mm_andnot_si128(mask1, pborder));			// combined result

				p0_f = _mm_cvtepi32_ps(p0);
				p1_f = _mm_cvtepi32_ps(p1);
				p2_f = _mm_cvtepi32_ps(p2);
				p3_f = _mm_cvtepi32_ps(p3);

				p0_f = _mm_mul_ps(p0_f, one_minus_xFraction);
				p0_f = _mm_mul_ps(p0_f, one_minus_yFraction);
				p1_f = _mm_mul_ps(p1_f, xFraction);
				p1_f = _mm_mul_ps(p1_f, one_minus_yFraction);
				p2_f = _mm_mul_ps(p2_f, one_minus_xFraction);
				p2_f = _mm_mul_ps(p2_f, yFraction);
				p3_f = _mm_mul_ps(p3_f, xFraction);
				p3_f = _mm_mul_ps(p3_f, yFraction);

				p0_f = _mm_add_ps(p0_f, p1_f);
				p0_f = _mm_add_ps(p0_f, p2_f);
				p0_f = _mm_add_ps(p0_f, p3_f);
				p0 = _mm_cvttps_epi32(p0_f);

				// convert to unsigned char and write to dst
				p0 = _mm_packus_epi32(p0, zeromask);
				p0 = _mm_packus_epi16(p0, zeromask);
				*dst++ = M128I(p0).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}else{ 
		// do the plain vanilla version with floating point division in inner_loop
		while (y < dstHeight)
		{
			xdest = _mm_set1_ps(y*b + c);
			ydest = _mm_set1_ps(y*e + f);
			zdest = _mm_set1_ps(y*h + i);
			x = 0;
			unsigned int *dst = (unsigned int *)pDstImage;
			while (x < dstWidth)
			{
				__m128 xmap, ymap, zmap;
				__m128 xFraction, yFraction, one_minus_xFraction, one_minus_yFraction;
				__m128 p0_f, p1_f, p2_f, p3_f;
				__m128i p0, p1, p2, p3;			// pixels in src 
				unsigned char *psrc;

				zmap = _mm_load_ps(&G_x[x]);
				xmap = _mm_load_ps(&A_x[x]);
				zmap = _mm_add_ps(zmap, zdest);
				ymap = _mm_load_ps(&D_x[x]);
				zmap = _mm_div_ps(oneFloat, zmap);
				xmap = _mm_add_ps(xmap, xdest);
				ymap = _mm_add_ps(ymap, ydest);
				xmap = _mm_mul_ps(xmap, zmap);
				ymap = _mm_mul_ps(ymap, zmap);

				// convert to integer with rounding towards zero
				xint.i = _mm_cvttps_epi32(xmap);
				yint.i = _mm_cvttps_epi32(ymap);

				//xFraction = xmap-xint;
				//yFraction = ymap-yint;
				xFraction = _mm_cvtepi32_ps(xint.i);
				yFraction = _mm_cvtepi32_ps(yint.i);
				xFraction = _mm_sub_ps(xmap, xFraction);
				yFraction = _mm_sub_ps(ymap, yFraction);

				//(1-xFraction)
				//(1-yFraction)
				one_minus_xFraction = _mm_sub_ps(oneFloat, xFraction);
				one_minus_yFraction = _mm_sub_ps(oneFloat, yFraction);

				// read pixels from src and re-arrange
				psrc = pSrcImage + (yint.s32[0] * srcImageStrideInBytes + xint.s32[0]);
				M128I(p0).m128i_u32[0] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[0] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[1] * srcImageStrideInBytes + xint.s32[1]);
				M128I(p0).m128i_u32[1] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[1] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[2] * srcImageStrideInBytes + xint.s32[2]);
				M128I(p0).m128i_u32[2] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[2] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				psrc = pSrcImage + (yint.s32[3] * srcImageStrideInBytes + xint.s32[3]);
				M128I(p0).m128i_u32[3] = ((unsigned int *)psrc)[0];
				M128I(p2).m128i_u32[3] = ((unsigned int *)(psrc + srcImageStrideInBytes))[0];

				// get p0, p1, p2, p3 by masking and shifting
				p1 = p0;
				p0 = _mm_and_si128(p0, p0mask);
				p1 = _mm_srli_epi32(p1, 8);
				p3 = p2;
				p2 = _mm_and_si128(p2, p0mask);
				p3 = _mm_srli_epi32(p3, 8);
				p1 = _mm_and_si128(p1, p0mask);
				p3 = _mm_and_si128(p3, p0mask);

				p0_f = _mm_cvtepi32_ps(p0);
				p1_f = _mm_cvtepi32_ps(p1);
				p2_f = _mm_cvtepi32_ps(p2);
				p3_f = _mm_cvtepi32_ps(p3);

				p0_f = _mm_mul_ps(p0_f, one_minus_xFraction);
				p0_f = _mm_mul_ps(p0_f, one_minus_yFraction);
				p1_f = _mm_mul_ps(p1_f, xFraction);
				p1_f = _mm_mul_ps(p1_f, one_minus_yFraction);
				p2_f = _mm_mul_ps(p2_f, one_minus_xFraction);
				p2_f = _mm_mul_ps(p2_f, yFraction);
				p3_f = _mm_mul_ps(p3_f, xFraction);
				p3_f = _mm_mul_ps(p3_f, yFraction);

				p0_f = _mm_add_ps(p0_f, p1_f);
				p2_f = _mm_add_ps(p2_f, p3_f);
				p0_f = _mm_add_ps(p0_f, p2_f);
				p0 = _mm_cvttps_epi32(p0_f);

				// convert to unsigned char and write to dst
				p0 = _mm_packus_epi32(p0, zeromask);
				p0 = _mm_packus_epi16(p0, zeromask);
				*dst++ = M128I(p0).m128i_i32[0];
				x += 4;
			}
			y++;
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}


int HafCpu_ScaleImage_U8_U8_Nearest
(
vx_uint32            dstWidth,
vx_uint32            dstHeight,
vx_uint8           * pDstImage,
vx_uint32            dstImageStrideInBytes,
vx_uint32            srcWidth,
vx_uint32            srcHeight,
vx_uint8           * pSrcImage,
vx_uint32            srcImageStrideInBytes,
ago_scale_matrix_t * matrix
)
{
	int xinc, yinc, ypos, xpos, yoffs, xoffs;// , newDstHeight, newDstWidth;

	// precompute Xmap and Ymap  based on scale factors
	unsigned short *Xmap = (unsigned short *)((vx_uint8*)matrix + sizeof(AgoConfigScaleMatrix));
	unsigned short *Ymap = Xmap + ((dstWidth+15)&~15);
	unsigned int x, y;

	yinc = (int)(FP_MUL * matrix->yscale);		// to convert to fixed point
	xinc = (int)(FP_MUL * matrix->xscale);
	yoffs = (int)(FP_MUL * matrix->yoffset);		// to convert to fixed point
	xoffs = (int)(FP_MUL * matrix->xoffset);
	// generate ymap;
	for (y = 0, ypos = yoffs; y < (int)dstHeight; y++, ypos += yinc)
	{
		int ymap;
		ymap = (ypos >> FP_BITS);
		if (ymap > (int)(srcHeight - 1)){
			ymap = srcHeight - 1;
		}
		if (ymap < 0) ymap = 0;
		Ymap[y] = (unsigned short)ymap;
		
	}
	// generate xmap;
	for (x = 0, xpos = xoffs; x < (int)dstWidth; x++, xpos += xinc)
	{
		int xmap;
		xmap = (xpos >> FP_BITS);
		if (xmap > (int)(srcWidth - 1)){
			xmap = (srcWidth - 1);
		}
		if (xmap < 0) xmap = 0;
		Xmap[x] = (unsigned short)xmap;
	}
	// now do the scaling
	__m128i zeromask = _mm_set1_epi32((int)0);
	if (dstWidth >= 16){
		for (y = 0; y < dstHeight; y++)
		{
			unsigned int yadd = Ymap[y] * srcImageStrideInBytes;
			__m128i syint = _mm_set1_epi32(yadd);
			unsigned int *pdst = (unsigned int *)pDstImage;
			for (x = 0; x <= (dstWidth - 16); x += 16)
			{
				__m128i mapx0, mapx1, mapx2, mapx3;
				mapx0 = _mm_load_si128((__m128i *)&Xmap[x]);
				mapx1 = _mm_load_si128((__m128i *)&Xmap[x + 8]);
				mapx2 = _mm_unpackhi_epi16(mapx0, zeromask);
				mapx0 = _mm_cvtepi16_epi32(mapx0);
				mapx3 = _mm_unpackhi_epi16(mapx1, zeromask);
				mapx1 = _mm_cvtepi16_epi32(mapx1);
				mapx0 = _mm_add_epi32(mapx0, syint);
				mapx2 = _mm_add_epi32(mapx2, syint);
				mapx1 = _mm_add_epi32(mapx1, syint);
				mapx3 = _mm_add_epi32(mapx3, syint);
				// copy to dst
				*pdst++ = pSrcImage[M128I(mapx0).m128i_i32[0]] | (pSrcImage[M128I(mapx0).m128i_i32[1]] << 8) |
					(pSrcImage[M128I(mapx0).m128i_i32[2]] << 16) | (pSrcImage[M128I(mapx0).m128i_i32[3]] << 24);

				*pdst++ = pSrcImage[M128I(mapx2).m128i_i32[0]] | (pSrcImage[M128I(mapx2).m128i_i32[1]] << 8) |
					(pSrcImage[M128I(mapx2).m128i_i32[2]] << 16) | (pSrcImage[M128I(mapx2).m128i_i32[3]] << 24);

				*pdst++ = pSrcImage[M128I(mapx1).m128i_i32[0]] | (pSrcImage[M128I(mapx1).m128i_i32[1]] << 8) |
					(pSrcImage[M128I(mapx1).m128i_i32[2]] << 16) | (pSrcImage[M128I(mapx1).m128i_i32[3]] << 24);

				*pdst++ = pSrcImage[M128I(mapx3).m128i_i32[0]] | (pSrcImage[M128I(mapx3).m128i_i32[1]] << 8) |
					(pSrcImage[M128I(mapx3).m128i_i32[2]] << 16) | (pSrcImage[M128I(mapx3).m128i_i32[3]] << 24);

			}
			for (; x < dstWidth; x++)
				pDstImage[x] = pSrcImage[Xmap[x] + yadd];

			pDstImage += dstImageStrideInBytes;

		}
	}
	else
	{
		for (y = 0; y < dstHeight; y++)
		{
			unsigned int yadd = Ymap[y] * srcImageStrideInBytes;
			x = 0;
			for (; x < dstWidth; x++)
				pDstImage[x] = pSrcImage[Xmap[x] + yadd];
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_ScaleImage_U8_U8_Bilinear
(
vx_uint32            dstWidth,
vx_uint32            dstHeight,
vx_uint8           * pDstImage,
vx_uint32            dstImageStrideInBytes,
vx_uint32            srcWidth,
vx_uint32            srcHeight,
vx_uint8           * pSrcImage,
vx_uint32            srcImageStrideInBytes,
ago_scale_matrix_t * matrix
)
{
	int xinc, yinc,xoffs, yoffs;

	unsigned char *pdst = pDstImage;
	yinc = (int)(FP_MUL * matrix->yscale);		// to convert to fixed point
	xinc = (int)(FP_MUL * matrix->xscale);
	yoffs = (int)(FP_MUL * matrix->yoffset);		// to convert to fixed point
	xoffs = (int)(FP_MUL * matrix->xoffset);

	// SSE4  version
	int alignW = (dstWidth + 15)&~15;
	unsigned short *Xmap = (unsigned short *)((vx_uint8*)matrix + sizeof(AgoConfigScaleMatrix));
	unsigned short *Xfrac = Xmap + alignW;
	unsigned short *One_min_xf = Xfrac + alignW;

	int xpos = xoffs;
	for (unsigned int x = 0; x < dstWidth; x++, xpos += xinc)
	{
		int xf;
		int xmap = (xpos >> FP_BITS);
		if (xmap >= (int)(srcWidth - 1)){
			Xmap[x] = (unsigned short)(srcWidth - 1);
		}
		Xmap[x] = (xmap<0)? 0: (unsigned short)xmap;
		xf = ((xpos & 0x3ffff)+0x200)>>10;
		Xfrac[x] = xf;
		One_min_xf[x] = (0x100 - xf);
	}

	XMM128 pp1 = { 0 }, pp2 = { 0 };
	const __m128i mask = _mm_set1_epi16((short)0xff);
	const __m128i round = _mm_set1_epi16((short)0x80);
	unsigned int newDstWidth = dstWidth & ~7;	// nearest multiple of 8

	for (int y = 0, ypos = yoffs; y < (int)dstHeight; y++, ypos += yinc)
	{
		int ym, yf, one_min_yf;
		__m128i rxmm0, rxmm7;
		vx_uint8 *pSrc1, *pSrc2;

		ym = (ypos >> FP_BITS);
		yf = ((ypos & 0x3ffff)+0x200)>>10;
		one_min_yf = (0x100 - yf);
		yoffs = ym*srcImageStrideInBytes;
		if (ym < 0){
			ym = yoffs = 0;
			pSrc1 = pSrc2 = pSrcImage;
		}
		else if (ym >= (int)(srcHeight - 1)){
			ym = srcHeight - 1;
			pSrc1 = pSrc2 = pSrcImage + ym*srcImageStrideInBytes;
		}
		else
		{
			pSrc1 = pSrcImage + ym*srcImageStrideInBytes;
			pSrc2 = pSrc1 + srcImageStrideInBytes;
		}
		rxmm0 = _mm_set1_epi16((unsigned short)one_min_yf);
		rxmm7 = _mm_set1_epi16((unsigned short)yf);
		unsigned int x = 0;
		for (; x < newDstWidth; x += 8)
		{
			__m128i mapxy, rxmm1, rxmm2, rxmm3, rxmm4;
			mapxy = _mm_load_si128((__m128i *)&Xmap[x]);		// mapped table [srcx7...src_x3,src_x2,src_x1,src_x0]
			// load pixels for mapxy
			for (int xx = 0; xx < 8; xx++)
			{
				pp1.u16[xx] = ((unsigned short*)&pSrc1[M128I(mapxy).m128i_i16[xx]])[0];
				pp2.u16[xx] = ((unsigned short*)&pSrc2[M128I(mapxy).m128i_i16[xx]])[0];
			}
			// unpack src for p1 and p2
			rxmm1 = _mm_and_si128(pp1.i, mask);		// p1
			pp1.i = _mm_srli_epi16(pp1.i, 8);		// p2
			// unpack pp2 for p3 and p4
			rxmm4 = _mm_and_si128(pp2.i, mask);		// p3
			pp2.i = _mm_srli_epi16(pp2.i, 8);		// p4

			// load xf and 1-xf
			rxmm2 = _mm_load_si128((__m128i *)&Xfrac[x]);			// xf
			rxmm3 = _mm_load_si128((__m128i *)&One_min_xf[x]);		// 1-xf

			// t1 = (unsigned char)((ione_minus_x *p1 + ifraction_x *p2) >> FW_WEIGHT); 
			rxmm1 = _mm_mullo_epi16(rxmm1, rxmm3);	//  ione_minus_xf *p1 	
			pp1.i = _mm_mullo_epi16(pp1.i, rxmm2);	//  ifraction_x  *p2		
			rxmm1 = _mm_add_epi16(rxmm1, pp1.i);
			rxmm1 = _mm_add_epi16(rxmm1, round);
			rxmm1 = _mm_srli_epi16(rxmm1, 8);

			//  t2 = (unsigned char)((ione_minus_x *p3 + ifraction_x *p4) >> FW_WEIGHT); 	
			rxmm4 = _mm_mullo_epi16(rxmm4, rxmm3);	//  ione_minus_x *p3 	
			pp2.i = _mm_mullo_epi16(pp2.i, rxmm2);	//  ifraction_x  *p4		
			rxmm4 = _mm_add_epi16(rxmm4, pp2.i);
			rxmm4 = _mm_add_epi16(rxmm4, round);
			rxmm4 = _mm_srli_epi16(rxmm4, 8);


			// *(pDst + x + y*dstStep) = (unsigned char)((ione_minus_y *t1 + ifraction_y * t2) >> FW_WEIGHT)	
			rxmm1 = _mm_mullo_epi16(rxmm1, rxmm0);	//  ione_minus_y * t1 	
			rxmm4 = _mm_mullo_epi16(rxmm4, rxmm7);	//  ifraction_y  * t2		
			rxmm1 = _mm_add_epi16(rxmm1, rxmm4);
			rxmm1 = _mm_add_epi16(rxmm1, round);
			rxmm1 = _mm_srli_epi16(rxmm1, 8);
			rxmm1 = _mm_packus_epi16(rxmm1, rxmm1);

			_mm_storel_epi64((__m128i *)(pDstImage + x), rxmm1);
		}
		for (x = newDstWidth; x < dstWidth; x++) {
			const unsigned char *p0 = pSrc1 + Xmap[x];
			const unsigned char *p1 = pSrc2 + Xmap[x];
			pDstImage[x] = ((One_min_xf[x] * one_min_yf*p0[0]) + (Xfrac[x] * one_min_yf*p0[1]) + (One_min_xf[x] * yf*p1[0]) + (Xfrac[x] * yf*p1[1]) + 0x8000) >> 16;
		}

		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}


int HafCpu_ScaleImage_U8_U8_Bilinear_Replicate
(
vx_uint32            dstWidth,
vx_uint32            dstHeight,
vx_uint8           * pDstImage,
vx_uint32            dstImageStrideInBytes,
vx_uint32            srcWidth,
vx_uint32            srcHeight,
vx_uint8           * pSrcImage,
vx_uint32            srcImageStrideInBytes,
ago_scale_matrix_t * matrix
)
{

	// SSE4  version
	int xinc, yinc, xoffs, yoffs;

	unsigned char *pdst = pDstImage;
	yinc = (int)(FP_MUL * matrix->yscale);		// to convert to fixed point
	xinc = (int)(FP_MUL * matrix->xscale);
	yoffs = (int)(FP_MUL * matrix->yoffset);		// to convert to fixed point
	xoffs = (int)(FP_MUL * matrix->xoffset);
	int alignW = (dstWidth + 15)&~15;
	unsigned short *Xmap = (unsigned short *)((vx_uint8*)matrix + sizeof(AgoConfigScaleMatrix));
	unsigned short *Xfrac = Xmap + alignW;
	unsigned short *One_min_xf = Xfrac + alignW;

	int xpos = xoffs;
	vx_uint32 newDstWidth = 0;
	for (unsigned int x = 0; x < dstWidth; x++, xpos += xinc)
	{
		int xf;
		int xmap = (xpos >> FP_BITS);
		if (xmap >= (int)(srcWidth - 1)){
			if (!newDstWidth) newDstWidth = x - 1;
			Xmap[x] = (unsigned short)(srcWidth - 1);
		}
		else {
			Xmap[x] = (xmap < 0) ? 0 : (unsigned short)xmap;
		}
		xf = ((xpos & 0x3ffff)+0x200)>>10;
		Xfrac[x] = xf;
		One_min_xf[x] = (0x100 - xf);
	}
	if (dstWidth & 7)
	{
		newDstWidth &= ~7;	// nearest multiple of 8
	}

	XMM128 pp1 = { 0 }, pp2 = { 0 };
	const __m128i mask = _mm_set1_epi16((short)0xff);
	const __m128i round = _mm_set1_epi16((short)0x80);
	for (int y = 0, ypos = yoffs; y < (int)dstHeight; y++, ypos += yinc)
	{
		int ym, yf, one_min_yf;
		__m128i rxmm0, rxmm7;
		unsigned int yoffs;
		vx_uint8 *pSrc1, *pSrc2;

		ym = (ypos >> FP_BITS);
		yf = ((ypos & 0x3ffff)+0x200)>>10;
		one_min_yf = (0x100 - yf);
		yoffs = ym*srcImageStrideInBytes;
		if (ym < 0){
			ym = yoffs = 0;
			pSrc1 = pSrc2 = pSrcImage;
		}
		else if (ym >= (int)(srcHeight - 1)){
			ym = srcHeight - 1;
			pSrc1 = pSrc2 = pSrcImage + ym*srcImageStrideInBytes;
		}
		else
		{
			pSrc1 = pSrcImage + ym*srcImageStrideInBytes;
			pSrc2 = pSrc1 + srcImageStrideInBytes;
		}
		rxmm0 = _mm_set1_epi16((unsigned short)one_min_yf);
		rxmm7 = _mm_set1_epi16((unsigned short)yf);
		unsigned int x = 0;
		for (; x < newDstWidth; x += 8)
		{
			__m128i mapxy, rxmm1, rxmm2, rxmm3, rxmm4;
			mapxy = _mm_load_si128((__m128i *)&Xmap[x]);		// mapped table [srcx7...src_x3,src_x2,src_x1,src_x0]

			// load pixels for mapxy
			for (int xx = 0; xx < 8; xx++)
			{
				pp1.u16[xx] = ((unsigned short*)&pSrc1[M128I(mapxy).m128i_i16[xx]])[0];
				pp2.u16[xx] = ((unsigned short*)&pSrc2[M128I(mapxy).m128i_i16[xx]])[0];
			}
			// unpack src for p1 and p2
			rxmm1 = _mm_and_si128(pp1.i, mask);		// p1
			pp1.i = _mm_srli_epi16(pp1.i, 8);		// p2
			// unpack pp2 for p3 and p4
			rxmm4 = _mm_and_si128(pp2.i, mask);		// p3
			pp2.i = _mm_srli_epi16(pp2.i, 8);		// p4

			// load xf and 1-xf
			rxmm2 = _mm_load_si128((__m128i *)&Xfrac[x]);			// xf
			rxmm3 = _mm_load_si128((__m128i *)&One_min_xf[x]);		// 1-xf

			// t1 = (unsigned char)((ione_minus_x *p1 + ifraction_x *p2) >> FW_WEIGHT); 
			rxmm1 = _mm_mullo_epi16(rxmm1, rxmm3);	//  ione_minus_xf *p1 	
			pp1.i = _mm_mullo_epi16(pp1.i, rxmm2);	//  ifraction_x  *p2		
			rxmm1 = _mm_add_epi16(rxmm1, pp1.i);
			rxmm1 = _mm_add_epi16(rxmm1, round);
			rxmm1 = _mm_srli_epi16(rxmm1, 8);

			//  t2 = (unsigned char)((ione_minus_x *p3 + ifraction_x *p4) >> FW_WEIGHT); 	
			rxmm4 = _mm_mullo_epi16(rxmm4, rxmm3);	//  ione_minus_x *p3 	
			pp2.i = _mm_mullo_epi16(pp2.i, rxmm2);	//  ifraction_x  *p4		
			rxmm4 = _mm_add_epi16(rxmm4, pp2.i);
			rxmm4 = _mm_add_epi16(rxmm4, round);
			rxmm4 = _mm_srli_epi16(rxmm4, 8);


			// *(pDst + x + y*dstStep) = (unsigned char)((ione_minus_y *t1 + ifraction_y * t2) >> FW_WEIGHT)	
			rxmm1 = _mm_mullo_epi16(rxmm1, rxmm0);	//  ione_minus_y * t1 	
			rxmm4 = _mm_mullo_epi16(rxmm4, rxmm7);	//  ifraction_y  * t2		
			rxmm1 = _mm_add_epi16(rxmm1, rxmm4);
			rxmm1 = _mm_add_epi16(rxmm1, round);
			rxmm1 = _mm_srli_epi16(rxmm1, 8);
			rxmm1 = _mm_packus_epi16(rxmm1, rxmm1);

			_mm_storel_epi64((__m128i *)(pDstImage + x), rxmm1);
		}
		// todo: if (upscale; recompute x=0, x=dwidth-1)
		if (matrix->xscale < 1){
			unsigned int p0, p1, p2, p3;
			p0 = p1 = pSrc1[0];
			p2 = p3 = pSrc2[0];
			pDstImage[0] = ((One_min_xf[0] * one_min_yf*p0) + (Xfrac[0] * one_min_yf*p1) + (One_min_xf[0] * yf*p2) + (Xfrac[0]*yf*p3) + 0x8000) >> 16;
		}
		x = newDstWidth;
		while (x < dstWidth){
			unsigned int p0, p1, p2, p3;
			const unsigned char *p = pSrc1 + Xmap[x];
			p0 = p[0];
			p1 = (Xmap[x] < (srcWidth - 1)) ? p[1] : p0;
			p = pSrc2 + Xmap[x];
			p2 = p[0];
			p3 = (Xmap[x] < (srcWidth - 1)) ? p[1]: p2;
			pDstImage[x] = ((One_min_xf[x] * one_min_yf*p0) + (Xfrac[x] * one_min_yf*p1) + (One_min_xf[x] * yf*p2) + (Xfrac[x] * yf*p3) + 0x8000) >> 16;
			x++;
		}

		pDstImage += dstImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

int HafCpu_ScaleImage_U8_U8_Bilinear_Constant
(
vx_uint32            dstWidth,
vx_uint32            dstHeight,
vx_uint8           * pDstImage,
vx_uint32            dstImageStrideInBytes,
vx_uint32            srcWidth,
vx_uint32            srcHeight,
vx_uint8           * pSrcImage,
vx_uint32            srcImageStrideInBytes,
ago_scale_matrix_t * matrix,
vx_uint8             border
)
{
	int xinc, yinc, xoffs, yoffs;

	unsigned int sline = srcImageStrideInBytes;
	unsigned char *pdst = pDstImage;
	unsigned char *pSrcLast = pSrcImage + (srcImageStrideInBytes*(srcWidth - 1));
	yinc = (int)(FP_MUL * matrix->yscale);		// to convert to fixed point
	xinc = (int)(FP_MUL * matrix->xscale);
	yoffs = (int)(FP_MUL * matrix->yoffset);		// to convert to fixed point
	xoffs = (int)(FP_MUL * matrix->xoffset);
	int alignW = (dstWidth + 15)&~15;
	unsigned short *Xmap = (unsigned short *)((vx_uint8*)matrix + sizeof(AgoConfigScaleMatrix));
	unsigned short *Xfrac = Xmap + alignW;
	unsigned short *One_min_xf = Xfrac + alignW;
	vx_uint8 *pSrcBorder = (vx_uint8 *)(One_min_xf + alignW);
	memset(pSrcBorder, border, srcWidth);

	int xpos = xoffs;
	vx_uint32 newDstWidth = 0;
	for (unsigned int x = 0; x < dstWidth; x++, xpos += xinc)
	{
		int xf;
		int xmap = (xpos >> FP_BITS);
		if (xmap >= (int)(srcWidth - 1)){
			if (!newDstWidth) newDstWidth = x - 1;
			Xmap[x] = (unsigned short)(srcWidth - 1);
		}
		else {
			Xmap[x] = (xmap < 0) ? 0 : (unsigned short)xmap;
		}
		xf = ((xpos & 0x3ffff)+0x200)>>10;
		Xfrac[x] = xf;
		One_min_xf[x] = (0x100 - xf);
	}
	if (dstWidth & 7)
	{
		newDstWidth &= ~7;	// nearest multiple of 8
	}

	XMM128 pp1 = { 0 }, pp2 = { 0 };
	const __m128i mask = _mm_set1_epi16((short)0xff);
	const __m128i round = _mm_set1_epi16((short)0x80);
	for (int y = 0, ypos = yoffs; y < (int)dstHeight; y++, ypos += yinc)
	{
		int ym, yf, one_min_yf;
		unsigned int yoffs;
		vx_uint8 *pSrc1, *pSrc2;

		ym = (ypos >> FP_BITS);
		yf = ((ypos & 0x3ffff)+0x200)>>10;
		one_min_yf = (0x100 - yf);
		if (ym < 0){
			ym = yoffs = 0;
			pSrc1 = pSrcBorder;
			pSrc2 = pSrcImage;
		}
		else if (ym >= (int)(srcHeight - 1)){
			ym = srcHeight - 1;
			pSrc1 = pSrcImage + ym*srcImageStrideInBytes;
			pSrc2 = pSrcBorder;
			yoffs = ym*srcImageStrideInBytes;
		}
		else
		{
			pSrc1 = pSrcImage + ym*srcImageStrideInBytes;
			pSrc2 = pSrc1 + srcImageStrideInBytes;
			yoffs = ym*srcImageStrideInBytes;
		}

		__m128i rxmm0, rxmm7;
		rxmm0 = _mm_set1_epi16((unsigned short)one_min_yf);
		rxmm7 = _mm_set1_epi16((unsigned short)yf);
		unsigned int x = 0;
		for (; x < newDstWidth; x += 8)
		{
			__m128i mapxy, rxmm1, rxmm2, rxmm3, rxmm4;
			mapxy = _mm_load_si128((__m128i *)&Xmap[x]);		// mapped table [srcx7...src_x3,src_x2,src_x1,src_x0]
			// load pixels for mapxy
			for (int xx = 0; xx < 8; xx++)
			{
				pp1.u16[xx] = ((unsigned short*)&pSrc1[M128I(mapxy).m128i_i16[xx]])[0];
				pp2.u16[xx] = ((unsigned short*)&pSrc2[M128I(mapxy).m128i_i16[xx]])[0];
			}
			// unpack src for p1 and p2
			rxmm1 = _mm_and_si128(pp1.i, mask);		// p1
			pp1.i = _mm_srli_epi16(pp1.i, 8);		// p2
			// unpack pp2 for p3 and p4
			rxmm4 = _mm_and_si128(pp2.i, mask);		// p3
			pp2.i = _mm_srli_epi16(pp2.i, 8);		// p4

			// load xf and 1-xf
			rxmm2 = _mm_load_si128((__m128i *)&Xfrac[x]);			// xf
			rxmm3 = _mm_load_si128((__m128i *)&One_min_xf[x]);		// 1-xf

			// t1 = (unsigned char)((ione_minus_x *p1 + ifraction_x *p2) >> FW_WEIGHT); 
			rxmm1 = _mm_mullo_epi16(rxmm1, rxmm3);	//  ione_minus_xf *p1 	
			pp1.i = _mm_mullo_epi16(pp1.i, rxmm2);	//  ifraction_x  *p2		
			rxmm1 = _mm_add_epi16(rxmm1, pp1.i);
			rxmm1 = _mm_add_epi16(rxmm1, round);
			rxmm1 = _mm_srli_epi16(rxmm1, 8);

			//  t2 = (unsigned char)((ione_minus_x *p3 + ifraction_x *p4) >> FW_WEIGHT); 	
			rxmm4 = _mm_mullo_epi16(rxmm4, rxmm3);	//  ione_minus_x *p3 	
			pp2.i = _mm_mullo_epi16(pp2.i, rxmm2);	//  ifraction_x  *p4		
			rxmm4 = _mm_add_epi16(rxmm4, pp2.i);
			rxmm4 = _mm_add_epi16(rxmm4, round);
			rxmm4 = _mm_srli_epi16(rxmm4, 8);


			// *(pDst + x + y*dstStep) = (unsigned char)((ione_minus_y *t1 + ifraction_y * t2) >> FW_WEIGHT)	
			rxmm1 = _mm_mullo_epi16(rxmm1, rxmm0);	//  ione_minus_y * t1 	
			rxmm4 = _mm_mullo_epi16(rxmm4, rxmm7);	//  ifraction_y  * t2		
			rxmm1 = _mm_add_epi16(rxmm1, rxmm4);
			rxmm1 = _mm_add_epi16(rxmm1, round);
			rxmm1 = _mm_srli_epi16(rxmm1, 8);
			rxmm1 = _mm_packus_epi16(rxmm1, rxmm1);

			_mm_storel_epi64((__m128i *)(pDstImage + x), rxmm1);
		}
		// todo: if (upscale; recompute x=0, x=dwidth-1)
		if (matrix->xscale < 1){
			unsigned int p0, p1, p2, p3;
			p0 = border;
			p1 = (ypos >> 8) < 0 ? border : pSrc1[0];
			p2 = border;
			p3 = pSrc2[0];
			pDstImage[0] = ((One_min_xf[0] * one_min_yf*p0) + (Xfrac[0] * one_min_yf*p1) + (One_min_xf[0] * yf*p2) + (Xfrac[0] * yf*p3) + 0x8000) >> 16;
		}
		x = newDstWidth ;
		while (x < dstWidth){
			unsigned int p0, p1, p2, p3;
			const unsigned char *p = pSrc1 + Xmap[x];
			p0 = p[0];
			p1 = (Xmap[x] < (srcWidth - 1)) ? p[1] : border;
			p = pSrc2 + Xmap[x];
			p2 = p[0];
			p3 = (Xmap[x] < (srcWidth - 1)) ? p[1] : border;
			pDstImage[x] = ((One_min_xf[x] * one_min_yf*p0) + (Xfrac[x] * one_min_yf*p1) + (One_min_xf[x] * yf*p2) + (Xfrac[x] * yf*p3) + 0x8000) >> 16;
			x++;
		}
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}


// upsample 2x2 (used for 4:2:0 to 4:4:4 conversion)
int HafCpu_ScaleUp2x2_U8_U8
(
vx_uint32     dstWidth,
vx_uint32     dstHeight,
vx_uint8    * pDstImage,
vx_uint32     dstImageStrideInBytes,
vx_uint8    * pSrcImage,
vx_uint32     srcImageStrideInBytes
)
{

	__m128i pixels1, pixels2;

	unsigned char *pchDst = (unsigned char*)pDstImage;
	unsigned char *pchDstlast = (unsigned char*)pDstImage + dstHeight*dstImageStrideInBytes;
	while (pchDst < pchDstlast)
	{
		__m128i * src = (__m128i*)pSrcImage;
		__m128i * dst = (__m128i*)pchDst;
		__m128i * dstNext = (__m128i*)(pchDst + dstImageStrideInBytes);
		__m128i * dstlast = dst + (dstWidth >> 4);

		bool fallbackToC = false;

		// Fallback to C if image width is not divisible by 16.
		// Or if image width is smaller then 16px.
		if ((dstWidth & 15) > 0)
			fallbackToC = true;

		// Fallback to C if image width requires odd number of vectors to process it.
		// Do not process the last, odd, vector with MSA.
		if (((dstWidth >> 4) % 2) == 1)
		{
			dstlast--;
			fallbackToC = true;
		}

		while (dst < dstlast)
		{
			pixels1 = _mm_loadu_si128(src++);		// src (0-15)
			pixels2 = _mm_unpacklo_epi8(pixels1, pixels1);		// dst (0-15)
			pixels1 = _mm_unpackhi_epi8(pixels1, pixels1);		// dst (16-31)
			_mm_store_si128(dst++, pixels2);
			_mm_store_si128(dst++, pixels1);
			_mm_store_si128(dstNext++, pixels2);
			_mm_store_si128(dstNext++, pixels1);
		}

		if (fallbackToC)
		{
			unsigned char *src = (unsigned char *) pSrcImage;
			unsigned char *dst = (unsigned char *) pchDst;
			unsigned char *dstNext = (unsigned char *) (pchDst + dstImageStrideInBytes);
			unsigned char *dstLast = dst + dstWidth - 1;

			while (dst < dstLast)
			{
				*dst++ = *src;
				*dst++ = *src;
				*dstNext++ = *src;
				*dstNext++ = *src++;
			}
		}

		pchDst += (dstImageStrideInBytes * 2);
		pSrcImage += srcImageStrideInBytes;
	}

	return AGO_SUCCESS;
}

int HafCpu_ScaleImage_U8_U8_Area
(
vx_uint32            dstWidth,
vx_uint32            dstHeight,
vx_uint8           * pDstImage,
vx_uint32            dstImageStrideInBytes,
vx_uint32            srcWidth,
vx_uint32            srcHeight,
vx_uint8           * pSrcImage,
vx_uint32            srcImageStrideInBytes,
ago_scale_matrix_t * matrix
)
{
	if (matrix->xscale == 1.0f && matrix->yscale == 1.0f)
	{
		vx_uint8 *pSrcB = pSrcImage + (dstHeight - 1)*srcImageStrideInBytes;
		// no scaling. Just do a copy from src to dst
		for (unsigned int y = 0; y < dstHeight; y++)
		{
			vx_uint8 *pSrc = pSrcImage + (int)(matrix->yoffset+y)*srcImageStrideInBytes + (int)matrix->xoffset;
			// clamp to boundary
			if (pSrc < pSrcImage) pSrc = pSrcImage;
			if (pSrc > pSrcB) pSrc = pSrcB;
			memcpy(pDstImage, pSrc, dstWidth);
			pDstImage += dstImageStrideInBytes;
		}
	} 
	else if (matrix->xscale == 2.0f && matrix->yscale == 2.0f)
	{
		__m128i zero = _mm_setzero_si128();
		__m128i delta2 = _mm_set1_epi16(2);
		__m128i masklow = _mm_set1_epi16(0x00ff);
		vx_uint8 *pSrcB = pSrcImage + (srcHeight - 2)*srcImageStrideInBytes;
		// 2x2 image scaling
		for (unsigned int y = 0; y < dstHeight; y++)
		{
			vx_uint8 *S0 = pSrcImage + (int)(matrix->yoffset+(y*2))*srcImageStrideInBytes + (int)(matrix->xoffset);
			if (S0 < pSrcImage) S0 = pSrcImage;
			if (S0 > pSrcB) S0 = pSrcB;
			vx_uint8 *S1 = S0 + srcImageStrideInBytes;
			vx_uint8 *D = pDstImage;
			for (unsigned int dx = 0; dx <= dstWidth - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
			{
				__m128i r0 = _mm_loadu_si128((const __m128i*)S0);
				__m128i r1 = _mm_loadu_si128((const __m128i*)S1);

				__m128i s0 = _mm_add_epi16(_mm_srli_epi16(r0, 8), _mm_and_si128(r0, masklow));
				__m128i s1 = _mm_add_epi16(_mm_srli_epi16(r1, 8), _mm_and_si128(r1, masklow));
				s0 = _mm_add_epi16(_mm_add_epi16(s0, s1), delta2);
				s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);

				_mm_storel_epi64((__m128i*)D, s0);
			}
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		int xinc, yinc, xoffs, yoffs, xpos, ypos, x, y;
		// Intermideate buffers to store results between horizontally filtered rows
		int alignWidth = (dstWidth + 15) & ~15;
		vx_uint16 *Xmap = (unsigned short *)((vx_uint8*)matrix + sizeof(ago_scale_matrix_t));
		vx_uint16 *Ymap = Xmap + alignWidth+8;
		__m128i z = _mm_setzero_si128();
		// do generic area scaling
		yinc = (int)(FP_MUL * matrix->yscale);		// to convert to fixed point
		xinc = (int)(FP_MUL * matrix->xscale);
		yoffs = (int)(FP_MUL * matrix->yoffset);		// to convert to fixed point
		xoffs = (int)(FP_MUL * matrix->xoffset);
		int xscale = (int)(matrix->xscale + 0.5);
		int yscale = (int)(matrix->yscale + 0.5);
		float inv_scale = 1.0f / (xscale*yscale);
		int area_div = (int)(FP_MUL * inv_scale);
		vx_uint8 *src_b = pSrcImage + srcWidth*(srcHeight - 1);
		//int area_sz = (area + (1 << (FP_BITS - 1))) >> FP_BITS;
		// generate xmap;
		for (x = 0, xpos = xoffs; x <= (int)dstWidth; x++, xpos += xinc)
		{
			int xmap;
			xmap = ((xpos + FP_ROUND) >> FP_BITS);
			if (xmap >(int)(srcWidth - 1)){
				xmap = (srcWidth - 1);
			}
			if (xmap < 0) xmap = 0;
			Xmap[x] = (unsigned short)xmap;
		}
		for (y = 0, ypos = yoffs; y < (int)dstHeight; y++, ypos += yinc)
		{
			int ymap;
			ymap = ((ypos + FP_ROUND )>> FP_BITS);
			if (ymap >(int)(srcHeight - 1)){
				ymap = srcHeight - 1;
			}
			if (ymap < 0) ymap = 0;
			// compute vertical sum and store in intermediate buffer
			vx_uint8 *S0 = pSrcImage + (int)ymap*srcImageStrideInBytes;
			vx_uint8 *D = pDstImage;
			for (x = Xmap[0]; x <= (Xmap[dstWidth] - 7); x += 8)
			{
				__m128i r0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(S0 + x)), z);
				vx_uint8 *S1 = S0 + srcImageStrideInBytes;
				for (int i = 1; i < yscale; i++){
					if (S1 > src_b)S1 = src_b;
					__m128i r1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(S1 + x)), z);
					r0 = _mm_add_epi16(r0, r1);
					S1 += srcImageStrideInBytes;
				}
				_mm_store_si128((__m128i*)&Ymap[x], r0);
			}
			// do horizontal scaling on intermediate buffer
			for (x = 0; x < (int)dstWidth; x++)
			{
				int x0 = Xmap[x];
				int x1 = x0 + xscale;
				int sum = Ymap[x0];
				while(++x0<x1) {
					sum += Ymap[x0];
				};
				// divide sum by area and copy to dest
				*D++ = (vx_uint8)(((sum*area_div) + (1 << 15)) >> FP_BITS);

			}
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_ScaleImage_U8_U8_Area_Constant
(
vx_uint32            dstWidth,
vx_uint32            dstHeight,
vx_uint8           * pDstImage,
vx_uint32            dstImageStrideInBytes,
vx_uint32            srcWidth,
vx_uint32            srcHeight,
vx_uint8           * pSrcImage,
vx_uint32            srcImageStrideInBytes,
ago_scale_matrix_t * matrix,
vx_uint8             border
)
{
	if (matrix->xscale == 1.0f && matrix->yscale == 1.0f)
	{
		vx_uint8 *pSrcB = pSrcImage + (dstHeight - 1)*srcImageStrideInBytes;
		// no scaling. Just do a copy from src to dst
		for (unsigned int y = 0; y < dstHeight; y++)
		{
			vx_uint8 *pSrc = pSrcImage + (int)(matrix->yoffset + y)*srcImageStrideInBytes + (int)matrix->xoffset;
			// clamp to boundary
			if ((pSrc < pSrcImage) || (pSrc > pSrcB)){
				memset(pDstImage, border, dstWidth) ;
			}
			else
				memcpy(pDstImage, pSrc, dstWidth);
			pDstImage += dstImageStrideInBytes;
		}
	}
	else if (matrix->xscale == 2.0f && matrix->yscale == 2.0f)
	{
		__m128i zero = _mm_setzero_si128();
		__m128i delta2 = _mm_set1_epi16(2);
		__m128i masklow = _mm_set1_epi16(0x00ff);
		__m128i bound = _mm_set1_epi16(border);
		vx_uint8 *pSrcB = pSrcImage + (srcHeight - 2)*srcImageStrideInBytes;
		// 2x2 image scaling
		for (unsigned int y = 0; y < dstHeight; y++)
		{
			vx_uint8 *S0 = pSrcImage + (int)(matrix->yoffset + (y*2))*srcImageStrideInBytes + (int)(matrix->xoffset);
			if (S0 < pSrcImage) S0 = pSrcImage;
			if (S0 > pSrcB) S0 = pSrcB;
			vx_uint8 *S1 = S0 + srcImageStrideInBytes;
			vx_uint8 *D = pDstImage;
			for (unsigned int dx = 0; dx <= dstWidth - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
			{
				__m128i r0 = _mm_loadu_si128((const __m128i*)S0);
				__m128i r1 = _mm_loadu_si128((const __m128i*)S1);

				__m128i s0 = _mm_add_epi16(_mm_srli_epi16(r0, 8), _mm_and_si128(r0, masklow));
				__m128i s1 = _mm_add_epi16(_mm_srli_epi16(r1, 8), _mm_and_si128(r1, masklow));
				s0 = _mm_add_epi16(_mm_add_epi16(s0, s1), delta2);
				s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);

				_mm_storel_epi64((__m128i*)D, s0);
			}
			pDstImage += dstImageStrideInBytes;
		}
	}
	else
	{
		int xinc, yinc, xoffs, yoffs, xpos, ypos, x, y;
		vx_uint16 *Xmap = (unsigned short *)((vx_uint8*)matrix + sizeof(ago_scale_matrix_t));
		vx_uint16 *Ymap = Xmap + dstWidth;
		__m128i z = _mm_setzero_si128();
		// do generic area scaling
		yinc = (int)(FP_MUL * matrix->yscale);		// to convert to fixed point
		xinc = (int)(FP_MUL * matrix->xscale);
		yoffs = (int)(FP_MUL * matrix->yoffset);		// to convert to fixed point
		xoffs = (int)(FP_MUL * matrix->xoffset);
		int xscale = (int)(matrix->xscale + 0.5);
		int yscale = (int)(matrix->yscale + 0.5);
		float inv_scale = 1.0f / (xscale*yscale);
		int area_div = (int)(FP_MUL * inv_scale);
		vx_uint8 *src_b = pSrcImage + srcWidth*(srcHeight - 1);
		// generate xmap;
		for (x = 0, xpos = xoffs; x <= (int)dstWidth; x++, xpos += xinc)
		{
			int xmap;
			xmap = ((xpos + FP_ROUND) >> FP_BITS);
			if (xmap >(int)(srcWidth - 1)){
				xmap = (srcWidth - 1);
			}
			if (xmap < 0) xmap = 0;
			Xmap[x] = (unsigned short)xmap;
		}
		for (y = 0, ypos = yoffs; y < (int)dstHeight; y++, ypos += yinc)
		{
			int ymap;
			ymap = ((ypos + FP_ROUND) >> FP_BITS);
			if (ymap >(int)(srcHeight - 1)){
				ymap = srcHeight - 1;
			}
			if (ymap < 0) ymap = 0;
			// compute vertical sum and store in intermediate buffer
			vx_uint8 *S0 = pSrcImage + (int)ymap*srcImageStrideInBytes;
			vx_uint8 *D = pDstImage;
			for (x = Xmap[0]; x <= (Xmap[dstWidth] - 7); x += 8)
			{
				__m128i r0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(S0 + x)), z);
				vx_uint8 *S1 = S0 + srcImageStrideInBytes;
				for (int i = 1; i < yscale; i++){
					if (S1 > src_b)S1 = src_b;
					__m128i r1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(S1 + x)), z);
					r0 = _mm_add_epi16(r0, r1);
					S1 += srcImageStrideInBytes;
				}
				_mm_store_si128((__m128i*)&Ymap[x], r0);
			}
			// do horizontal scaling on intermediate buffer
			for (x = 0; x < (int)dstWidth; x++)
			{
				int x0 = Xmap[x];
				int x1 = x0 + xscale;
				int sum = Ymap[x0];
				while (++x0<x1) {
					sum += Ymap[x0];
				};
				// divide sum by area and copy to dest
				*D++ = (vx_uint8)(((sum*area_div) + (1 << 15)) >> FP_BITS);

			}
			pDstImage += dstImageStrideInBytes;
		}
	}
	return AGO_SUCCESS;
}

int HafCpu_ScaleImage_U8_U8_Area_Replicate
(
vx_uint32            dstWidth,
vx_uint32            dstHeight,
vx_uint8           * pDstImage,
vx_uint32            dstImageStrideInBytes,
vx_uint32            srcWidth,
vx_uint32            srcHeight,
vx_uint8           * pSrcImage,
vx_uint32            srcImageStrideInBytes,
ago_scale_matrix_t * matrix
)
{
	return HafCpu_ScaleImage_U8_U8_Area(dstWidth, dstHeight, pDstImage, dstImageStrideInBytes, srcWidth, srcHeight, pSrcImage, srcImageStrideInBytes, matrix);
}

/*
Performs a Gaussian blur(3x3) and half scales it
gaussian filter
Kernel			1   2   1 			1		1   2   1
				2   4   2 			2					>>4
				1   2   1     =		1									
*/

int HafCpu_ScaleGaussianHalf_U8_U8_3x3
(
	vx_uint32     dstWidth,
	vx_uint32     dstHeight,
	vx_uint8    * pDstImage,
	vx_uint32     dstImageStrideInBytes,
	vx_uint8    * pSrcImage,
	vx_uint32     srcImageStrideInBytes,
	vx_uint8    * pLocalData
)
{
	unsigned int x, y;
	//	float scale = (float)128 / 180.f;

	pSrcImage += srcImageStrideInBytes;
	__m128i z = _mm_setzero_si128(), mask = _mm_set1_epi32((int)0x0000FFFF);
	vx_uint16 *r0 = (vx_uint16*)(pLocalData + 16);
	unsigned int W = 2 * dstWidth;

	for (y = 0; y < dstHeight; y++)
	{
		const vx_uint8* srow0 = pSrcImage - srcImageStrideInBytes;
		const vx_uint8* srow1 = pSrcImage;
		const vx_uint8* srow2 = pSrcImage + srcImageStrideInBytes;
		vx_uint8* pDst = (vx_uint8*)pDstImage;

		// do vertical convolution
		x = 0;
		for (; x < W; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i t0 = _mm_add_epi16(_mm_add_epi16(s0, s2), _mm_slli_epi16(s1, 1));
			_mm_store_si128((__m128i*)(r0 + x), t0);
		}

		// do horizontal convolution, interleave the results and store them to dst
		x = 0;
		for (; x < W; x += 16, pDst+=8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(r0 + x - 1));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(r0 + x));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(r0 + x + 1));

			__m128i t0 = _mm_add_epi16(_mm_add_epi16(s0, s2), _mm_slli_epi16(s1, 1));
			s0 = _mm_loadu_si128((const __m128i*)(r0 + x + 7));
			s1 = _mm_loadu_si128((const __m128i*)(r0 + x + 8));
			s2 = _mm_loadu_si128((const __m128i*)(r0 + x + 9));
			s0 = _mm_add_epi16(_mm_add_epi16(s0, s2), _mm_slli_epi16(s1, 1));

			t0 = _mm_packus_epi32(_mm_and_si128(t0, mask), _mm_and_si128(s0, mask));
			t0 = _mm_srli_epi16(t0, 4);
			t0 = _mm_packus_epi16(t0, t0);
			_mm_storel_epi64((__m128i*)pDst, t0);
		}
		pSrcImage += (srcImageStrideInBytes + srcImageStrideInBytes);	// do alternate rows for /2 scaling
		pDstImage += dstImageStrideInBytes;
	}
	return AGO_SUCCESS;
}
