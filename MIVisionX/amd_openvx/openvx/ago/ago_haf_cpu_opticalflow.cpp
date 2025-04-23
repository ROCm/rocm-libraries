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

/* Implements OpticalFlow pyramid algorithm*/

typedef struct {
	vx_float32 x;                 /*!< \brief The x coordinate. */
	vx_float32 y;                 /*!< \brief The y coordinate. */
}pt2f;

typedef struct {
	vx_int32 x;                 /*!< \brief The x coordinate. */
	vx_int32 y;                 /*!< \brief The y coordinate. */
}pt2i;

// helper functions for floating point math: used in haf_cpu implementation
// flRound: convert to nearest integer
inline int flRound(float value)
{
	__m128d t = _mm_set_sd(value);
	return _mm_cvtsd_si32(t);
}

inline int flFloor(float value)
{
	int i = flRound(value);
	return i - (((float)(value - 1)) < 0);
}

static const int W_BITS = 14;
static const float FLT_SCALE = 1.f / (1 << 20);
static const float MinEugThreshold = 1.0e-04F;
static const float Epsilon = 1.0e-07F;

#define DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

// helper functions
static inline void pt_copy(pt2f &pt1, pt2f &pt2) { pt1.x = pt2.x; pt1.y = pt2.y; }
static inline void pt_copy_scale(pt2f &pt1, pt2f &pt2, float &s) { pt1.x = pt2.x*s; pt1.y = pt2.y*s; }



static void ComputeSharr(
	vx_uint32   dstImageStrideInBytes,
	vx_uint8	*dst,
	vx_uint32	srcWidth,
	vx_uint32   srcHeight,
	vx_uint32   srcImageStrideInBytes,
	vx_uint8	*src,
	vx_uint8	*pScharrScratch
)
{
	unsigned int y,x;
	__m128i z = _mm_setzero_si128(), c3 = _mm_set1_epi16(3), c10 = _mm_set1_epi16(10);
	vx_uint16	*_tempBuf = (vx_uint16*)pScharrScratch;
	vx_uint16	*trow0 = (vx_uint16 *)ALIGN16(_tempBuf+1);
	vx_uint16   *trow1 = (vx_uint16 *)ALIGN16(trow0 + srcWidth+2);

#if 0		// C reference code for testing
	vx_int16 ops[] = { 3, 10, 3, -3, -10, -3 };
	src += srcImageStrideInBytes;
	dst += dstImageStrideInBytes;
	for (y = 1; y < srcHeight - 1; y++)
	{
		const vx_uint8* srow0 = src - srcImageStrideInBytes;
		const vx_uint8* srow1 = src;
		const vx_uint8* srow2 = src + srcImageStrideInBytes;
		vx_int16* drow = (vx_int16*)dst;
		drow+=2;
		for (x = 1; x < srcWidth - 1; x++, drow+=2)
		{
			// calculate g_x
			drow[0] = (srow0[x + 1] * ops[0]) + (srow1[x + 1] * ops[1]) + (srow2[x + 1] * ops[2]) +
				(srow0[x - 1] * ops[3]) + (srow1[x - 1] * ops[4]) + (srow2[x - 1] * ops[5]);
			drow[1] = (srow2[x - 1] * ops[0]) + (srow2[x] * ops[1]) + (srow2[x + 1] * ops[2]) +
				(srow0[x - 1] * ops[3]) + (srow0[x] * ops[4]) + (srow0[x + 1] * ops[5]);
		}
		src += srcImageStrideInBytes;
		dst += dstImageStrideInBytes;
	}
#else
	src += srcImageStrideInBytes;
	dst += dstImageStrideInBytes;
	for (y = 1; y < srcHeight-1; y++)
	{
		const vx_uint8* srow0 = y > 0 ? src - srcImageStrideInBytes : src;
		const vx_uint8* srow1 = src;
		const vx_uint8* srow2 = y < srcHeight - 1 ? src + srcImageStrideInBytes : src;
		vx_uint16* drow = (vx_uint16*)dst;

		// do vertical convolution
		x = 0;
		for (; x <= srcWidth - 8; x += 8)
		{
			__m128i s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow0 + x)), z);
			__m128i s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow1 + x)), z);
			__m128i s2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(srow2 + x)), z);
			__m128i t0 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s0, s2), c3), _mm_mullo_epi16(s1, c10));
			__m128i t1 = _mm_sub_epi16(s2, s0);
			_mm_store_si128((__m128i*)(trow0 + x), t0);
			_mm_store_si128((__m128i*)(trow1 + x), t1);
		}
		// make border: is this really needed.
		//trow0[-1] = trow0[0]; trow0[srcWidth] = trow0[srcWidth-1];
		//trow1[-1] = trow1[0]; trow1[srcWidth] = trow1[srcWidth - 1];

		// do horizontal convolution, interleave the results and store them to dst
		x = 0;
		for (; x <= srcWidth - 8; x += 8)
		{
			__m128i s0 = _mm_loadu_si128((const __m128i*)(trow0 + x - 1));
			__m128i s1 = _mm_loadu_si128((const __m128i*)(trow0 + x + 1));
			__m128i s2 = _mm_loadu_si128((const __m128i*)(trow1 + x - 1));
			__m128i s3 = _mm_loadu_si128((const __m128i*)(trow1 + x));
			__m128i s4 = _mm_loadu_si128((const __m128i*)(trow1 + x + 1));

			__m128i t0 = _mm_sub_epi16(s1, s0);
			__m128i t1 = _mm_add_epi16(_mm_mullo_epi16(_mm_add_epi16(s2, s4), c3), _mm_mullo_epi16(s3, c10));
			__m128i t2 = _mm_unpacklo_epi16(t0, t1);
			t0 = _mm_unpackhi_epi16(t0, t1);
			// this can probably be replaced with aligned stores if we aligned dst properly.
			_mm_storeu_si128((__m128i*)(drow + x * 2), t2);
			_mm_storeu_si128((__m128i*)(drow + x * 2 + 8), t0);
		}
		src += srcImageStrideInBytes;
		dst += dstImageStrideInBytes;
	}
#endif
}

int HafCpu_OpticalFlowPyrLK_XY_XY_Generic
(
vx_keypoint_t      newKeyPoint[],
vx_float32         pyramidScale,
vx_uint32          pyramidLevelCount,
ago_pyramid_u8_t * oldPyramid,
ago_pyramid_u8_t * newPyramid,
vx_uint32          keyPointCount,
vx_keypoint_t      oldKeyPoint[],
vx_keypoint_t      newKeyPointEstimate[],
vx_enum            termination,
vx_float32         epsilon,
vx_uint32          num_iterations,
vx_bool            use_initial_estimate,
vx_uint32		   dataStrideInBytes,
vx_uint8		 * DataPtr,
vx_int32		   winsz
)
{
	vx_size halfWin = (vx_size)(winsz>>1);  //(winsz *0.5f);
	__m128i z = _mm_setzero_si128();
	__m128i qdelta_d = _mm_set1_epi32(1 << (W_BITS - 1));
	__m128i qdelta = _mm_set1_epi32(1 << (W_BITS - 5 - 1));
	// allocate matrix for I and dI 
	vx_int16 Imat[256];				// enough to accomodate max win size of 15
	vx_int16 dIMat[256*2];
	vx_uint8 * pScharrScratch = DataPtr;
	vx_uint8 * pScratch = DataPtr + (oldPyramid[0].width + 2) * 4 + 64;
	ago_keypoint_t *pNextPtArray = (ago_keypoint_t *)(pScratch + (oldPyramid[0].width*oldPyramid[0].height * 4));

	for (int level = pyramidLevelCount - 1; level >= 0; level--)
	{
		int bBound;
		vx_uint32 dWidth = oldPyramid[level].width-2;
		vx_uint32 dHeight = oldPyramid[level].height-2;			// first and last row is not accounted
		vx_uint32 JWidth = newPyramid[level].width;
		vx_uint32 JHeight = newPyramid[level].height;
		vx_uint32 IStride = oldPyramid[level].strideInBytes, JStride = newPyramid[level].strideInBytes;
		vx_uint32 dStride = dataStrideInBytes>>1;		//in #of elements
		vx_uint8 *SrcBase = oldPyramid[level].pImage;
		vx_uint8 *JBase = newPyramid[level].pImage;
		vx_int16 *DIBase = (vx_int16 *)pScratch;

		// calculate sharr derivatives Ix and Iy
		ComputeSharr(dataStrideInBytes, pScratch, oldPyramid[level].width, oldPyramid[level].height, oldPyramid[level].strideInBytes, oldPyramid[level].pImage, pScharrScratch);
		float ptScale = (float)(pow(pyramidScale, level));

		// do the Lukas Kanade tracking for each feature point
		for (unsigned int pt = 0; pt < keyPointCount; pt++){
			if (!oldKeyPoint[pt].tracking_status)	{
				newKeyPoint[pt].x = oldKeyPoint[pt].x;
				newKeyPoint[pt].y = oldKeyPoint[pt].y;
				newKeyPoint[pt].strength = oldKeyPoint[pt].strength;
				newKeyPoint[pt].tracking_status = oldKeyPoint[pt].tracking_status;
				newKeyPoint[pt].scale = oldKeyPoint[pt].scale;
				newKeyPoint[pt].error = oldKeyPoint[pt].error;
				continue;
			}
			
			pt2f PrevPt, nextPt;
			bool bUseIE = false;
			PrevPt.x = oldKeyPoint[pt].x*ptScale;
			PrevPt.y = oldKeyPoint[pt].y*ptScale;
			if (level == pyramidLevelCount-1){
				if (use_initial_estimate){
					nextPt.x = newKeyPointEstimate[pt].x*ptScale;
					nextPt.y = newKeyPointEstimate[pt].y*ptScale;
					bUseIE = true;
					newKeyPoint[pt].strength = newKeyPointEstimate[pt].strength;
					newKeyPoint[pt].tracking_status = newKeyPointEstimate[pt].tracking_status;
					newKeyPoint[pt].error = newKeyPointEstimate[pt].error;
				}
				else
				{
					pt_copy(nextPt, PrevPt);
					newKeyPoint[pt].tracking_status = oldKeyPoint[pt].tracking_status;
					newKeyPoint[pt].strength = oldKeyPoint[pt].strength;
				}
				pNextPtArray[pt].x = nextPt.x;
				pNextPtArray[pt].y = nextPt.y;
			}
			else
			{
				pNextPtArray[pt].x *= 2.0f;
				pNextPtArray[pt].y *= 2.0f;
				nextPt.x = pNextPtArray[pt].x;
				nextPt.y = pNextPtArray[pt].y;
			}

			if (!newKeyPoint[pt].tracking_status){
				continue;
			}

			pt2i iprevPt, inextPt;
			PrevPt.x = PrevPt.x - halfWin;
			PrevPt.y = PrevPt.y - halfWin;
			nextPt.x = nextPt.x - halfWin;
			nextPt.y = nextPt.y - halfWin;

			iprevPt.x = (vx_int32)floor(PrevPt.x);
			iprevPt.y = (vx_int32)floor(PrevPt.y);
			// check if the point is out of bounds in the derivative image
			bBound = (iprevPt.x >> 31) | (iprevPt.x >= (vx_int32)(dWidth - winsz)) | (iprevPt.y >> 31) | (iprevPt.y >= (vx_int32)(dHeight - winsz));
			if (bBound){
				if (!level){
					newKeyPoint[pt].x = (vx_int32)nextPt.x;
					newKeyPoint[pt].y = (vx_int32)nextPt.y;
					newKeyPoint[pt].tracking_status = 0;
					newKeyPoint[pt].error = 0;
				}
				continue;	// go to next point.
			}
			// calulate weights for interpolation
			float a = PrevPt.x - iprevPt.x;
			float b = PrevPt.y - iprevPt.y;
			float A11 = 0, A12 = 0, A22 = 0;
			int x, y;
			int iw00, iw01, iw10, iw11;
			if ((a==0.0) && (b==0.0))
			{
				// no need to do interpolation for the source and derivatives
				int x, y;
				for (y = 0; y < winsz; y++)
				{
					const unsigned char* src = SrcBase + (y + iprevPt.y)*IStride + iprevPt.x;
					const vx_int16* dsrc = DIBase + (y + iprevPt.y)*dStride + iprevPt.x * 2;

					vx_int16* Iptr = &Imat[y*winsz];
					vx_int16* dIptr = &dIMat[y*winsz * 2];
					x = 0;
					for (; x < winsz - 4; x += 4, dsrc += 8, dIptr += 8)
					{
						__m128i v00, v01, v10, v11, v12;
						v00 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x)), z);
						v01 = _mm_loadu_si128((const __m128i*)(dsrc));
						v10 = _mm_shufflelo_epi16(v01, 0xd8);		// copy with shuffle
						v10 = _mm_shufflehi_epi16(v10, 0xd8);		// iy3, iy2, ix3,ix2, iy1, iy0, ix1,ix0
						v10 = _mm_shuffle_epi32(v10, 0xd8);			// iy3, iy2, iy1, iy0, ix3,ix2, ix1,ix0
						v11 = _mm_shuffle_epi32(v10, 0xe4);			// copy
						v12 = _mm_shuffle_epi32(v10, 0x4e);         // ix3,ix2, ix1,ix0, iy3, iy2, iy1, iy0
						v00 = _mm_slli_epi16(v00, 5);
						v12 = _mm_madd_epi16(v12, v10);			// A121, A120
						v10 = _mm_madd_epi16(v10, v11);			// A221, A220, A111, A110
						A11 += (float)(M128I(v10).m128i_i32[0] + M128I(v10).m128i_i32[1]);
						A22 += (float)(M128I(v10).m128i_i32[2] + M128I(v10).m128i_i32[3]);
						A12 += (float)(M128I(v12).m128i_i32[0] + M128I(v12).m128i_i32[1]);
						_mm_storeu_si128((__m128i*)dIptr, v01);
						_mm_storel_epi64((__m128i*)(Iptr + x), v00);
					}
					for (; x < winsz; x ++, dsrc += 2, dIptr += 2)
					{

						int ival = (src[x]<<5);
						int ixval = dsrc[0];
						int iyval = dsrc[1];

						Iptr[x] = (short)ival;
						dIptr[0] = (short)ixval;
						dIptr[1] = (short)iyval;

						A11 += (float)(ixval*ixval);
						A12 += (float)(ixval*iyval);
						A22 += (float)(iyval*iyval);
					}
				}
				A11 *= FLT_SCALE;
				A12 *= FLT_SCALE;
				A22 *= FLT_SCALE;
			}
			else
			{
				int iw00 = (int)(((1.f - a)*(1.f - b)*(1 << W_BITS)) + 0.5);
				int iw01 = (int)((a*(1.f - b)*(1 << W_BITS)) + 0.5);
				int iw10 = (int)(((1.f - a)*b*(1 << W_BITS)) + 0.5);
				int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
				__m128i qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
				__m128i qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
				__m128 qA11 = _mm_setzero_ps(), qA12 = _mm_setzero_ps(), qA22 = _mm_setzero_ps();
				// extract the patch from the old image, compute covariation matrix of derivatives
				for (y = 0; y < winsz; y++)
				{
					const unsigned char* src = SrcBase + (y + iprevPt.y)*IStride + iprevPt.x;
					const vx_int16* dsrc = DIBase + (y + iprevPt.y)*dStride + iprevPt.x * 2;

					vx_int16* Iptr = &Imat[y*winsz];
					vx_int16* dIptr = &dIMat[y*winsz * 2];

					x = 0;
					for (; x <= winsz - 4; x += 4, dsrc += 4 * 2, dIptr += 4 * 2)
					{
						__m128i v00, v01, v10, v11, t0, t1;

						v00 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x)), z);
						v01 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + 1)), z);
						v10 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + IStride)), z);
						v11 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + IStride + 1)), z);

						t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
							_mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
						t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS - 5);
						_mm_storel_epi64((__m128i*)(Iptr + x), _mm_packs_epi32(t0, t0));

						v00 = _mm_loadu_si128((const __m128i*)(dsrc));
						v01 = _mm_loadu_si128((const __m128i*)(dsrc + 2));
						v10 = _mm_loadu_si128((const __m128i*)(dsrc + dStride));
						v11 = _mm_loadu_si128((const __m128i*)(dsrc + dStride + 2));

						t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
							_mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
						t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
							_mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
						t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta_d), W_BITS);
						t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta_d), W_BITS);
						v00 = _mm_packs_epi32(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...

						_mm_storeu_si128((__m128i*)dIptr, v00);
						t0 = _mm_srai_epi32(v00, 16); // Iy0 Iy1 Iy2 Iy3
						t1 = _mm_srai_epi32(_mm_slli_epi32(v00, 16), 16); // Ix0 Ix1 Ix2 Ix3

						__m128 fy = _mm_cvtepi32_ps(t0);
						__m128 fx = _mm_cvtepi32_ps(t1);

						qA22 = _mm_add_ps(qA22, _mm_mul_ps(fy, fy));
						qA12 = _mm_add_ps(qA12, _mm_mul_ps(fx, fy));
						qA11 = _mm_add_ps(qA11, _mm_mul_ps(fx, fx));
					}
					// do computation for remaining x if any
					for (; x < winsz; x++, dsrc += 2, dIptr += 2)
					{
						int ival = DESCALE(src[x] * iw00 + src[x + 1] * iw01 +
							src[x + IStride] * iw10 + src[x + IStride + 1] * iw11, W_BITS - 5);
						int ixval = DESCALE(dsrc[0] * iw00 + dsrc[2] * iw01 +
							dsrc[dStride] * iw10 + dsrc[dStride + 2] * iw11, W_BITS);
						int iyval = DESCALE(dsrc[1] * iw00 + dsrc[3] * iw01 + dsrc[dStride + 1] * iw10 +
							dsrc[dStride + 3] * iw11, W_BITS);

						Iptr[x] = (short)ival;
						dIptr[0] = (short)ixval;
						dIptr[1] = (short)iyval;

						A11 += (float)(ixval*ixval);
						A12 += (float)(ixval*iyval);
						A22 += (float)(iyval*iyval);
					}
				}
				// add with SSE output
				if (winsz >= 4){
					float DECL_ALIGN(16) A11buf[4] ATTR_ALIGN(16), A12buf[4] ATTR_ALIGN(16), A22buf[4] ATTR_ALIGN(16);
					_mm_store_ps(A11buf, qA11);
					_mm_store_ps(A12buf, qA12);
					_mm_store_ps(A22buf, qA22);
					A11 += A11buf[0] + A11buf[1] + A11buf[2] + A11buf[3];
					A12 += A12buf[0] + A12buf[1] + A12buf[2] + A12buf[3];
					A22 += A22buf[0] + A22buf[1] + A22buf[2] + A22buf[3];
				}
				A11 *= FLT_SCALE;
				A12 *= FLT_SCALE;
				A22 *= FLT_SCALE;
			}

			float D = A11*A22 - A12*A12;
			float minEig = (A22 + A11 - std::sqrt((A11 - A22)*(A11 - A22) +
				4.f*A12*A12)) / (2 * winsz*winsz);

			if (minEig < 1.0e-04F || D < 1.0e-07F)
			{
				if (!level){
					newKeyPoint[pt].x = (vx_int32)nextPt.x;
					newKeyPoint[pt].y = (vx_int32)nextPt.y;
					newKeyPoint[pt].tracking_status = 0;
					newKeyPoint[pt].error = 0;
				}
				continue;
			}
			D = 1.f / D;
			float prevDelta_x = 0.f, prevDelta_y = 0.f;
			float delta_dx = 0.f, delta_dy = 0.f;
			unsigned int j = 0;
			while (j < num_iterations || termination == VX_TERM_CRITERIA_EPSILON)
			{
				__m128i qw0, qw1;
				inextPt.x = (vx_int32)floor(nextPt.x);
				inextPt.y = (vx_int32)floor(nextPt.y);
				bBound = (inextPt.x >> 31) | (inextPt.x >=(vx_int32)(JWidth - winsz)) | (inextPt.y >> 31) | (inextPt.y >= (vx_int32)(JHeight - winsz));
				if (bBound){
					if (!level){
						newKeyPoint[pt].tracking_status = 0;
						newKeyPoint[pt].error = 0;
					}
					break;	// go to next point.
				}
				a = nextPt.x - inextPt.x;
				b = nextPt.y - inextPt.y;
				iw00 = (int)(((1.f - a)*(1.f - b)*(1 << W_BITS)) +0.5);
				iw01 = (int)((a*(1.f - b)*(1 << W_BITS)) + 0.5);
				iw10 = (int)(((1.f - a)*b*(1 << W_BITS))+0.5);
				iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
				double ib1 = 0, ib2 = 0;
				float b1, b2;
				//double b1, b2;
				qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
				qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
				__m128 qb0 = _mm_setzero_ps(), qb1 = _mm_setzero_ps();
				for (y = 0; y < winsz; y++)
				{
					const unsigned char* Jptr = JBase + (y + inextPt.y)*JStride + inextPt.x;;
					vx_int16* Iptr = &Imat[y*winsz];
					vx_int16* dIptr = &dIMat[y*winsz*2];

					x = 0;
					for (; x <= winsz - 8; x += 8, dIptr += 8 * 2)
					{
						__m128i diff0 = _mm_loadu_si128((const __m128i*)(Iptr + x)), diff1;
						__m128i v00 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x)), z);
						__m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + 1)), z);
						__m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + JStride)), z);
						__m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + JStride + 1)), z);

						__m128i t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
							_mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
						__m128i t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
							_mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
						t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS - 5);
						t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS - 5);
						diff0 = _mm_subs_epi16(_mm_packs_epi32(t0, t1), diff0);
						diff1 = _mm_unpackhi_epi16(diff0, diff0);
						diff0 = _mm_unpacklo_epi16(diff0, diff0); // It0 It0 It1 It1 ...
						v00 = _mm_loadu_si128((const __m128i*)(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
						v01 = _mm_loadu_si128((const __m128i*)(dIptr + 8));
						v10 = _mm_mullo_epi16(v00, diff0);
						v11 = _mm_mulhi_epi16(v00, diff0);
						v00 = _mm_unpacklo_epi16(v10, v11);
						v10 = _mm_unpackhi_epi16(v10, v11);
						qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
						qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
						v10 = _mm_mullo_epi16(v01, diff1);
						v11 = _mm_mulhi_epi16(v01, diff1);
						v00 = _mm_unpacklo_epi16(v10, v11);
						v10 = _mm_unpackhi_epi16(v10, v11);
						qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
						qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
					}
					for (; x < winsz; x++, dIptr += 2)
					{
						int diff = DESCALE(Jptr[x] * iw00 + Jptr[x + 1] * iw01 +
							Jptr[x + JStride] * iw10 + Jptr[x + JStride + 1] * iw11,
							W_BITS - 5);
						diff -= Iptr[x];
						ib1 += (float)(diff*dIptr[0]);
						ib2 += (float)(diff*dIptr[1]);
					}
				}
				if (winsz >= 8)
				{
					float DECL_ALIGN(16) bbuf[4] ATTR_ALIGN(16);
					_mm_store_ps(bbuf, _mm_add_ps(qb0, qb1));
					ib1 += bbuf[0] + bbuf[2];
					ib2 += bbuf[1] + bbuf[3];

				}
				b1 = (float)(ib1*FLT_SCALE);
				b2 = (float)(ib2*FLT_SCALE);
				// calculate delta
				float delta_x = (float)((A12*b2 - A22*b1) * D);
				float delta_y = (float)((A12*b1 - A11*b2) * D);
				// add to nextPt
				nextPt.x += delta_x;
				nextPt.y += delta_y;
				if ((delta_x*delta_x + delta_y*delta_y) <= epsilon && (termination == VX_TERM_CRITERIA_EPSILON || termination == VX_TERM_CRITERIA_BOTH)){
					break;
				}
				if (j > 0 && abs(delta_x + prevDelta_x) < 0.01 && abs(delta_y + prevDelta_y) < 0.01)
				{
					delta_dx = delta_x*0.5f;
					delta_dy = delta_y*0.5f;
					break;
				}
				prevDelta_x = delta_x;
				prevDelta_y = delta_y;
				j++;
			}
			if (!level){
				newKeyPoint[pt].x = (vx_int32)(nextPt.x + halfWin - delta_dx + 0.5f);
				newKeyPoint[pt].y = (vx_int32)(nextPt.y + halfWin - delta_dy + 0.5f);
			}
			else
			{
				pNextPtArray[pt].x = (nextPt.x + halfWin - delta_dx);
				pNextPtArray[pt].y = (nextPt.y + halfWin - delta_dy);
			}
		}
	}
	return AGO_SUCCESS;
}
