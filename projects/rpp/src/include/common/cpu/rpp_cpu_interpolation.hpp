/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_CPU_INTERPOLATION_HPP
#define RPP_CPU_INTERPOLATION_HPP

/* Generic interpolation helper functions */

template <typename T>
inline void compute_generic_bilinear_srclocs_and_interpolate(T *srcPtrChannel, RpptDescPtr srcDescPtr, Rpp32f &srcY, Rpp32f &srcX, RpptROI* roiLTRB, T *dst)
{
    RpptPoint2D srcLT, srcRB;
    Rpp32f weightParams[4], bilinearCoeffs[4];
    Rpp32s srcLoc[4];
    srcLT.y = (Rpp32s) srcY;                                    // Bilinear LT point y value
    srcLT.y = std::min(srcLT.y, roiLTRB->ltrbROI.rb.y - 1);
    srcRB.y = std::min(srcLT.y + 1, roiLTRB->ltrbROI.rb.y - 1); // Bilinear RB point y value
    srcLT.x = (Rpp32s) srcX;                                    // Bilinear LT point x value
    srcLT.x = std::min(srcLT.x, roiLTRB->ltrbROI.rb.x - 1);
    srcRB.x = std::min(srcLT.x + 1, roiLTRB->ltrbROI.rb.x - 1); // Bilinear RB point x value
    weightParams[0] = srcY - srcLT.y;                           // weightedHeight
    weightParams[1] = 1 - weightParams[0];                      // 1 - weightedHeight
    weightParams[2] = srcX - srcLT.x;                           // weightedWidth
    weightParams[3] = 1 - weightParams[2];                      // 1 - weightedWidth
    bilinearCoeffs[0] = weightParams[1] * weightParams[3];      // (1 - weightedHeight) * (1 - weightedWidth)
    bilinearCoeffs[1] = weightParams[1] * weightParams[2];      // (1 - weightedHeight) * weightedWidth
    bilinearCoeffs[2] = weightParams[0] * weightParams[3];      // weightedHeight * (1 - weightedWidth)
    bilinearCoeffs[3] = weightParams[0] * weightParams[2];      // weightedHeight * weightedWidth
    srcLT.y *= srcDescPtr->strides.hStride;                     // LT Row * hStride
    srcRB.y *= srcDescPtr->strides.hStride;                     // RB Row * hStride
    srcLT.x *= srcDescPtr->strides.wStride;                     // LT Col * wStride
    srcRB.x *= srcDescPtr->strides.wStride;                     // LT Col * wStride
    srcLoc[0] = srcLT.y + srcLT.x;                              // Left-Top pixel memory location
    srcLoc[1] = srcLT.y + srcRB.x;                              // Right-Top pixel memory location
    srcLoc[2] = srcRB.y + srcLT.x;                              // Left-Bottom pixel memory location
    srcLoc[3] = srcRB.y + srcRB.x;                              // Right-Bottom pixel memory location

    for (int c = 0; c < srcDescPtr->c; c++)
    {
        if constexpr (std::is_same<T, Rpp8s>::value || std::is_same<T, Rpp8u>::value)
          dst[c] = (T)std::nearbyintf(((*(srcPtrChannel + srcLoc[0]) * bilinearCoeffs[0]) +        // TopRow R01 Pixel * coeff0
                    (*(srcPtrChannel + srcLoc[1]) * bilinearCoeffs[1]) +                           // TopRow R02 Pixel * coeff1
                    (*(srcPtrChannel + srcLoc[2]) * bilinearCoeffs[2]) +                           // BottomRow R01 Pixel * coeff2
                    (*(srcPtrChannel + srcLoc[3]) * bilinearCoeffs[3])));                          // BottomRow R02 Pixel * coeff3
        else if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
          dst[c] = (T)(((*(srcPtrChannel + srcLoc[0]) * bilinearCoeffs[0]) +                       // TopRow R01 Pixel * coeff0
                    (*(srcPtrChannel + srcLoc[1]) * bilinearCoeffs[1]) +                           // TopRow R02 Pixel * coeff1
                    (*(srcPtrChannel + srcLoc[2]) * bilinearCoeffs[2]) +                           // BottomRow R01 Pixel * coeff2
                    (*(srcPtrChannel + srcLoc[3]) * bilinearCoeffs[3])));                          // BottomRow R02 Pixel * coeff3

        srcPtrChannel += srcDescPtr->strides.cStride;
    }
}

inline void compute_generic_bilinear_srclocs_1c_avx(__m256 &pSrcY, __m256 &pSrcX, RpptBilinearNbhoodLocsVecLen8 &srcLocs, __m256 *pBilinearCoeffs, __m256 &pSrcStrideH, __m256i *pxSrcStridesCHW, __m256 *pRoiLTRB)
{
    __m256 pWeightParams[4], pSrcBilinearLTyx[4];
    pSrcBilinearLTyx[0] = _mm256_floor_ps(pSrcY);                               // srcLT->y = (Rpp32s) srcY;
    pSrcBilinearLTyx[1] = _mm256_floor_ps(pSrcX);                               // srcLT->x = (Rpp32s) srcX;
    pWeightParams[0] = _mm256_sub_ps(pSrcY, pSrcBilinearLTyx[0]);               // weightParams[0] = srcY - srcLT->y;
    pWeightParams[1] = _mm256_sub_ps(avx_p1, pWeightParams[0]);                 // weightParams[1] = 1 - weightParams[0];
    pWeightParams[2] = _mm256_sub_ps(pSrcX, pSrcBilinearLTyx[1]);               // weightParams[2] = srcX - srcLT->x;
    pWeightParams[3] = _mm256_sub_ps(avx_p1, pWeightParams[2]);                 // weightParams[3] = 1 - weightParams[2]
    pBilinearCoeffs[0] = _mm256_mul_ps(pWeightParams[1], pWeightParams[3]);     // (1 - weightedHeight) * (1 - weightedWidth)
    pBilinearCoeffs[1] = _mm256_mul_ps(pWeightParams[1], pWeightParams[2]);     // (1 - weightedHeight) * weightedWidth
    pBilinearCoeffs[2] = _mm256_mul_ps(pWeightParams[0], pWeightParams[3]);     // weightedHeight * (1 - weightedWidth)
    pBilinearCoeffs[3] = _mm256_mul_ps(pWeightParams[0], pWeightParams[2]);     // weightedHeight * weightedWidth
    pSrcBilinearLTyx[0] = _mm256_min_ps(_mm256_max_ps(pSrcBilinearLTyx[0], pRoiLTRB[1]), _mm256_sub_ps(pRoiLTRB[3], avx_p1));
    pSrcBilinearLTyx[1] = _mm256_min_ps(_mm256_max_ps(pSrcBilinearLTyx[1], pRoiLTRB[0]), _mm256_sub_ps(pRoiLTRB[2], avx_p1));
    pSrcBilinearLTyx[2] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pSrcBilinearLTyx[0], avx_p1), pRoiLTRB[1]), _mm256_sub_ps(pRoiLTRB[3], avx_p1));
    pSrcBilinearLTyx[3] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pSrcBilinearLTyx[1], avx_p1), pRoiLTRB[0]), _mm256_sub_ps(pRoiLTRB[2], avx_p1));
    __m256i pxSrcLocsTL =  _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[0], pSrcStrideH, pSrcBilinearLTyx[1]));     // 8 Top-Left memory locations = 8 Top-Left srcYs * hStride + 8 Top-Left srcXs
    __m256i pxSrcLocsTR =  _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[0], pSrcStrideH, pSrcBilinearLTyx[3]));     // 8 Top-Right memory locations = 8 Top-Left srcYs * hStride + 8 Bottom-right srcXs
    __m256i pxSrcLocsBL = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[2], pSrcStrideH, pSrcBilinearLTyx[1]));      // 8 Bottom-Left memory locations = 8 Bottom-right srcYs * hStride + 8 Top-Left srcXs
    __m256i pxSrcLocsBR = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[2], pSrcStrideH, pSrcBilinearLTyx[3]));      // 8 Bottom-Right memory locations = 8 Bottom-right srcYs * hStride + 8 Bottom-right srcXs
    _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsTL.data[0], pxSrcLocsTL);    // Store precomputed bilinear Top-Left locations
    _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsTR.data[0], pxSrcLocsTR);    // Store precomputed bilinear Top-Right locations
    _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsBL.data[0], pxSrcLocsBL);    // Store precomputed bilinear Bottom-Left locations
    _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsBR.data[0], pxSrcLocsBR);    // Store precomputed bilinear Bottom-Right locations
}

inline void compute_generic_bilinear_srclocs_3c_avx(__m256 &pSrcY, __m256 &pSrcX, RpptBilinearNbhoodLocsVecLen8 &srcLocs, __m256 *pBilinearCoeffs, __m256 &pSrcStrideH, __m256i *pxSrcStridesCHW, Rpp32s srcChannels, __m256 *pRoiLTRB, bool isSrcPKD3 = false)
{
    __m256 pWeightParams[4], pSrcBilinearLTyx[4];
    pSrcBilinearLTyx[0] = _mm256_floor_ps(pSrcY);                               // srcLT->y = (Rpp32s) srcY;
    pSrcBilinearLTyx[1] = _mm256_floor_ps(pSrcX);                               // srcLT->x = (Rpp32s) srcX;
    pWeightParams[0] = _mm256_sub_ps(pSrcY, pSrcBilinearLTyx[0]);               // weightParams[0] = srcY - srcLT->y;
    pWeightParams[1] = _mm256_sub_ps(avx_p1, pWeightParams[0]);                 // weightParams[1] = 1 - weightParams[0];
    pWeightParams[2] = _mm256_sub_ps(pSrcX, pSrcBilinearLTyx[1]);               // weightParams[2] = srcX - srcLT->x;
    pWeightParams[3] = _mm256_sub_ps(avx_p1, pWeightParams[2]);                 // weightParams[3] = 1 - weightParams[2]
    pBilinearCoeffs[0] = _mm256_mul_ps(pWeightParams[1], pWeightParams[3]);     // (1 - weightedHeight) * (1 - weightedWidth)
    pBilinearCoeffs[1] = _mm256_mul_ps(pWeightParams[1], pWeightParams[2]);     // (1 - weightedHeight) * weightedWidth
    pBilinearCoeffs[2] = _mm256_mul_ps(pWeightParams[0], pWeightParams[3]);     // weightedHeight * (1 - weightedWidth)
    pBilinearCoeffs[3] = _mm256_mul_ps(pWeightParams[0], pWeightParams[2]);     // weightedHeight * weightedWidth
    pSrcBilinearLTyx[0] = _mm256_min_ps(_mm256_max_ps(pSrcBilinearLTyx[0], pRoiLTRB[1]), _mm256_sub_ps(pRoiLTRB[3], avx_p1));
    pSrcBilinearLTyx[1] = _mm256_min_ps(_mm256_max_ps(pSrcBilinearLTyx[1], pRoiLTRB[0]), _mm256_sub_ps(pRoiLTRB[2], avx_p1));
    pSrcBilinearLTyx[2] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pSrcBilinearLTyx[0], avx_p1), pRoiLTRB[1]), _mm256_sub_ps(pRoiLTRB[3], avx_p1));
    pSrcBilinearLTyx[3] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pSrcBilinearLTyx[1], avx_p1), pRoiLTRB[0]), _mm256_sub_ps(pRoiLTRB[2], avx_p1));
    if(isSrcPKD3)
    {
        pSrcBilinearLTyx[1] = _mm256_mul_ps(pSrcBilinearLTyx[1], avx_p3);       // if pkd3, multiply Left-Top column location by 3
        pSrcBilinearLTyx[3] = _mm256_mul_ps(pSrcBilinearLTyx[3], avx_p3);       // if pkd3, multiply Right-Top column location by 3
    }
    __m256i pxSrcLocsTL =  _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[0], pSrcStrideH, pSrcBilinearLTyx[1]));    // 8 Top-Left memory locations = 8 Top-Left srcYs * hStride + 8 Top-Left srcXs
    __m256i pxSrcLocsTR =  _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[0], pSrcStrideH, pSrcBilinearLTyx[3]));    // 8 Top-Right memory locations = 8 Top-Left srcYs * hStride + 8 Bottom-right srcXs
    __m256i pxSrcLocsBL = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[2], pSrcStrideH, pSrcBilinearLTyx[1]));     // 8 Bottom-Left memory locations = 8 Bottom-right srcYs * hStride + 8 Top-Left srcXs
    __m256i pxSrcLocsBR = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcBilinearLTyx[2], pSrcStrideH, pSrcBilinearLTyx[3]));     // 8 Bottom-Right memory locations = 8 Bottom-right srcYs * hStride + 8 Bottom-right srcXs
    for (int c = 0; c < srcChannels * 8; c += 8)
    {
        _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsTL.data[c], pxSrcLocsTL);    // Store precomputed bilinear Top-Left locations
        _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsTR.data[c], pxSrcLocsTR);    // Store precomputed bilinear Top-Right locations
        _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsBL.data[c], pxSrcLocsBL);    // Store precomputed bilinear Bottom-Left locations
        _mm256_storeu_si256((__m256i*) &srcLocs.srcLocsBR.data[c], pxSrcLocsBR);    // Store precomputed bilinear Bottom-Right locations
        pxSrcLocsTL = _mm256_add_epi32(pxSrcLocsTL, pxSrcStridesCHW[0]);            // Increment Top-Left locations by cStride
        pxSrcLocsTR = _mm256_add_epi32(pxSrcLocsTR, pxSrcStridesCHW[0]);            // Increment Top-Right locations by cStride
        pxSrcLocsBL = _mm256_add_epi32(pxSrcLocsBL, pxSrcStridesCHW[0]);            // Increment Bottom-Left locations by cStride
        pxSrcLocsBR = _mm256_add_epi32(pxSrcLocsBR, pxSrcStridesCHW[0]);            // Increment Bottom-Right locations by cStride
    }
}

template <typename T>
inline void compute_generic_bilinear_interpolation_pkd3_to_pln3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTempR, T *dstPtrTempG, T *dstPtrTempB, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    Rpp32s srcXFloor = std::floor(srcX);
    Rpp32s srcYFloor = std::floor(srcY);
    if ((srcXFloor < roiLTRB->ltrbROI.lt.x) || (srcYFloor < roiLTRB->ltrbROI.lt.y) || (srcXFloor > roiLTRB->ltrbROI.rb.x) || (srcYFloor > roiLTRB->ltrbROI.rb.y))
    {
        *dstPtrTempR = 0;
        *dstPtrTempG = 0;
        *dstPtrTempB = 0;
    }
    else
    {
        T dst[3];
        compute_generic_bilinear_srclocs_and_interpolate(srcPtrChannel, srcDescPtr, srcY, srcX, roiLTRB, dst);
        *dstPtrTempR = dst[0];
        *dstPtrTempG = dst[1];
        *dstPtrTempB = dst[2];
    }
}

template <typename T>
inline void compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    Rpp32s srcXFloor = std::floor(srcX);
    Rpp32s srcYFloor = std::floor(srcY);
    if ((srcXFloor < roiLTRB->ltrbROI.lt.x) || (srcYFloor < roiLTRB->ltrbROI.lt.y) || (srcXFloor > roiLTRB->ltrbROI.rb.x) || (srcYFloor > roiLTRB->ltrbROI.rb.y))
    {
        memset(dstPtrTemp, 0, 3 * sizeof(T));
    }
    else
    {
        compute_generic_bilinear_srclocs_and_interpolate(srcPtrChannel, srcDescPtr, srcY, srcX, roiLTRB, dstPtrTemp);
    }
}

template <typename T>
inline void compute_generic_bilinear_interpolation_pln_to_pln(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    Rpp32s srcXFloor = std::floor(srcX);
    Rpp32s srcYFloor = std::floor(srcY);
    if ((srcXFloor < roiLTRB->ltrbROI.lt.x) || (srcYFloor < roiLTRB->ltrbROI.lt.y) || (srcXFloor > roiLTRB->ltrbROI.rb.x) || (srcYFloor > roiLTRB->ltrbROI.rb.y))
    {
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            *dstPtrTemp = 0;
            dstPtrTemp += dstDescPtr->strides.cStride;
        }
    }
    else
    {
        T dst[3];
        compute_generic_bilinear_srclocs_and_interpolate(srcPtrChannel, srcDescPtr, srcY, srcX, roiLTRB, dst);
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            *dstPtrTemp = dst[c];
            dstPtrTemp += dstDescPtr->strides.cStride;
        }
    }
}

inline void compute_generic_nn_srclocs_and_validate_sse(__m128 pSrcY, __m128 pSrcX, __m128 *pRoiLTRB, __m128 pSrcStrideH, Rpp32s *srcLoc, Rpp32s *invalidLoad, bool hasRGBChannels = false)
{
    pSrcY = _mm_round_ps(pSrcY, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));        // Nearest Neighbor Y location vector
    pSrcX = _mm_round_ps(pSrcX, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));        // Nearest Neighbor X location vector
    _mm_storeu_si128((__m128i*) invalidLoad, _mm_cvtps_epi32(_mm_or_ps(                 // Vectorized ROI boundary check
        _mm_or_ps(_mm_cmplt_ps(pSrcX, pRoiLTRB[0]), _mm_cmplt_ps(pSrcY, pRoiLTRB[1])),
        _mm_or_ps(_mm_cmpgt_ps(pSrcX, pRoiLTRB[2]), _mm_cmpgt_ps(pSrcY, pRoiLTRB[3]))
    )));
    if (hasRGBChannels)
        pSrcX = _mm_mul_ps(pSrcX, xmm_p3);
    __m128i pxSrcLoc = _mm_cvtps_epi32(_mm_fmadd_ps(pSrcY, pSrcStrideH, pSrcX));
    _mm_storeu_si128((__m128i*) srcLoc, pxSrcLoc);
}

inline void compute_generic_nn_srclocs_and_validate_avx(__m256 pSrcY, __m256 pSrcX, __m256 *pRoiLTRB, __m256 pSrcStrideH, Rpp32s *srcLoc, Rpp32s *invalidLoad, bool hasRGBChannels = false)
{
    pSrcY = _mm256_round_ps(pSrcY, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));              // Nearest Neighbor Y location vector
    pSrcX = _mm256_round_ps(pSrcX, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));              // Nearest Neighbor X location vector
    _mm256_storeu_si256((__m256i*) invalidLoad, _mm256_cvtps_epi32(_mm256_or_ps(                 // Vectorized ROI boundary check
        _mm256_or_ps(_mm256_cmp_ps(pSrcX, pRoiLTRB[0], _CMP_LT_OQ), _mm256_cmp_ps(pSrcY, pRoiLTRB[1],_CMP_LT_OQ)),
        _mm256_or_ps(_mm256_cmp_ps(pSrcX, pRoiLTRB[2], _CMP_GT_OQ), _mm256_cmp_ps(pSrcY, pRoiLTRB[3], _CMP_GT_OQ))
    )));
    if (hasRGBChannels)
        pSrcX = _mm256_mul_ps(pSrcX, avx_p3);
    __m256i pxSrcLoc = _mm256_cvtps_epi32(_mm256_fmadd_ps(pSrcY, pSrcStrideH, pSrcX));
    _mm256_storeu_si256((__m256i*) srcLoc, pxSrcLoc);
}

template <typename T>
inline void compute_generic_nn_interpolation_pkd3_to_pln3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTempR, T *dstPtrTempG, T *dstPtrTempB, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    srcY = std::round(srcY);    // Nearest Neighbor Y location
    srcX = std::round(srcX);    // Nearest Neighbor X location
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        *dstPtrTempR = 0;
        *dstPtrTempG = 0;
        *dstPtrTempB = 0;
    }
    else
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
        *dstPtrTempR = *srcPtrTemp++;
        *dstPtrTempG = *srcPtrTemp++;
        *dstPtrTempB = *srcPtrTemp;
    }
}

template <typename T>
inline void compute_generic_nn_interpolation_pkd3_to_pkd3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    srcY = std::round(srcY);    // Nearest Neighbor Y location
    srcX = std::round(srcX);    // Nearest Neighbor X location
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        memset(dstPtrTemp, 0, 3 * sizeof(T));
    }
    else
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
        memcpy(dstPtrTemp, srcPtrTemp, 3 * sizeof(T));
    }
}

template <typename T>
inline void compute_generic_nn_interpolation_pln3_to_pkd3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    srcY = std::round(srcY);    // Nearest Neighbor Y location
    srcX = std::round(srcX);    // Nearest Neighbor X location
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        memset(dstPtrTemp, 0, 3 * sizeof(T));
    }
    else
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
        *dstPtrTemp++ = *srcPtrTemp;
        srcPtrTemp += srcDescPtr->strides.cStride;
        *dstPtrTemp++ = *srcPtrTemp;
        srcPtrTemp += srcDescPtr->strides.cStride;
        *dstPtrTemp = *srcPtrTemp;
    }
}

template <typename T>
inline void compute_generic_nn_interpolation_pln_to_pln(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTemp, T *srcPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    srcY = std::round(srcY);    // Nearest Neighbor Y location
    srcX = std::round(srcX);    // Nearest Neighbor X location
    if ((srcX < roiLTRB->ltrbROI.lt.x) || (srcY < roiLTRB->ltrbROI.lt.y) || (srcX > roiLTRB->ltrbROI.rb.x) || (srcY > roiLTRB->ltrbROI.rb.y))
    {
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            *dstPtrTemp = 0;
            dstPtrTemp += dstDescPtr->strides.cStride;
        }
    }
    else
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            *dstPtrTemp = *srcPtrTemp;
            srcPtrTemp += srcDescPtr->strides.cStride;
            dstPtrTemp += dstDescPtr->strides.cStride;
        }
    }
}

inline void compute_bilinear_coefficients(Rpp32f *weightParams, Rpp32f *bilinearCoeffs)
{
    bilinearCoeffs[0] = weightParams[1] * weightParams[3];    // (1 - weightedHeight) * (1 - weightedWidth)
    bilinearCoeffs[1] = weightParams[1] * weightParams[2];    // (1 - weightedHeight) * weightedWidth
    bilinearCoeffs[2] = weightParams[0] * weightParams[3];    // weightedHeight * (1 - weightedWidth)
    bilinearCoeffs[3] = weightParams[0] * weightParams[2];    // weightedHeight * weightedWidth
}

inline void compute_bilinear_coefficients_avx(__m256 *pWeightParams, __m256 *pBilinearCoeffs)
{
    pBilinearCoeffs[0] = _mm256_mul_ps(pWeightParams[1], pWeightParams[3]);    // (1 - weightedHeight) * (1 - weightedWidth)
    pBilinearCoeffs[1] = _mm256_mul_ps(pWeightParams[1], pWeightParams[2]);    // (1 - weightedHeight) * weightedWidth
    pBilinearCoeffs[2] = _mm256_mul_ps(pWeightParams[0], pWeightParams[3]);    // weightedHeight * (1 - weightedWidth)
    pBilinearCoeffs[3] = _mm256_mul_ps(pWeightParams[0], pWeightParams[2]);    // weightedHeight * weightedWidth
}

template <typename T, typename U>
inline void compute_bilinear_interpolation_1c(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32s limit, Rpp32f *bilinearCoeffs, U *dstPtr)
{
    Rpp32s loc1 = std::min(std::max(loc, 0), limit);
    Rpp32s loc2 = std::min(std::max(loc + 1, 0), limit);
    *dstPtr = (U)(((*(srcRowPtrsForInterp[0] + loc1)) * bilinearCoeffs[0]) +     // TopRow 1st Pixel * coeff0
                  ((*(srcRowPtrsForInterp[0] + loc2)) * bilinearCoeffs[1]) +     // TopRow 2nd Pixel * coeff1
                  ((*(srcRowPtrsForInterp[1] + loc1)) * bilinearCoeffs[2]) +     // BottomRow 1st Pixel * coeff2
                  ((*(srcRowPtrsForInterp[1] + loc2)) * bilinearCoeffs[3]));    // BottomRow 2nd Pixel * coeff3
}

template <typename T, typename U>
inline void compute_bilinear_interpolation_3c_pkd(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32s limit, Rpp32f *bilinearCoeffs, U *dstPtrR, U *dstPtrG, U *dstPtrB)
{
    Rpp32s loc1 = std::min(std::max(loc, 0), limit);
    Rpp32s loc2 = std::min(std::max(loc + 3, 0), limit);
    *dstPtrR = (U)(((*(srcRowPtrsForInterp[0] + loc1)) * bilinearCoeffs[0]) +        // TopRow R01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc2)) * bilinearCoeffs[1]) +        // TopRow R02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc1)) * bilinearCoeffs[2]) +        // BottomRow R01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc2)) * bilinearCoeffs[3]));       // BottomRow R02 Pixel * coeff3
    *dstPtrG = (U)(((*(srcRowPtrsForInterp[0] + loc1 + 1)) * bilinearCoeffs[0]) +    // TopRow G01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc2 + 1)) * bilinearCoeffs[1]) +    // TopRow G02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc1 + 1)) * bilinearCoeffs[2]) +    // BottomRow G01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc2 + 1)) * bilinearCoeffs[3]));   // BottomRow G02 Pixel * coeff3
    *dstPtrB = (U)(((*(srcRowPtrsForInterp[0] + loc1 + 2)) * bilinearCoeffs[0]) +    // TopRow B01 Pixel * coeff0
                   ((*(srcRowPtrsForInterp[0] + loc2 + 2)) * bilinearCoeffs[1]) +    // TopRow B02 Pixel * coeff1
                   ((*(srcRowPtrsForInterp[1] + loc1 + 2)) * bilinearCoeffs[2]) +    // BottomRow B01 Pixel * coeff2
                   ((*(srcRowPtrsForInterp[1] + loc2 + 2)) * bilinearCoeffs[3]));   // BottomRow B02 Pixel * coeff3
}

template <typename T, typename U>
inline void compute_bilinear_interpolation_3c_pln(T **srcRowPtrsForInterp, Rpp32s loc, Rpp32s limit, Rpp32f *bilinearCoeffs, U *dstPtrR, U *dstPtrG, U *dstPtrB)
{
    compute_bilinear_interpolation_1c(srcRowPtrsForInterp, loc, limit, bilinearCoeffs, dstPtrR);
    compute_bilinear_interpolation_1c(srcRowPtrsForInterp + 2, loc, limit, bilinearCoeffs, dstPtrG);
    compute_bilinear_interpolation_1c(srcRowPtrsForInterp + 4, loc, limit, bilinearCoeffs, dstPtrB);
}

inline void compute_bilinear_interpolation_1c_avx(__m256 *pSrcPixels, __m256 *pBilinearCoeffs, __m256 &pDstPixels)
{
    pDstPixels = _mm256_fmadd_ps(pSrcPixels[3], pBilinearCoeffs[3], _mm256_fmadd_ps(pSrcPixels[2], pBilinearCoeffs[2],
                 _mm256_fmadd_ps(pSrcPixels[1], pBilinearCoeffs[1], _mm256_mul_ps(pSrcPixels[0], pBilinearCoeffs[0]))));
}

inline void compute_bilinear_interpolation_3c_avx(__m256 *pSrcPixels, __m256 *pBilinearCoeffs, __m256 *pDstPixels)
{
    compute_bilinear_interpolation_1c_avx(pSrcPixels, pBilinearCoeffs, pDstPixels[0]);
    compute_bilinear_interpolation_1c_avx(pSrcPixels + 4, pBilinearCoeffs, pDstPixels[1]);
    compute_bilinear_interpolation_1c_avx(pSrcPixels + 8, pBilinearCoeffs, pDstPixels[2]);
}

template <typename T>
inline void compute_src_row_ptrs_for_bilinear_interpolation(T **rowPtrsForInterp, T *srcPtr, Rpp32s loc, Rpp32s limit, RpptDescPtr descPtr)
{
    rowPtrsForInterp[0] = srcPtr + std::min(std::max(loc, 0), limit) * descPtr->strides.hStride;          // TopRow for bilinear interpolation
    rowPtrsForInterp[1]  = srcPtr + std::min(std::max(loc + 1, 0), limit) * descPtr->strides.hStride;     // BottomRow for bilinear interpolation
}

template <typename T>
inline void compute_src_row_ptrs_for_bilinear_interpolation_pln(T **rowPtrsForInterp, T *srcPtr, Rpp32s loc, Rpp32s limit, RpptDescPtr descPtr)
{
    rowPtrsForInterp[0] = srcPtr + std::min(std::max(loc, 0), limit) * descPtr->strides.hStride;          // TopRow for bilinear interpolation (R channel)
    rowPtrsForInterp[1] = srcPtr + std::min(std::max(loc + 1, 0), limit) * descPtr->strides.hStride;      // BottomRow for bilinear interpolation (R channel)
    rowPtrsForInterp[2] = rowPtrsForInterp[0] + descPtr->strides.cStride;   // TopRow for bilinear interpolation (G channel)
    rowPtrsForInterp[3] = rowPtrsForInterp[1] + descPtr->strides.cStride;   // BottomRow for bilinear interpolation (G channel)
    rowPtrsForInterp[4] = rowPtrsForInterp[2] + descPtr->strides.cStride;   // TopRow for bilinear interpolation (B channel)
    rowPtrsForInterp[5] = rowPtrsForInterp[3] + descPtr->strides.cStride;   // BottomRow for bilinear interpolation (B channel)
}

#endif // RPP_CPU_INTERPOLATION_HPP
