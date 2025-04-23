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

#include "host_tensor_executors.hpp"

inline void compute_gridmask_masks_16_host(__m128 *pCol, __m128 *pGridRowRatio, __m128 pCosRatio, __m128 pSinRatio, __m128 pGridRatio, __m128 *pMask)
{
    __m128 pCalc[2];

    pCalc[0] = _mm_fmadd_ps(pCol[0], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[0], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[0] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[1], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[1], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[1] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[2], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[2], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[2] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCalc[0] = _mm_fmadd_ps(pCol[3], pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol[3], pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask[3] = _mm_or_ps(pCalc[0], pCalc[1]);

    pCol[0] = _mm_add_ps(pCol[0], xmm_p16);
    pCol[1] = _mm_add_ps(pCol[1], xmm_p16);
    pCol[2] = _mm_add_ps(pCol[2], xmm_p16);
    pCol[3] = _mm_add_ps(pCol[3], xmm_p16);
}

inline void compute_gridmask_masks_4_host(__m128 &pCol, __m128 *pGridRowRatio, __m128 pCosRatio, __m128 pSinRatio, __m128 pGridRatio, __m128 &pMask)
{
    __m128 pCalc[2];

    pCalc[0] = _mm_fmadd_ps(pCol, pCosRatio, pGridRowRatio[0]);
    pCalc[1] = _mm_fmadd_ps(pCol, pSinRatio, pGridRowRatio[1]);
    pCalc[0] = _mm_cmpge_ps(_mm_sub_ps(pCalc[0], _mm_floor_ps(pCalc[0])), pGridRatio);
    pCalc[1] = _mm_cmpge_ps(_mm_sub_ps(pCalc[1], _mm_floor_ps(pCalc[1])), pGridRatio);
    pMask = _mm_or_ps(pCalc[0], pCalc[1]);
    pCol = _mm_add_ps(pCol, xmm_p4);
}

inline void compute_gridmask_result_48_host(__m128 *p, __m128 *pMask)
{
    p[0] = _mm_and_ps(p[0], pMask[0]);
    p[1] = _mm_and_ps(p[1], pMask[1]);
    p[2] = _mm_and_ps(p[2], pMask[2]);
    p[3] = _mm_and_ps(p[3], pMask[3]);
    p[4] = _mm_and_ps(p[4], pMask[0]);
    p[5] = _mm_and_ps(p[5], pMask[1]);
    p[6] = _mm_and_ps(p[6], pMask[2]);
    p[7] = _mm_and_ps(p[7], pMask[3]);
    p[8] = _mm_and_ps(p[8], pMask[0]);
    p[9] = _mm_and_ps(p[9], pMask[1]);
    p[10] = _mm_and_ps(p[10], pMask[2]);
    p[11] = _mm_and_ps(p[11], pMask[3]);
}

inline void compute_gridmask_result_16_host(__m128 *p, __m128 *pMask)
{
    p[0] = _mm_and_ps(p[0], pMask[0]);
    p[1] = _mm_and_ps(p[1], pMask[1]);
    p[2] = _mm_and_ps(p[2], pMask[2]);
    p[3] = _mm_and_ps(p[3], pMask[3]);
}

inline void compute_gridmask_result_12_host(__m128 *p, __m128 pMask)
{
    p[0] = _mm_and_ps(p[0], pMask);
    p[1] = _mm_and_ps(p[1], pMask);
    p[2] = _mm_and_ps(p[2], pMask);
}

inline void compute_gridmask_result_4_host(__m128 *p, __m128 pMask)
{
    p[0] = _mm_and_ps(p[0], pMask);
}

RppStatus gridmask_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u tileWidth,
                                     Rpp32f gridRatio,
                                     Rpp32f gridAngle,
                                     RpptUintVector2D translateVector,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f cosRatio, sinRatio, tileWidthInv;
        RpptFloatVector2D translateVectorRatio;
        tileWidthInv = 1.0f / (Rpp32f)tileWidth;
        cosRatio = cos(gridAngle) * tileWidthInv;
        sinRatio = sin(gridAngle) * tileWidthInv;
        translateVectorRatio.x = translateVector.x * tileWidthInv;
        translateVectorRatio.y = translateVector.y * tileWidthInv;

        __m128 pCosRatio, pSinRatio, pGridRatio, pColInit[4];
        pCosRatio = _mm_set1_ps(cosRatio);
        pSinRatio = _mm_set1_ps(sinRatio);
        pGridRatio = _mm_set1_ps(gridRatio);
        pColInit[0] = _mm_setr_ps(0, 1, 2, 3);
        pColInit[1] = _mm_setr_ps(4, 5, 6, 7);
        pColInit[2] = _mm_setr_ps(8, 9, 10, 11);
        pColInit[3] = _mm_setr_ps(12, 13, 14, 15);

        // Gridmask with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[12];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_gridmask_result_48_host(p, pMask);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *srcPtrTemp * m;
                        srcPtrTemp += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    dstPtrTempR++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB,  *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[12];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_gridmask_result_48_host(p, pMask);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp8u *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = *srcPtrTempChn * m;
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTemp += dstDescPtr->strides.cStride;
                    }

                    srcPtrTempR++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[12];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_gridmask_result_48_host(p, pMask);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp * m;
                        srcPtrTemp += srcDescPtr->strides.cStride;
                        dstPtrTemp += dstDescPtr->strides.cStride;
                    }
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[12];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_gridmask_result_48_host(p, pMask);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp8u *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrTempR;
                    dstPtrTempChn = dstPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *srcPtrTempChn * m;
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    srcPtrTempR++;
                    dstPtrTempR++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask for single channel images (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[4];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);    // simd loads
                    compute_gridmask_result_16_host(p, pMask);
                    rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 16;
                    dstPtrTemp += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);
                    *dstPtrTemp = *srcPtrTemp * m;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gridmask_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u tileWidth,
                                       Rpp32f gridRatio,
                                       Rpp32f gridAngle,
                                       RpptUintVector2D translateVector,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f cosRatio, sinRatio, tileWidthInv;
        RpptFloatVector2D translateVectorRatio;
        tileWidthInv = 1.0f / (Rpp32f)tileWidth;
        cosRatio = cos(gridAngle) * tileWidthInv;
        sinRatio = sin(gridAngle) * tileWidthInv;
        translateVectorRatio.x = translateVector.x * tileWidthInv;
        translateVectorRatio.y = translateVector.y * tileWidthInv;

        __m128 pCosRatio, pSinRatio, pGridRatio, pColInit;
        pCosRatio = _mm_set1_ps(cosRatio);
        pSinRatio = _mm_set1_ps(sinRatio);
        pGridRatio = _mm_set1_ps(gridRatio);
        pColInit = _mm_setr_ps(0, 1, 2, 3);

        // Gridmask with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p[4];
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_gridmask_result_12_host(p, pMask);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *srcPtrTemp * m;
                        srcPtrTemp += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    dstPtrTempR++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB,  *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p[4];
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_gridmask_result_12_host(p, pMask);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp32f *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = *srcPtrTempChn * m;
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTemp += dstDescPtr->strides.cStride;
                    }

                    srcPtrTempR++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p[4];
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_gridmask_result_12_host(p, pMask);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp * m;
                        srcPtrTemp += srcDescPtr->strides.cStride;
                        dstPtrTemp += dstDescPtr->strides.cStride;
                    }
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p[4];
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_gridmask_result_12_host(p, pMask);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp32f *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrTempR;
                    dstPtrTempChn = dstPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *srcPtrTempChn * m;
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    srcPtrTempR++;
                    dstPtrTempR++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask for single channel images (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p;
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, &p);    // simd loads
                    compute_gridmask_result_4_host(&p, pMask);
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &p);    // simd stores

                    srcPtrTemp += 4;
                    dstPtrTemp += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);
                    *dstPtrTemp = *srcPtrTemp * m;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gridmask_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u tileWidth,
                                       Rpp32f gridRatio,
                                       Rpp32f gridAngle,
                                       RpptUintVector2D translateVector,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f cosRatio, sinRatio, tileWidthInv;
        RpptFloatVector2D translateVectorRatio;
        tileWidthInv = 1.0f / (Rpp32f)tileWidth;
        cosRatio = cos(gridAngle) * tileWidthInv;
        sinRatio = sin(gridAngle) * tileWidthInv;
        translateVectorRatio.x = translateVector.x * tileWidthInv;
        translateVectorRatio.y = translateVector.y * tileWidthInv;

        __m128 pCosRatio, pSinRatio, pGridRatio, pColInit;
        pCosRatio = _mm_set1_ps(cosRatio);
        pSinRatio = _mm_set1_ps(sinRatio);
        pGridRatio = _mm_set1_ps(gridRatio);
        pColInit = _mm_setr_ps(0, 1, 2, 3);

        // Gridmask with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p[4];
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_gridmask_result_12_host(p, pMask);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *srcPtrTemp * m;
                        srcPtrTemp += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    dstPtrTempR++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB,  *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p[4];
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    compute_gridmask_result_12_host(p, pMask);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp16f *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = *srcPtrTempChn * m;
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTemp += dstDescPtr->strides.cStride;
                    }

                    srcPtrTempR++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p[4];
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_gridmask_result_12_host(p, pMask);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp * m;
                        srcPtrTemp += srcDescPtr->strides.cStride;
                        dstPtrTemp += dstDescPtr->strides.cStride;
                    }
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p[4];
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    compute_gridmask_result_12_host(p, pMask);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp16f *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrTempR;
                    dstPtrTempChn = dstPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *srcPtrTempChn * m;
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    srcPtrTempR++;
                    dstPtrTempR++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask for single channel images (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~3;

            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol;
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol = pColInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pMask, p;
                    compute_gridmask_masks_4_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    Rpp32f srcPtrTemp_ps[4], dstPtrTemp_ps[4];
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }
                    rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, &p);    // simd loads
                    compute_gridmask_result_4_host(&p, pMask);
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, &p);    // simd stores
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTemp += 4;
                    dstPtrTemp += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);
                    *dstPtrTemp = *srcPtrTemp * m;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gridmask_i8_i8_host_tensor(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u tileWidth,
                                     Rpp32f gridRatio,
                                     Rpp32f gridAngle,
                                     RpptUintVector2D translateVector,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f cosRatio, sinRatio, tileWidthInv;
        RpptFloatVector2D translateVectorRatio;
        tileWidthInv = 1.0f / (Rpp32f)tileWidth;
        cosRatio = cos(gridAngle) * tileWidthInv;
        sinRatio = sin(gridAngle) * tileWidthInv;
        translateVectorRatio.x = translateVector.x * tileWidthInv;
        translateVectorRatio.y = translateVector.y * tileWidthInv;

        __m128 pCosRatio, pSinRatio, pGridRatio, pColInit[4];
        pCosRatio = _mm_set1_ps(cosRatio);
        pSinRatio = _mm_set1_ps(sinRatio);
        pGridRatio = _mm_set1_ps(gridRatio);
        pColInit[0] = _mm_setr_ps(0, 1, 2, 3);
        pColInit[1] = _mm_setr_ps(4, 5, 6, 7);
        pColInit[2] = _mm_setr_ps(8, 9, 10, 11);
        pColInit[3] = _mm_setr_ps(12, 13, 14, 15);

        // Gridmask with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[12];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_gridmask_result_48_host(p, pMask);
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *srcPtrTemp * m;
                        srcPtrTemp += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    dstPtrTempR++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB,  *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[12];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_gridmask_result_48_host(p, pMask);
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp8s *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = *srcPtrTempChn * m;
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTemp += dstDescPtr->strides.cStride;
                    }

                    srcPtrTempR++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[12];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_gridmask_result_48_host(p, pMask);
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp * m;
                        srcPtrTemp += srcDescPtr->strides.cStride;
                        dstPtrTemp += dstDescPtr->strides.cStride;
                    }
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[12];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_gridmask_result_48_host(p, pMask);
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);

                    Rpp8s *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrTempR;
                    dstPtrTempChn = dstPtrTempR;

                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *srcPtrTempChn * m;
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    srcPtrTempR++;
                    dstPtrTempR++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gridmask for single channel images (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                RpptFloatVector2D gridRowRatio;
                gridRowRatio.x = -translateVectorRatio.x + i * -sinRatio;
                gridRowRatio.y = -translateVectorRatio.y + i * cosRatio;

                __m128 pGridRowRatio[2], pCol[4];
                pGridRowRatio[0] = _mm_set1_ps(gridRowRatio.x);
                pGridRowRatio[1] = _mm_set1_ps(gridRowRatio.y);
                pCol[0] = pColInit[0];
                pCol[1] = pColInit[1];
                pCol[2] = pColInit[2];
                pCol[3] = pColInit[3];

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 pMask[4], p[4];
                    compute_gridmask_masks_16_host(pCol, pGridRowRatio, pCosRatio, pSinRatio, pGridRatio, pMask);

                    rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);    // simd loads
                    compute_gridmask_result_16_host(p, pMask);
                    rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 16;
                    dstPtrTemp += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatVector2D gridColRatio;
                    gridColRatio.x = gridRowRatio.x + vectorLoopCount * cosRatio;
                    gridColRatio.y = gridRowRatio.y + vectorLoopCount * sinRatio;
                    auto m = (gridColRatio.x - std::floor(gridColRatio.x) >= gridRatio) ||
                             (gridColRatio.y - std::floor(gridColRatio.y) >= gridRatio);
                    *dstPtrTemp = *srcPtrTemp * m;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
