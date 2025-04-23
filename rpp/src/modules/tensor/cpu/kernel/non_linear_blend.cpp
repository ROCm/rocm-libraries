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
#include "rpp_cpu_simd_math.hpp"

// -------------------- non_linear_blend host helpers --------------------

inline void compute_non_linear_blend_48_host(__m256 *p1, __m256 *p2, __m256 &pMultiplier, __m256 &pILocComponent, __m256 &pJLocComponent)
{
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_fmadd_ps(_mm256_mul_ps(pJLocComponent, pJLocComponent), pMultiplier, pILocComponent));
    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p1[0], p2[0]), pGaussianValue, p2[0]);    // non_linear_blend adjustment
    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p1[2], p2[2]), pGaussianValue, p2[2]);    // non_linear_blend adjustment
    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p1[4], p2[4]), pGaussianValue, p2[4]);    // non_linear_blend adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
    pGaussianValue = fast_exp_avx(_mm256_fmadd_ps(_mm256_mul_ps(pJLocComponent, pJLocComponent), pMultiplier, pILocComponent));
    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p1[1], p2[1]), pGaussianValue, p2[1]);    // non_linear_blend adjustment
    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p1[3], p2[3]), pGaussianValue, p2[3]);    // non_linear_blend adjustment
    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p1[5], p2[5]), pGaussianValue, p2[5]);    // non_linear_blend adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_non_linear_blend_24_host(__m256 *p1, __m256 *p2, __m256 &pMultiplier, __m256 &pILocComponent, __m256 &pJLocComponent)
{
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_fmadd_ps(_mm256_mul_ps(pJLocComponent, pJLocComponent), pMultiplier, pILocComponent));
    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p1[0], p2[0]), pGaussianValue, p2[0]);    // non_linear_blend adjustment
    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p1[1], p2[1]), pGaussianValue, p2[1]);    // non_linear_blend adjustment
    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p1[2], p2[2]), pGaussianValue, p2[2]);    // non_linear_blend adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_non_linear_blend_16_host(__m256 *p1, __m256 *p2, __m256 &pMultiplier, __m256 &pILocComponent, __m256 &pJLocComponent)
{
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_fmadd_ps(_mm256_mul_ps(pJLocComponent, pJLocComponent), pMultiplier, pILocComponent));
    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p1[0], p2[0]), pGaussianValue, p2[0]);    // non_linear_blend adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
    pGaussianValue = fast_exp_avx(_mm256_fmadd_ps(_mm256_mul_ps(pJLocComponent, pJLocComponent), pMultiplier, pILocComponent));
    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p1[1], p2[1]), pGaussianValue, p2[1]);    // non_linear_blend adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

inline void compute_non_linear_blend_8_host(__m256 *p1, __m256 *p2, __m256 &pMultiplier, __m256 &pILocComponent, __m256 &pJLocComponent)
{
    __m256 pGaussianValue;
    pGaussianValue = fast_exp_avx(_mm256_fmadd_ps(_mm256_mul_ps(pJLocComponent, pJLocComponent), pMultiplier, pILocComponent));
    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p1[0], p2[0]), pGaussianValue, p2[0]);    // non_linear_blend adjustment
    pJLocComponent = _mm256_add_ps(pJLocComponent, avx_p8);
}

// -------------------- non_linear_blend host executors --------------------

RppStatus non_linear_blend_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                             Rpp8u *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8u *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *stdDevTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle &handle)
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

        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32f multiplier = -0.5f / (stdDev * stdDev);
        Rpp32s halfHeight = (Rpp32s) (roi.xywhROI.roiHeight >> 1);
        Rpp32s halfWidth = (Rpp32s) (roi.xywhROI.roiWidth >> 1);

        Rpp8u *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~15;

        Rpp8u *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        __m256 pMultiplier = _mm256_set1_ps(multiplier);
        __m256 pHalfWidth = _mm256_set1_ps(halfWidth);

        // Non linear blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);                               // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr2Temp, p2);                               // simd loads
                    compute_non_linear_blend_48_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);   // simd stores
                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f)srcPtr1Temp[0] - (Rpp32f)srcPtr2Temp[0]) * gaussianValue) + (Rpp32f)srcPtr2Temp[0]);
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f)srcPtr1Temp[1] - (Rpp32f)srcPtr2Temp[1]) * gaussianValue) + (Rpp32f)srcPtr2Temp[1]);
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f)srcPtr1Temp[2] - (Rpp32f)srcPtr2Temp[2]) * gaussianValue) + (Rpp32f)srcPtr2Temp[2]);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);  // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);  // simd loads
                    compute_non_linear_blend_48_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p1);                              // simd stores
                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f)*srcPtr1TempR - (Rpp32f)*srcPtr2TempR) * gaussianValue) + (Rpp32f)*srcPtr2TempR);
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f)*srcPtr1TempG - (Rpp32f)*srcPtr2TempG) * gaussianValue) + (Rpp32f)*srcPtr2TempG);
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f)*srcPtr1TempB - (Rpp32f)*srcPtr2TempB) * gaussianValue) + (Rpp32f)*srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);                               // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr2Temp, p2);                               // simd loads
                    compute_non_linear_blend_48_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p1);                              // simd stores
                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f)srcPtr1Temp[0] - (Rpp32f)srcPtr2Temp[0]) * gaussianValue) + (Rpp32f)srcPtr2Temp[0]);
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f)srcPtr1Temp[1] - (Rpp32f)srcPtr2Temp[1]) * gaussianValue) + (Rpp32f)srcPtr2Temp[1]);
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f)srcPtr1Temp[2] - (Rpp32f)srcPtr2Temp[2]) * gaussianValue) + (Rpp32f)srcPtr2Temp[2]);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTemp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);  // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);  // simd loads
                    compute_non_linear_blend_48_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);   // simd stores
                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f)*srcPtr1TempR - (Rpp32f)*srcPtr2TempR) * gaussianValue) + (Rpp32f)*srcPtr2TempR);
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f)*srcPtr1TempG - (Rpp32f)*srcPtr2TempG) * gaussianValue) + (Rpp32f)*srcPtr2TempG);
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f)*srcPtr1TempB - (Rpp32f)*srcPtr2TempB) * gaussianValue) + (Rpp32f)*srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr1Temp, p1);  // simd loads
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr2Temp, p2);  // simd loads
                    compute_non_linear_blend_16_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p1);   // simd stores
                    srcPtr1Temp += 16;
                    srcPtr2Temp += 16;
                    dstPtrTemp += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);
                    *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f)*srcPtr1Temp - (Rpp32f)*srcPtr2Temp) * gaussianValue) + (Rpp32f)*srcPtr2Temp);
                    srcPtr1Temp++;
                    srcPtr2Temp++;
                    dstPtrTemp++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus non_linear_blend_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                               Rpp32f *srcPtr2,
                                               RpptDescPtr srcDescPtr,
                                               Rpp32f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *stdDevTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle &handle)
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

        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32f multiplier = -0.5f / (stdDev * stdDev);
        Rpp32s halfHeight = (Rpp32s) (roi.xywhROI.roiHeight >> 1);
        Rpp32s halfWidth = (Rpp32s) (roi.xywhROI.roiWidth >> 1);

        Rpp32f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;

        Rpp32f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        __m256 pMultiplier = _mm256_set1_ps(multiplier);
        __m256 pHalfWidth = _mm256_set1_ps(halfWidth);

        // Non linear blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p1[3], p2[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr1Temp, p1);                              // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr2Temp, p2);                              // simd loads
                    compute_non_linear_blend_24_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);  // simd stores
                    srcPtr1Temp += 24;
                    srcPtr2Temp += 24;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    *dstPtrTempR = RPPPIXELCHECKF32(((srcPtr1Temp[0] - srcPtr2Temp[0]) * gaussianValue) + srcPtr2Temp[0]);
                    *dstPtrTempG = RPPPIXELCHECKF32(((srcPtr1Temp[1] - srcPtr2Temp[1]) * gaussianValue) + srcPtr2Temp[1]);
                    *dstPtrTempB = RPPPIXELCHECKF32(((srcPtr1Temp[2] - srcPtr2Temp[2]) * gaussianValue) + srcPtr2Temp[2]);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p1[3], p2[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1); // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2); // simd loads
                    compute_non_linear_blend_24_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p1);                             // simd stores
                    srcPtr1TempR += 8;
                    srcPtr1TempG += 8;
                    srcPtr1TempB += 8;
                    srcPtr2TempR += 8;
                    srcPtr2TempG += 8;
                    srcPtr2TempB += 8;
                    dstPtrTemp += 24;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    dstPtrTemp[0] = RPPPIXELCHECKF32(((*srcPtr1TempR - *srcPtr2TempR) * gaussianValue) + *srcPtr2TempR);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(((*srcPtr1TempG - *srcPtr2TempG) * gaussianValue) + *srcPtr2TempG);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(((*srcPtr1TempB - *srcPtr2TempB) * gaussianValue) + *srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p1[3], p2[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr1Temp, p1);                      // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr2Temp, p2);                      // simd loads
                    compute_non_linear_blend_24_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);  // non_linear_blend adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p1);                     // simd stores
                    srcPtr1Temp += 24;
                    srcPtr2Temp += 24;
                    dstPtrTemp += 24;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    dstPtrTemp[0] = RPPPIXELCHECKF32(((srcPtr1Temp[0] - srcPtr2Temp[0]) * gaussianValue) + srcPtr2Temp[0]);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(((srcPtr1Temp[1] - srcPtr2Temp[1]) * gaussianValue) + srcPtr2Temp[1]);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(((srcPtr1Temp[2] - srcPtr2Temp[2]) * gaussianValue) + srcPtr2Temp[2]);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTemp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1); // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2); // simd loads
                    compute_non_linear_blend_24_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);  // simd stores
                    srcPtr1TempR += 8;
                    srcPtr1TempG += 8;
                    srcPtr1TempB += 8;
                    srcPtr2TempR += 8;
                    srcPtr2TempG += 8;
                    srcPtr2TempB += 8;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    *dstPtrTempR = RPPPIXELCHECKF32(((*srcPtr1TempR - *srcPtr2TempR) * gaussianValue) + *srcPtr2TempR);
                    *dstPtrTempG = RPPPIXELCHECKF32(((*srcPtr1TempG - *srcPtr2TempG) * gaussianValue) + *srcPtr2TempG);
                    *dstPtrTempB = RPPPIXELCHECKF32(((*srcPtr1TempB - *srcPtr2TempB) * gaussianValue) + *srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr1Temp, p1);                               // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr2Temp, p2);                               // simd loads
                    compute_non_linear_blend_8_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);   // non_linear_blend adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p1);                              // simd stores
                    srcPtr1Temp += 8;
                    srcPtr2Temp += 8;
                    dstPtrTemp += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);
                    *dstPtrTemp = RPPPIXELCHECKF32(((*srcPtr1Temp - *srcPtr2Temp) * gaussianValue) + *srcPtr2Temp);
                    srcPtr1Temp++;
                    srcPtr2Temp++;
                    dstPtrTemp++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus non_linear_blend_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                             Rpp8s *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8s *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *stdDevTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle &handle)
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

        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32f multiplier = -0.5f / (stdDev * stdDev);
        Rpp32s halfHeight = (Rpp32s) (roi.xywhROI.roiHeight >> 1);
        Rpp32s halfWidth = (Rpp32s) (roi.xywhROI.roiWidth >> 1);

        Rpp8s *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~15;

        Rpp8s *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        __m256 pMultiplier = _mm256_set1_ps(multiplier);
        __m256 pHalfWidth = _mm256_set1_ps(halfWidth);

        // Non linear blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);                               // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr2Temp, p2);                               // simd loads
                    compute_non_linear_blend_48_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);   // simd stores
                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)srcPtr1Temp[0] - (Rpp32f)srcPtr2Temp[0]) * gaussianValue) + (Rpp32f)srcPtr2Temp[0]);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)srcPtr1Temp[1] - (Rpp32f)srcPtr2Temp[1]) * gaussianValue) + (Rpp32f)srcPtr2Temp[1]);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)srcPtr1Temp[2] - (Rpp32f)srcPtr2Temp[2]) * gaussianValue) + (Rpp32f)srcPtr2Temp[2]);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);  // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);  // simd loads
                    compute_non_linear_blend_48_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p1);                              // simd stores
                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)*srcPtr1TempR - (Rpp32f)*srcPtr2TempR) * gaussianValue) + (Rpp32f)*srcPtr2TempR);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)*srcPtr1TempG - (Rpp32f)*srcPtr2TempG) * gaussianValue) + (Rpp32f)*srcPtr2TempG);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)*srcPtr1TempB - (Rpp32f)*srcPtr2TempB) * gaussianValue) + (Rpp32f)*srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);                               // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr2Temp, p2);                               // simd loads
                    compute_non_linear_blend_48_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p1);                              // simd stores
                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)srcPtr1Temp[0] - (Rpp32f)srcPtr2Temp[0]) * gaussianValue) + (Rpp32f)srcPtr2Temp[0]);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)srcPtr1Temp[1] - (Rpp32f)srcPtr2Temp[1]) * gaussianValue) + (Rpp32f)srcPtr2Temp[1]);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)srcPtr1Temp[2] - (Rpp32f)srcPtr2Temp[2]) * gaussianValue) + (Rpp32f)srcPtr2Temp[2]);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTemp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);  // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);  // simd loads
                    compute_non_linear_blend_48_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);   // simd stores
                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)*srcPtr1TempR - (Rpp32f)*srcPtr2TempR) * gaussianValue) + (Rpp32f)*srcPtr2TempR);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)*srcPtr1TempG - (Rpp32f)*srcPtr2TempG) * gaussianValue) + (Rpp32f)*srcPtr2TempG);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)*srcPtr1TempB - (Rpp32f)*srcPtr2TempB) * gaussianValue) + (Rpp32f)*srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr1Temp, p1);  // simd loads
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr2Temp, p2);  // simd loads
                    compute_non_linear_blend_16_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p1);   // simd stores
                    srcPtr1Temp += 16;
                    srcPtr2Temp += 16;
                    dstPtrTemp += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);
                    *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f)*srcPtr1Temp - (Rpp32f)*srcPtr2Temp) * gaussianValue) + (Rpp32f)*srcPtr2Temp);
                    srcPtr1Temp++;
                    srcPtr2Temp++;
                    dstPtrTemp++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus non_linear_blend_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                               Rpp16f *srcPtr2,
                                               RpptDescPtr srcDescPtr,
                                               Rpp16f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *stdDevTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle &handle)
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

        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32f multiplier = -0.5f / (stdDev * stdDev);
        Rpp32s halfHeight = (Rpp32s) (roi.xywhROI.roiHeight >> 1);
        Rpp32s halfWidth = (Rpp32s) (roi.xywhROI.roiWidth >> 1);

        Rpp16f *srcPtr1Image, *srcPtr2Image, *dstPtrImage;
        srcPtr1Image = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        srcPtr2Image = srcPtr2 + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;

        Rpp16f *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
        srcPtr1Channel = srcPtr1Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtr2Channel = srcPtr2Image + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        __m256 pMultiplier = _mm256_set1_ps(multiplier);
        __m256 pHalfWidth = _mm256_set1_ps(halfWidth);

        // Non linear blend with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    Rpp32f srcPtr1Temp_ps[24], srcPtr2Temp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    for(int cnt = 0; cnt < 24; cnt++)
                    {
                        srcPtr1Temp_ps[cnt] = (Rpp32f) srcPtr1Temp[cnt];
                        srcPtr2Temp_ps[cnt] = (Rpp32f) srcPtr2Temp[cnt];
                    }
                    __m256 p1[3], p2[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr1Temp_ps, p1);                              // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr2Temp_ps, p2);                              // simd loads
                    compute_non_linear_blend_24_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p1);  // simd stores
                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    srcPtr1Temp += 24;
                    srcPtr2Temp += 24;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(((srcPtr1Temp[0] - srcPtr2Temp[0]) * gaussianValue) + srcPtr2Temp[0]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(((srcPtr1Temp[1] - srcPtr2Temp[1]) * gaussianValue) + srcPtr2Temp[1]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(((srcPtr1Temp[2] - srcPtr2Temp[2]) * gaussianValue) + srcPtr2Temp[2]);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    Rpp32f srcPtr1TempR_ps[8], srcPtr1TempG_ps[8], srcPtr1TempB_ps[8];
                    Rpp32f srcPtr2TempR_ps[8], srcPtr2TempG_ps[8], srcPtr2TempB_ps[8];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        srcPtr1TempR_ps[cnt] = (Rpp32f) srcPtr1TempR[cnt];
                        srcPtr1TempG_ps[cnt] = (Rpp32f) srcPtr1TempG[cnt];
                        srcPtr1TempB_ps[cnt] = (Rpp32f) srcPtr1TempB[cnt];
                        srcPtr2TempR_ps[cnt] = (Rpp32f) srcPtr2TempR[cnt];
                        srcPtr2TempG_ps[cnt] = (Rpp32f) srcPtr2TempG[cnt];
                        srcPtr2TempB_ps[cnt] = (Rpp32f) srcPtr2TempB[cnt];
                    }
                    __m256 p1[3], p2[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr1TempR_ps, srcPtr1TempG_ps, srcPtr1TempB_ps, p1); // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr2TempR_ps, srcPtr2TempG_ps, srcPtr2TempB_ps, p2); // simd loads
                    compute_non_linear_blend_24_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p1);                             // simd stores
                    for(int cnt = 0; cnt < 24; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtr1TempR += 8;
                    srcPtr1TempG += 8;
                    srcPtr1TempB += 8;
                    srcPtr2TempR += 8;
                    srcPtr2TempG += 8;
                    srcPtr2TempB += 8;
                    dstPtrTemp += 24;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((*srcPtr1TempR - *srcPtr2TempR) * gaussianValue) + *srcPtr2TempR);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((*srcPtr1TempG - *srcPtr2TempG) * gaussianValue) + *srcPtr2TempG);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((*srcPtr1TempB - *srcPtr2TempB) * gaussianValue) + *srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    Rpp32f srcPtr1Temp_ps[24], srcPtr2Temp_ps[24];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < 24; cnt++)
                    {
                        srcPtr1Temp_ps[cnt] = (Rpp32f) srcPtr1Temp[cnt];
                        srcPtr2Temp_ps[cnt] = (Rpp32f) srcPtr2Temp[cnt];
                    }
                    __m256 p1[3], p2[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr1Temp_ps, p1);                      // simd loads
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr2Temp_ps, p2);                      // simd loads
                    compute_non_linear_blend_24_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);  // non_linear_blend adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p1);                     // simd stores
                    for(int cnt = 0; cnt < 24; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtr1Temp += 24;
                    srcPtr2Temp += 24;
                    dstPtrTemp += 24;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((srcPtr1Temp[0] - srcPtr2Temp[0]) * gaussianValue) + srcPtr2Temp[0]);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((srcPtr1Temp[1] - srcPtr2Temp[1]) * gaussianValue) + srcPtr2Temp[1]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((srcPtr1Temp[2] - srcPtr2Temp[2]) * gaussianValue) + srcPtr2Temp[2]);

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTemp += 3;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1RowR = srcPtr1Channel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = srcPtr2Channel;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    Rpp32f srcPtr1TempR_ps[8], srcPtr1TempG_ps[8], srcPtr1TempB_ps[8];
                    Rpp32f srcPtr2TempR_ps[8], srcPtr2TempG_ps[8], srcPtr2TempB_ps[8];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        srcPtr1TempR_ps[cnt] = (Rpp32f) srcPtr1TempR[cnt];
                        srcPtr1TempG_ps[cnt] = (Rpp32f) srcPtr1TempG[cnt];
                        srcPtr1TempB_ps[cnt] = (Rpp32f) srcPtr1TempB[cnt];
                        srcPtr2TempR_ps[cnt] = (Rpp32f) srcPtr2TempR[cnt];
                        srcPtr2TempG_ps[cnt] = (Rpp32f) srcPtr2TempG[cnt];
                        srcPtr2TempB_ps[cnt] = (Rpp32f) srcPtr2TempB[cnt];
                    }
                    __m256 p1[6], p2[6];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr1TempR_ps, srcPtr1TempG_ps, srcPtr1TempB_ps, p1); // simd loads
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr2TempR_ps, srcPtr2TempG_ps, srcPtr2TempB_ps, p2); // simd loads
                    compute_non_linear_blend_24_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);          // non_linear_blend adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p1);  // simd stores
                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    srcPtr1TempR += 8;
                    srcPtr1TempG += 8;
                    srcPtr1TempB += 8;
                    srcPtr2TempR += 8;
                    srcPtr2TempG += 8;
                    srcPtr2TempB += 8;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);

                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(((*srcPtr1TempR - *srcPtr2TempR) * gaussianValue) + *srcPtr2TempR);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(((*srcPtr1TempG - *srcPtr2TempG) * gaussianValue) + *srcPtr2TempG);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(((*srcPtr1TempB - *srcPtr2TempB) * gaussianValue) + *srcPtr2TempB);

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Non linear blend without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtr1Channel;
            srcPtr2Row = srcPtr2Channel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;

                Rpp32s iLoc = i - halfHeight;
                Rpp32f iLocComponent = iLoc * iLoc * multiplier;
                __m256 pILocComponent = _mm256_set1_ps(iLocComponent);
                __m256 pJLocComponent = _mm256_sub_ps(_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7), pHalfWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    Rpp32f srcPtr1Temp_ps[8], srcPtr2Temp_ps[8], dstPtrTemp_ps[8];
                    for(int cnt = 0; cnt < 8; cnt++)
                    {
                        srcPtr1Temp_ps[cnt] = (Rpp32f) srcPtr1Temp[cnt];
                        srcPtr2Temp_ps[cnt] = (Rpp32f) srcPtr2Temp[cnt];
                    }
                    __m256 p1[2], p2[2];
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr1Temp_ps, p1);                               // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr2Temp_ps, p2);                               // simd loads
                    compute_non_linear_blend_8_host(p1, p2, pMultiplier, pILocComponent, pJLocComponent);   // non_linear_blend adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p1);                              // simd stores
                    for(int cnt = 0; cnt < 8; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtr1Temp += 8;
                    srcPtr2Temp += 8;
                    dstPtrTemp += 8;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32s jLoc = vectorLoopCount - halfWidth;
                    Rpp32f jLocComponent = jLoc * jLoc * multiplier;
                    Rpp32f gaussianValue = std::exp(iLocComponent + jLocComponent);
                    *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((*srcPtr1Temp - *srcPtr2Temp) * gaussianValue) + *srcPtr2Temp);
                    srcPtr1Temp++;
                    srcPtr2Temp++;
                    dstPtrTemp++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
