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

inline void compute_variance_8_host(__m256d *p1, __m256d *pMean, __m256d *pVar)
{
    __m256d pSub = _mm256_sub_pd(p1[0], pMean[0]);
    pVar[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVar[0]);
    pSub = _mm256_sub_pd(p1[1], pMean[0]);
    pVar[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVar[0]);
}

inline void compute_variance_channel_pln3_24_host(__m256d *p1, __m256d *pMeanR, __m256d *pMeanG, __m256d *pMeanB, __m256d *pVarR, __m256d *pVarG, __m256d *pVarB)
{
    __m256d pSub = _mm256_sub_pd(p1[0], pMeanR[0]);
    pVarR[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarR[0]);
    pSub = _mm256_sub_pd(p1[1], pMeanR[0]);
    pVarR[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarR[0]);
    pSub = _mm256_sub_pd(p1[2], pMeanG[0]);
    pVarG[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarG[0]);
    pSub = _mm256_sub_pd(p1[3], pMeanG[0]);
    pVarG[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarG[0]);
    pSub = _mm256_sub_pd(p1[4], pMeanB[0]);
    pVarB[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarB[0]);
    pSub = _mm256_sub_pd(p1[5], pMeanB[0]);
    pVarB[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarB[0]);
}

inline void compute_variance_image_pln3_24_host(__m256d *p1, __m256d *pMean, __m256d *pVarR, __m256d *pVarG, __m256d *pVarB)
{
    __m256d pSub = _mm256_sub_pd(p1[0], pMean[0]);
    pVarR[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarR[0]);
    pSub = _mm256_sub_pd(p1[1], pMean[0]);
    pVarR[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarR[0]);
    pSub = _mm256_sub_pd(p1[2], pMean[0]);
    pVarG[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarG[0]);
    pSub = _mm256_sub_pd(pMean[0], p1[3]);
    pVarG[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarG[0]);
    pSub = _mm256_sub_pd(p1[4], pMean[0]);
    pVarB[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarB[0]);
    pSub = _mm256_sub_pd(p1[5], pMean[0]);
    pVarB[0] = _mm256_add_pd(_mm256_mul_pd(pSub, pSub), pVarB[0]);
}

RppStatus tensor_stddev_u8_f32_host(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *tensorStddevArr,
                                    Rpp32f *meanTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        Rpp32f totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;
        int idx = batchCount * 4;

        // Tensor Stddev without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            Rpp64f var = 0.0;
            Rpp32f stddev = 0.0;
            Rpp64f varAvx[4] = {0.0};
            Rpp32f mean = meanTensor[batchCount];

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMean = _mm256_set1_pd(mean);
            __m256d pVar = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_u8_to_f64_avx, srcPtrTemp, p1);
                    compute_variance_8_host(p1, &pMean, &pVar);

                    srcPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    var += (static_cast<Rpp64f>(*srcPtrTemp) - mean) * (static_cast<Rpp64f>(*srcPtrTemp) - mean);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvx, pVar);
            var += (varAvx[0] + varAvx[1] + varAvx[2] + varAvx[3]);
#endif
            stddev = sqrt(var / totalPixelsPerChannel);
            tensorStddevArr[batchCount] = static_cast<Rpp32f>(stddev);
        }

        // Tensor Stddev without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f varR, varG, varB, varImageR, varImageG, varImageB, varImage;
            Rpp32f stddevImage, stddevR, stddevG, stddevB;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};
            Rpp64f varAvxImageR[4] = {0.0};
            Rpp64f varAvxImageG[4] = {0.0};
            Rpp64f varAvxImageB[4] = {0.0};
            varR = varG = varB = varImageR = varImageG = varImageB = 0.0;

            Rpp32f meanR     = meanTensor[idx];
            Rpp32f meanG     = meanTensor[idx + 1];
            Rpp32f meanB     = meanTensor[idx + 2];
            Rpp32f meanImage = meanTensor[idx + 3];

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d pMeanR     = _mm256_set1_pd(meanR);
            __m256d pMeanG     = _mm256_set1_pd(meanG);
            __m256d pMeanB     = _mm256_set1_pd(meanB);
            __m256d pMeanImage = _mm256_set1_pd(meanImage);
            __m256d pVarR, pVarG, pVarB;
            __m256d pVarImageR, pVarImageG, pVarImageB;
            pVarR = pVarG = pVarB = pVarImageR = pVarImageG = pVarImageB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_u8pln3_to_f64pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_variance_channel_pln3_24_host(p, &pMeanR, &pMeanG, &pMeanB, &pVarR, &pVarG, &pVarB);
                    compute_variance_image_pln3_24_host(p, &pMeanImage, &pVarImageR, &pVarImageG, &pVarImageB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp64f srcPtrR = static_cast<Rpp64f>(*srcPtrTempR);
                    Rpp64f srcPtrG = static_cast<Rpp64f>(*srcPtrTempG);
                    Rpp64f srcPtrB = static_cast<Rpp64f>(*srcPtrTempB);
                    varR += (srcPtrR - meanR) * (srcPtrR - meanR);
                    varG += (srcPtrG - meanG) * (srcPtrG - meanG);
                    varB += (srcPtrB - meanB) * (srcPtrB - meanB);
                    varImageR += (srcPtrR - meanImage) * (srcPtrR - meanImage);
                    varImageG += (srcPtrG - meanImage) * (srcPtrG - meanImage);
                    varImageB += (srcPtrB - meanImage) * (srcPtrB - meanImage);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvxR, pVarR);
            _mm256_storeu_pd(varAvxG, pVarG);
            _mm256_storeu_pd(varAvxB, pVarB);
            _mm256_storeu_pd(varAvxImageR, pVarImageR);
            _mm256_storeu_pd(varAvxImageG, pVarImageG);
            _mm256_storeu_pd(varAvxImageB, pVarImageB);

            varR += (varAvxR[0] + varAvxR[1] + varAvxR[2] + varAvxR[3]);
            varG += (varAvxG[0] + varAvxG[1] + varAvxG[2] + varAvxG[3]);
            varB += (varAvxB[0] + varAvxB[1] + varAvxB[2] + varAvxB[3]);
            varImageR += (varAvxImageR[0] + varAvxImageR[1] + varAvxImageR[2] + varAvxImageR[3]);
            varImageG += (varAvxImageG[0] + varAvxImageG[1] + varAvxImageG[2] + varAvxImageG[3]);
            varImageB += (varAvxImageB[0] + varAvxImageB[1] + varAvxImageB[2] + varAvxImageB[3]);
#endif
            varImage = varImageR + varImageG + varImageB;
            stddevImage = static_cast<Rpp32f>(sqrt(varImage / (totalPixelsPerChannel * 3)));
            stddevR     = static_cast<Rpp32f>(sqrt(varR / totalPixelsPerChannel));
            stddevG     = static_cast<Rpp32f>(sqrt(varG / totalPixelsPerChannel));
            stddevB     = static_cast<Rpp32f>(sqrt(varB / totalPixelsPerChannel));
            tensorStddevArr[idx] = stddevR;
            tensorStddevArr[idx + 1] = stddevG;
            tensorStddevArr[idx + 2] = stddevB;
            tensorStddevArr[idx + 3] = stddevImage;
        }

        // Tensor Stddev without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f varR, varG, varB, varImageR, varImageG, varImageB, varImage;
            Rpp32f stddevImage, stddevR, stddevG, stddevB;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};
            Rpp64f varAvxImageR[4] = {0.0};
            Rpp64f varAvxImageG[4] = {0.0};
            Rpp64f varAvxImageB[4] = {0.0};
            varR = varG = varB = varImageR = varImageG = varImageB = 0.0;

            Rpp32f meanR     = meanTensor[idx];
            Rpp32f meanG     = meanTensor[idx + 1];
            Rpp32f meanB     = meanTensor[idx + 2];
            Rpp32f meanImage = meanTensor[idx + 3];

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMeanR     = _mm256_set1_pd(meanR);
            __m256d pMeanG     = _mm256_set1_pd(meanG);
            __m256d pMeanB     = _mm256_set1_pd(meanB);
            __m256d pMeanImage = _mm256_set1_pd(meanImage);
            __m256d pVarR, pVarG, pVarB;
            __m256d pVarImageR, pVarImageG, pVarImageB;
            pVarR = pVarG = pVarB = pVarImageR = pVarImageG = pVarImageB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_u8pkd3_to_f64pln3_avx, srcPtrTemp, p);
                    compute_variance_channel_pln3_24_host(p, &pMeanR, &pMeanG, &pMeanB, &pVarR, &pVarG, &pVarB);
                    compute_variance_image_pln3_24_host(p, &pMeanImage, &pVarImageR, &pVarImageG, &pVarImageB);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp64f srcPtrR = static_cast<Rpp64f>(srcPtrTemp[0]);
                    Rpp64f srcPtrG = static_cast<Rpp64f>(srcPtrTemp[1]);
                    Rpp64f srcPtrB = static_cast<Rpp64f>(srcPtrTemp[2]);
                    varR += (srcPtrR - meanR) * (srcPtrR - meanR);
                    varG += (srcPtrG - meanG) * (srcPtrG - meanG);
                    varB += (srcPtrB - meanB) * (srcPtrB - meanB);
                    varImageR += (srcPtrR - meanImage) * (srcPtrR - meanImage);
                    varImageG += (srcPtrG - meanImage) * (srcPtrG - meanImage);
                    varImageB += (srcPtrB - meanImage) * (srcPtrB - meanImage);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvxR, pVarR);
            _mm256_storeu_pd(varAvxG, pVarG);
            _mm256_storeu_pd(varAvxB, pVarB);
            _mm256_storeu_pd(varAvxImageR, pVarImageR);
            _mm256_storeu_pd(varAvxImageG, pVarImageG);
            _mm256_storeu_pd(varAvxImageB, pVarImageB);

            varR += (varAvxR[0] + varAvxR[1] + varAvxR[2] + varAvxR[3]);
            varG += (varAvxG[0] + varAvxG[1] + varAvxG[2] + varAvxG[3]);
            varB += (varAvxB[0] + varAvxB[1] + varAvxB[2] + varAvxB[3]);
            varImageR += (varAvxImageR[0] + varAvxImageR[1] + varAvxImageR[2] + varAvxImageR[3]);
            varImageG += (varAvxImageG[0] + varAvxImageG[1] + varAvxImageG[2] + varAvxImageG[3]);
            varImageB += (varAvxImageB[0] + varAvxImageB[1] + varAvxImageB[2] + varAvxImageB[3]);
#endif
            varImage = varImageR + varImageG + varImageB;
            stddevImage = static_cast<Rpp32f>(sqrt(varImage / (totalPixelsPerChannel * 3)));
            stddevR     = static_cast<Rpp32f>(sqrt(varR / totalPixelsPerChannel));
            stddevG     = static_cast<Rpp32f>(sqrt(varG / totalPixelsPerChannel));
            stddevB     = static_cast<Rpp32f>(sqrt(varB / totalPixelsPerChannel));
            tensorStddevArr[idx] = stddevR;
            tensorStddevArr[idx + 1] = stddevG;
            tensorStddevArr[idx + 2] = stddevB;
            tensorStddevArr[idx + 3] = stddevImage;
        }
    }

    return RPP_SUCCESS;
}

RppStatus tensor_stddev_f32_f32_host(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *tensorStddevArr,
                                     Rpp32f *meanTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        Rpp32f totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;
        int idx = batchCount * 4;

        // Tensor Stddev without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            Rpp64f var = 0.0;
            Rpp32f stddev = 0.0;
            Rpp64f varAvx[4] = {0.0};
            Rpp32f mean = meanTensor[batchCount];

            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMean = _mm256_set1_pd(mean);
            __m256d pVar = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_f32_to_f64_avx, srcPtrTemp, p1);
                    compute_variance_8_host(p1, &pMean, &pVar);
                    srcPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    var += (static_cast<Rpp64f>(*srcPtrTemp) - mean) * (static_cast<Rpp64f>(*srcPtrTemp) - mean);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvx, pVar);
            var += (varAvx[0] + varAvx[1] + varAvx[2] + varAvx[3]);
#endif
            stddev = sqrt(var / totalPixelsPerChannel) * 255;
            tensorStddevArr[batchCount] = static_cast<Rpp32f>(stddev);
        }

        // Tensor Stddev without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f varR, varG, varB, varImageR, varImageG, varImageB, varImage;
            Rpp32f stddevImage, stddevR, stddevG, stddevB;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};
            Rpp64f varAvxImageR[4] = {0.0};
            Rpp64f varAvxImageG[4] = {0.0};
            Rpp64f varAvxImageB[4] = {0.0};
            varR = varG = varB = varImageR = varImageG = varImageB = 0.0;

            Rpp32f meanR     = meanTensor[idx];
            Rpp32f meanG     = meanTensor[idx + 1];
            Rpp32f meanB     = meanTensor[idx + 2];
            Rpp32f meanImage = meanTensor[idx + 3];

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d pMeanR     = _mm256_set1_pd(meanR);
            __m256d pMeanG     = _mm256_set1_pd(meanG);
            __m256d pMeanB     = _mm256_set1_pd(meanB);
            __m256d pMeanImage = _mm256_set1_pd(meanImage);
            __m256d pVarR, pVarG, pVarB;
            __m256d pVarImageR, pVarImageG, pVarImageB;
            pVarR = pVarG = pVarB = pVarImageR = pVarImageG = pVarImageB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_f32pln3_to_f64pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_variance_channel_pln3_24_host(p, &pMeanR, &pMeanG, &pMeanB, &pVarR, &pVarG, &pVarB);
                    compute_variance_image_pln3_24_host(p, &pMeanImage, &pVarImageR, &pVarImageG, &pVarImageB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp64f srcPtrR = static_cast<Rpp64f>(*srcPtrTempR);
                    Rpp64f srcPtrG = static_cast<Rpp64f>(*srcPtrTempG);
                    Rpp64f srcPtrB = static_cast<Rpp64f>(*srcPtrTempB);
                    varR += (srcPtrR - meanR) * (srcPtrR - meanR);
                    varG += (srcPtrG - meanG) * (srcPtrG - meanG);
                    varB += (srcPtrB - meanB) * (srcPtrB - meanB);
                    varImageR += (srcPtrR - meanImage) * (srcPtrR - meanImage);
                    varImageG += (srcPtrG - meanImage) * (srcPtrG - meanImage);
                    varImageB += (srcPtrB - meanImage) * (srcPtrB - meanImage);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvxR, pVarR);
            _mm256_storeu_pd(varAvxG, pVarG);
            _mm256_storeu_pd(varAvxB, pVarB);
            _mm256_storeu_pd(varAvxImageR, pVarImageR);
            _mm256_storeu_pd(varAvxImageG, pVarImageG);
            _mm256_storeu_pd(varAvxImageB, pVarImageB);

            varR += (varAvxR[0] + varAvxR[1] + varAvxR[2] + varAvxR[3]);
            varG += (varAvxG[0] + varAvxG[1] + varAvxG[2] + varAvxG[3]);
            varB += (varAvxB[0] + varAvxB[1] + varAvxB[2] + varAvxB[3]);
            varImageR += (varAvxImageR[0] + varAvxImageR[1] + varAvxImageR[2] + varAvxImageR[3]);
            varImageG += (varAvxImageG[0] + varAvxImageG[1] + varAvxImageG[2] + varAvxImageG[3]);
            varImageB += (varAvxImageB[0] + varAvxImageB[1] + varAvxImageB[2] + varAvxImageB[3]);
#endif
            varImage = varImageR + varImageG + varImageB;
            stddevImage = static_cast<Rpp32f>(sqrt(varImage / (totalPixelsPerChannel * 3)) * 255); // multiply by 255 to normalize variation
            stddevR     = static_cast<Rpp32f>(sqrt(varR / totalPixelsPerChannel) * 255);
            stddevG     = static_cast<Rpp32f>(sqrt(varG / totalPixelsPerChannel) * 255);
            stddevB     = static_cast<Rpp32f>(sqrt(varB / totalPixelsPerChannel) * 255);
            tensorStddevArr[idx] = stddevR;
            tensorStddevArr[idx + 1] = stddevG;
            tensorStddevArr[idx + 2] = stddevB;
            tensorStddevArr[idx + 3] = stddevImage;
        }

        // Tensor Stddev without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f varR, varG, varB, varImageR, varImageG, varImageB, varImage;
            Rpp32f stddevImage, stddevR, stddevG, stddevB;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};
            Rpp64f varAvxImageR[4] = {0.0};
            Rpp64f varAvxImageG[4] = {0.0};
            Rpp64f varAvxImageB[4] = {0.0};
            varR = varG = varB = varImageR = varImageG = varImageB = 0.0;

            Rpp32f meanR     = meanTensor[idx];
            Rpp32f meanG     = meanTensor[idx + 1];
            Rpp32f meanB     = meanTensor[idx + 2];
            Rpp32f meanImage = meanTensor[idx + 3];

            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMeanR     = _mm256_set1_pd(meanR);
            __m256d pMeanG     = _mm256_set1_pd(meanG);
            __m256d pMeanB     = _mm256_set1_pd(meanB);
            __m256d pMeanImage = _mm256_set1_pd(meanImage);
            __m256d pVarR, pVarG, pVarB;
            __m256d pVarImageR, pVarImageG, pVarImageB;
            pVarR = pVarG = pVarB = pVarImageR = pVarImageG = pVarImageB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f64pln3_avx, srcPtrTemp, p);
                    compute_variance_channel_pln3_24_host(p, &pMeanR, &pMeanG, &pMeanB, &pVarR, &pVarG, &pVarB);
                    compute_variance_image_pln3_24_host(p, &pMeanImage, &pVarImageR, &pVarImageG, &pVarImageB);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp64f srcPtrR = static_cast<Rpp64f>(srcPtrTemp[0]);
                    Rpp64f srcPtrG = static_cast<Rpp64f>(srcPtrTemp[1]);
                    Rpp64f srcPtrB = static_cast<Rpp64f>(srcPtrTemp[2]);
                    varR += (srcPtrR - meanR) * (srcPtrR - meanR);
                    varG += (srcPtrG - meanG) * (srcPtrG - meanG);
                    varB += (srcPtrB - meanB) * (srcPtrB - meanB);
                    varImageR += (srcPtrR - meanImage) * (srcPtrR - meanImage);
                    varImageG += (srcPtrG - meanImage) * (srcPtrG - meanImage);
                    varImageB += (srcPtrB - meanImage) * (srcPtrB - meanImage);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvxR, pVarR);
            _mm256_storeu_pd(varAvxG, pVarG);
            _mm256_storeu_pd(varAvxB, pVarB);
            _mm256_storeu_pd(varAvxImageR, pVarImageR);
            _mm256_storeu_pd(varAvxImageG, pVarImageG);
            _mm256_storeu_pd(varAvxImageB, pVarImageB);

            varR += (varAvxR[0] + varAvxR[1] + varAvxR[2] + varAvxR[3]);
            varG += (varAvxG[0] + varAvxG[1] + varAvxG[2] + varAvxG[3]);
            varB += (varAvxB[0] + varAvxB[1] + varAvxB[2] + varAvxB[3]);
            varImageR += (varAvxImageR[0] + varAvxImageR[1] + varAvxImageR[2] + varAvxImageR[3]);
            varImageG += (varAvxImageG[0] + varAvxImageG[1] + varAvxImageG[2] + varAvxImageG[3]);
            varImageB += (varAvxImageB[0] + varAvxImageB[1] + varAvxImageB[2] + varAvxImageB[3]);
#endif
            varImage = varImageR + varImageG + varImageB;
            stddevImage = static_cast<Rpp32f>(sqrt(varImage / (totalPixelsPerChannel * 3)) * 255);
            stddevR     = static_cast<Rpp32f>(sqrt(varR / totalPixelsPerChannel) * 255);
            stddevG     = static_cast<Rpp32f>(sqrt(varG / totalPixelsPerChannel) * 255);
            stddevB     = static_cast<Rpp32f>(sqrt(varB / totalPixelsPerChannel) * 255);
            tensorStddevArr[idx] = stddevR;
            tensorStddevArr[idx + 1] = stddevG;
            tensorStddevArr[idx + 2] = stddevB;
            tensorStddevArr[idx + 3] = stddevImage;
        }
    }

    return RPP_SUCCESS;
}

RppStatus tensor_stddev_f16_f32_host(Rpp16f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *tensorStddevArr,
                                     Rpp32f *meanTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f *srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        Rpp32f totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;
        int idx = batchCount * 4;

        // Tensor Stddev without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            Rpp64f var = 0.0;
            Rpp32f stddev = 0.0;
            Rpp64f varAvx[4] = {0.0};
            Rpp32f mean = meanTensor[batchCount];

            Rpp16f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMean = _mm256_set1_pd(mean);
            __m256d pVar = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTemp_ps[8];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        srcPtrTemp_ps[cnt] = static_cast<Rpp32f>(srcPtrTemp[cnt]);

                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_f32_to_f64_avx, srcPtrTemp_ps, p1);
                    compute_variance_8_host(p1, &pMean, &pVar);

                    srcPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    var += (static_cast<Rpp64f>(*srcPtrTemp) - mean) * (static_cast<Rpp64f>(*srcPtrTemp) - mean);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvx, pVar);
            var += (varAvx[0] + varAvx[1] + varAvx[2] + varAvx[3]);
#endif
            stddev = sqrt(var / totalPixelsPerChannel) * 255;
            tensorStddevArr[batchCount] = static_cast<Rpp32f>(stddev);
        }

        // Tensor Stddev without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f varR, varG, varB, varImageR, varImageG, varImageB, varImage;
            Rpp32f stddevImage, stddevR, stddevG, stddevB;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};
            Rpp64f varAvxImageR[4] = {0.0};
            Rpp64f varAvxImageG[4] = {0.0};
            Rpp64f varAvxImageB[4] = {0.0};
            varR = varG = varB = varImageR = varImageG = varImageB = 0.0;

            Rpp32f meanR     = meanTensor[idx];
            Rpp32f meanG     = meanTensor[idx + 1];
            Rpp32f meanB     = meanTensor[idx + 2];
            Rpp32f meanImage = meanTensor[idx + 3];

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d pMeanR     = _mm256_set1_pd(meanR);
            __m256d pMeanG     = _mm256_set1_pd(meanG);
            __m256d pMeanB     = _mm256_set1_pd(meanB);
            __m256d pMeanImage = _mm256_set1_pd(meanImage);
            __m256d pVarR, pVarG, pVarB;
            __m256d pVarImageR, pVarImageG, pVarImageB;
            pVarR = pVarG = pVarB = pVarImageR = pVarImageG = pVarImageB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = static_cast<Rpp32f>(srcPtrTempR[cnt]);
                        srcPtrTempG_ps[cnt] = static_cast<Rpp32f>(srcPtrTempG[cnt]);
                        srcPtrTempB_ps[cnt] = static_cast<Rpp32f>(srcPtrTempB[cnt]);
                    }

                    __m256d p[6];
                    rpp_simd_load(rpp_load24_f32pln3_to_f64pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);
                    compute_variance_channel_pln3_24_host(p, &pMeanR, &pMeanG, &pMeanB, &pVarR, &pVarG, &pVarB);
                    compute_variance_image_pln3_24_host(p, &pMeanImage, &pVarImageR, &pVarImageG, &pVarImageB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp64f srcPtrR = static_cast<Rpp64f>(*srcPtrTempR);
                    Rpp64f srcPtrG = static_cast<Rpp64f>(*srcPtrTempG);
                    Rpp64f srcPtrB = static_cast<Rpp64f>(*srcPtrTempB);
                    varR += (srcPtrR - meanR) * (srcPtrR - meanR);
                    varG += (srcPtrG - meanG) * (srcPtrG - meanG);
                    varB += (srcPtrB - meanB) * (srcPtrB - meanB);
                    varImageR += (srcPtrR - meanImage) * (srcPtrR - meanImage);
                    varImageG += (srcPtrG - meanImage) * (srcPtrG - meanImage);
                    varImageB += (srcPtrB - meanImage) * (srcPtrB - meanImage);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvxR, pVarR);
            _mm256_storeu_pd(varAvxG, pVarG);
            _mm256_storeu_pd(varAvxB, pVarB);
            _mm256_storeu_pd(varAvxImageR, pVarImageR);
            _mm256_storeu_pd(varAvxImageG, pVarImageG);
            _mm256_storeu_pd(varAvxImageB, pVarImageB);

            varR += (varAvxR[0] + varAvxR[1] + varAvxR[2] + varAvxR[3]);
            varG += (varAvxG[0] + varAvxG[1] + varAvxG[2] + varAvxG[3]);
            varB += (varAvxB[0] + varAvxB[1] + varAvxB[2] + varAvxB[3]);
            varImageR += (varAvxImageR[0] + varAvxImageR[1] + varAvxImageR[2] + varAvxImageR[3]);
            varImageG += (varAvxImageG[0] + varAvxImageG[1] + varAvxImageG[2] + varAvxImageG[3]);
            varImageB += (varAvxImageB[0] + varAvxImageB[1] + varAvxImageB[2] + varAvxImageB[3]);
#endif
            varImage = varImageR + varImageG + varImageB;
            stddevImage = static_cast<Rpp32f>(sqrt(varImage / (totalPixelsPerChannel * 3)) * 255); // multiply by 255 to normalize variation
            stddevR     = static_cast<Rpp32f>(sqrt(varR / totalPixelsPerChannel) * 255);
            stddevG     = static_cast<Rpp32f>(sqrt(varG / totalPixelsPerChannel) * 255);
            stddevB     = static_cast<Rpp32f>(sqrt(varB / totalPixelsPerChannel) * 255);
            tensorStddevArr[idx] = stddevR;
            tensorStddevArr[idx + 1] = stddevG;
            tensorStddevArr[idx + 2] = stddevB;
            tensorStddevArr[idx + 3] = stddevImage;
        }

        // Tensor Stddev without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f varR, varG, varB, varImageR, varImageG, varImageB, varImage;
            Rpp32f stddevImage, stddevR, stddevG, stddevB;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};
            Rpp64f varAvxImageR[4] = {0.0};
            Rpp64f varAvxImageG[4] = {0.0};
            Rpp64f varAvxImageB[4] = {0.0};
            varR = varG = varB = varImageR = varImageG = varImageB = 0.0;

            Rpp32f meanR     = meanTensor[idx];
            Rpp32f meanG     = meanTensor[idx + 1];
            Rpp32f meanB     = meanTensor[idx + 2];
            Rpp32f meanImage = meanTensor[idx + 3];

            Rpp16f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMeanR     = _mm256_set1_pd(meanR);
            __m256d pMeanG     = _mm256_set1_pd(meanG);
            __m256d pMeanB     = _mm256_set1_pd(meanB);
            __m256d pMeanImage = _mm256_set1_pd(meanImage);
            __m256d pVarR, pVarG, pVarB;
            __m256d pVarImageR, pVarImageG, pVarImageB;
            pVarR = pVarG = pVarB = pVarImageR = pVarImageG = pVarImageB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = static_cast<Rpp32f>(srcPtrTemp[cnt]);

                    __m256d p[6];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f64pln3_avx, srcPtrTemp_ps, p);
                    compute_variance_channel_pln3_24_host(p, &pMeanR, &pMeanG, &pMeanB, &pVarR, &pVarG, &pVarB);
                    compute_variance_image_pln3_24_host(p, &pMeanImage, &pVarImageR, &pVarImageG, &pVarImageB);

                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp64f srcPtrR = static_cast<Rpp64f>(srcPtrTemp[0]);
                    Rpp64f srcPtrG = static_cast<Rpp64f>(srcPtrTemp[1]);
                    Rpp64f srcPtrB = static_cast<Rpp64f>(srcPtrTemp[2]);
                    varR += (srcPtrR - meanR) * (srcPtrR - meanR);
                    varG += (srcPtrG - meanG) * (srcPtrG - meanG);
                    varB += (srcPtrB - meanB) * (srcPtrB - meanB);
                    varImageR += (srcPtrR - meanImage) * (srcPtrR - meanImage);
                    varImageG += (srcPtrG - meanImage) * (srcPtrG - meanImage);
                    varImageB += (srcPtrB - meanImage) * (srcPtrB - meanImage);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvxR, pVarR);
            _mm256_storeu_pd(varAvxG, pVarG);
            _mm256_storeu_pd(varAvxB, pVarB);
            _mm256_storeu_pd(varAvxImageR, pVarImageR);
            _mm256_storeu_pd(varAvxImageG, pVarImageG);
            _mm256_storeu_pd(varAvxImageB, pVarImageB);

            varR += (varAvxR[0] + varAvxR[1] + varAvxR[2] + varAvxR[3]);
            varG += (varAvxG[0] + varAvxG[1] + varAvxG[2] + varAvxG[3]);
            varB += (varAvxB[0] + varAvxB[1] + varAvxB[2] + varAvxB[3]);
            varImageR += (varAvxImageR[0] + varAvxImageR[1] + varAvxImageR[2] + varAvxImageR[3]);
            varImageG += (varAvxImageG[0] + varAvxImageG[1] + varAvxImageG[2] + varAvxImageG[3]);
            varImageB += (varAvxImageB[0] + varAvxImageB[1] + varAvxImageB[2] + varAvxImageB[3]);
#endif
            varImage = varImageR + varImageG + varImageB;
            stddevImage = static_cast<Rpp32f>(sqrt(varImage / (totalPixelsPerChannel * 3)) * 255);
            stddevR     = static_cast<Rpp32f>(sqrt(varR / totalPixelsPerChannel) * 255);
            stddevG     = static_cast<Rpp32f>(sqrt(varG / totalPixelsPerChannel) * 255);
            stddevB     = static_cast<Rpp32f>(sqrt(varB / totalPixelsPerChannel) * 255);
            tensorStddevArr[idx] = stddevR;
            tensorStddevArr[idx + 1] = stddevG;
            tensorStddevArr[idx + 2] = stddevB;
            tensorStddevArr[idx + 3] = stddevImage;
        }
    }

    return RPP_SUCCESS;
}

RppStatus tensor_stddev_i8_f32_host(Rpp8s *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *tensorStddevArr,
                                    Rpp32f *meanTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8s *srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        Rpp32f totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;
        int idx = batchCount * 4;

        // Tensor Stddev without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            Rpp64f var = 0.0;
            Rpp32f stddev = 0.0;
            Rpp64f varAvx[4] = {0.0};
            Rpp32f mean = meanTensor[batchCount] + 128;

            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMean = _mm256_set1_pd(mean);
            __m256d pVar = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_i8_to_f64_avx, srcPtrTemp, p1);
                    compute_variance_8_host(p1, &pMean, &pVar);

                    srcPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    var += (static_cast<Rpp64f>(*srcPtrTemp + 128) - mean) * (static_cast<Rpp64f>(*srcPtrTemp + 128) - mean);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvx, pVar);
            var += (varAvx[0] + varAvx[1] + varAvx[2] + varAvx[3]);
#endif
            stddev = sqrt(var / totalPixelsPerChannel);
            tensorStddevArr[batchCount] = static_cast<Rpp32f>(stddev);
        }

        // Tensor Stddev without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f varR, varG, varB, varImageR, varImageG, varImageB, varImage;
            Rpp32f stddevImage, stddevR, stddevG, stddevB;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};
            Rpp64f varAvxImageR[4] = {0.0};
            Rpp64f varAvxImageG[4] = {0.0};
            Rpp64f varAvxImageB[4] = {0.0};
            varR = varG = varB = varImageR = varImageG = varImageB = 0.0;

            Rpp32f meanR     = meanTensor[idx] + 128;
            Rpp32f meanG     = meanTensor[idx + 1] + 128;
            Rpp32f meanB     = meanTensor[idx + 2] + 128;
            Rpp32f meanImage = meanTensor[idx + 3] + 128;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d pMeanR     = _mm256_set1_pd(meanR);
            __m256d pMeanG     = _mm256_set1_pd(meanG);
            __m256d pMeanB     = _mm256_set1_pd(meanB);
            __m256d pMeanImage = _mm256_set1_pd(meanImage);
            __m256d pVarR, pVarG, pVarB;
            __m256d pVarImageR, pVarImageG, pVarImageB;
            pVarR = pVarG = pVarB = pVarImageR = pVarImageG = pVarImageB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_i8pln3_to_f64pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_variance_channel_pln3_24_host(p, &pMeanR, &pMeanG, &pMeanB, &pVarR, &pVarG, &pVarB);
                    compute_variance_image_pln3_24_host(p, &pMeanImage, &pVarImageR, &pVarImageG, &pVarImageB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp64f srcPtrR = static_cast<Rpp64f>(*srcPtrTempR + 128);
                    Rpp64f srcPtrG = static_cast<Rpp64f>(*srcPtrTempG + 128);
                    Rpp64f srcPtrB = static_cast<Rpp64f>(*srcPtrTempB + 128);
                    varR += (srcPtrR - meanR) * (srcPtrR - meanR);
                    varG += (srcPtrG - meanG) * (srcPtrG - meanG);
                    varB += (srcPtrB - meanB) * (srcPtrB - meanB);
                    varImageR += (srcPtrR - meanImage) * (srcPtrR - meanImage);
                    varImageG += (srcPtrG - meanImage) * (srcPtrG - meanImage);
                    varImageB += (srcPtrB - meanImage) * (srcPtrB - meanImage);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvxR, pVarR);
            _mm256_storeu_pd(varAvxG, pVarG);
            _mm256_storeu_pd(varAvxB, pVarB);
            _mm256_storeu_pd(varAvxImageR, pVarImageR);
            _mm256_storeu_pd(varAvxImageG, pVarImageG);
            _mm256_storeu_pd(varAvxImageB, pVarImageB);

            varR += (varAvxR[0] + varAvxR[1] + varAvxR[2] + varAvxR[3]);
            varG += (varAvxG[0] + varAvxG[1] + varAvxG[2] + varAvxG[3]);
            varB += (varAvxB[0] + varAvxB[1] + varAvxB[2] + varAvxB[3]);
            varImageR += (varAvxImageR[0] + varAvxImageR[1] + varAvxImageR[2] + varAvxImageR[3]);
            varImageG += (varAvxImageG[0] + varAvxImageG[1] + varAvxImageG[2] + varAvxImageG[3]);
            varImageB += (varAvxImageB[0] + varAvxImageB[1] + varAvxImageB[2] + varAvxImageB[3]);
#endif
            varImage = varImageR + varImageG + varImageB;
            stddevImage = static_cast<Rpp32f>(sqrt(varImage / (totalPixelsPerChannel * 3)));
            stddevR     = static_cast<Rpp32f>(sqrt(varR / totalPixelsPerChannel));
            stddevG     = static_cast<Rpp32f>(sqrt(varG / totalPixelsPerChannel));
            stddevB     = static_cast<Rpp32f>(sqrt(varB / totalPixelsPerChannel));
            tensorStddevArr[idx] = stddevR;
            tensorStddevArr[idx + 1] = stddevG;
            tensorStddevArr[idx + 2] = stddevB;
            tensorStddevArr[idx + 3] = stddevImage;
        }

        // Tensor Stddev without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f varR, varG, varB, varImageR, varImageG, varImageB, varImage;
            Rpp32f stddevImage, stddevR, stddevG, stddevB;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};
            Rpp64f varAvxImageR[4] = {0.0};
            Rpp64f varAvxImageG[4] = {0.0};
            Rpp64f varAvxImageB[4] = {0.0};
            varR = varG = varB = varImageR = varImageG = varImageB = 0.0;

            Rpp32f meanR     = meanTensor[idx] + 128;
            Rpp32f meanG     = meanTensor[idx + 1] + 128;
            Rpp32f meanB     = meanTensor[idx + 2] + 128;
            Rpp32f meanImage = meanTensor[idx + 3] + 128;

            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMeanR     = _mm256_set1_pd(meanR);
            __m256d pMeanG     = _mm256_set1_pd(meanG);
            __m256d pMeanB     = _mm256_set1_pd(meanB);
            __m256d pMeanImage = _mm256_set1_pd(meanImage);
            __m256d pVarR, pVarG, pVarB;
            __m256d pVarImageR, pVarImageG, pVarImageB;
            pVarR = pVarG = pVarB = pVarImageR = pVarImageG = pVarImageB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_i8pkd3_to_f64pln3_avx, srcPtrTemp, p);
                    compute_variance_channel_pln3_24_host(p, &pMeanR, &pMeanG, &pMeanB, &pVarR, &pVarG, &pVarB);
                    compute_variance_image_pln3_24_host(p, &pMeanImage, &pVarImageR, &pVarImageG, &pVarImageB);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp64f srcPtrR = static_cast<Rpp64f>(srcPtrTemp[0] + 128);
                    Rpp64f srcPtrG = static_cast<Rpp64f>(srcPtrTemp[1] + 128);
                    Rpp64f srcPtrB = static_cast<Rpp64f>(srcPtrTemp[2] + 128);
                    varR += (srcPtrR - meanR) * (srcPtrR - meanR);
                    varG += (srcPtrG - meanG) * (srcPtrG - meanG);
                    varB += (srcPtrB - meanB) * (srcPtrB - meanB);
                    varImageR += (srcPtrR - meanImage) * (srcPtrR - meanImage);
                    varImageG += (srcPtrG - meanImage) * (srcPtrG - meanImage);
                    varImageB += (srcPtrB - meanImage) * (srcPtrB - meanImage);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(varAvxR, pVarR);
            _mm256_storeu_pd(varAvxG, pVarG);
            _mm256_storeu_pd(varAvxB, pVarB);
            _mm256_storeu_pd(varAvxImageR, pVarImageR);
            _mm256_storeu_pd(varAvxImageG, pVarImageG);
            _mm256_storeu_pd(varAvxImageB, pVarImageB);

            varR += (varAvxR[0] + varAvxR[1] + varAvxR[2] + varAvxR[3]);
            varG += (varAvxG[0] + varAvxG[1] + varAvxG[2] + varAvxG[3]);
            varB += (varAvxB[0] + varAvxB[1] + varAvxB[2] + varAvxB[3]);
            varImageR += (varAvxImageR[0] + varAvxImageR[1] + varAvxImageR[2] + varAvxImageR[3]);
            varImageG += (varAvxImageG[0] + varAvxImageG[1] + varAvxImageG[2] + varAvxImageG[3]);
            varImageB += (varAvxImageB[0] + varAvxImageB[1] + varAvxImageB[2] + varAvxImageB[3]);
#endif
            varImage = varImageR + varImageG + varImageB;
            stddevImage = static_cast<Rpp32f>(sqrt(varImage / (totalPixelsPerChannel * 3)));
            stddevR     = static_cast<Rpp32f>(sqrt(varR / totalPixelsPerChannel));
            stddevG     = static_cast<Rpp32f>(sqrt(varG / totalPixelsPerChannel));
            stddevB     = static_cast<Rpp32f>(sqrt(varB / totalPixelsPerChannel));
            tensorStddevArr[idx] = stddevR;
            tensorStddevArr[idx + 1] = stddevG;
            tensorStddevArr[idx + 2] = stddevB;
            tensorStddevArr[idx + 3] = stddevImage;
        }
    }

    return RPP_SUCCESS;
}
