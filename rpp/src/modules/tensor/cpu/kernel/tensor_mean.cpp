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
#include "rpp_cpu_arithmetic.hpp"

RppStatus tensor_mean_u8_f32_host(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *tensorMeanArr,
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
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32f totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;
        int idx = batchCount * 4;

        // Tensor Mean without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~15;
            Rpp32f mean = 0.0;
            Rpp32u sum = 0;
            Rpp32u sumAvx[8] = {0};

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256i pSum = avx_px0;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i p1[2];
                    rpp_simd_load(rpp_load16_u8_to_u32_avx, srcPtrTemp, p1);
                    compute_sum_16_host(p1, &pSum);
                    srcPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    sum += static_cast<Rpp32u>(*srcPtrTemp++);
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_store_si256((__m256i *)sumAvx, pSum);
            sum += (sumAvx[0] + sumAvx[1] + sumAvx[2] + sumAvx[3] + sumAvx[4] + sumAvx[5] + sumAvx[6] + sumAvx[7]);
#endif
            mean = static_cast<Rpp32f>(sum) / totalPixelsPerChannel;
            tensorMeanArr[batchCount] = mean;
        }

        // Tensor Mean without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64u sum;
            Rpp32u sumR = 0, sumG = 0, sumB = 0;
            Rpp32f mean, meanR = 0.0, meanG = 0.0, meanB = 0.0;
            Rpp32u sumAvxR[8] = {0};
            Rpp32u sumAvxG[8] = {0};
            Rpp32u sumAvxB[8] = {0};

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256i pSumR = avx_px0;
            __m256i pSumG = avx_px0;
            __m256i pSumB = avx_px0;
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
                    __m256i p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_u32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_sum_48_host(p, &pSumR, &pSumG, &pSumB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sumR += static_cast<Rpp32u>(*srcPtrTempR++);
                    sumG += static_cast<Rpp32u>(*srcPtrTempG++);
                    sumB += static_cast<Rpp32u>(*srcPtrTempB++);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_store_si256((__m256i *)sumAvxR, pSumR);
            _mm256_store_si256((__m256i *)sumAvxG, pSumG);
            _mm256_store_si256((__m256i *)sumAvxB, pSumB);
            sumR += (sumAvxR[0] + sumAvxR[1] + sumAvxR[2] + sumAvxR[3] + sumAvxR[4] + sumAvxR[5] + sumAvxR[6] + sumAvxR[7]);
            sumG += (sumAvxG[0] + sumAvxG[1] + sumAvxG[2] + sumAvxG[3] + sumAvxG[4] + sumAvxG[5] + sumAvxG[6] + sumAvxG[7]);
            sumB += (sumAvxB[0] + sumAvxB[1] + sumAvxB[2] + sumAvxB[3] + sumAvxB[4] + sumAvxB[5] + sumAvxB[6] + sumAvxB[7]);
#endif
            sum = static_cast<Rpp64u>(sumR) + static_cast<Rpp64u>(sumG) + static_cast<Rpp64u>(sumB);
            mean = (static_cast<Rpp64f>(sum) / (totalPixelsPerChannel * 3));
            meanR = (static_cast<Rpp32f>(sumR) / totalPixelsPerChannel);
            meanG = (static_cast<Rpp32f>(sumG) / totalPixelsPerChannel);
            meanB = (static_cast<Rpp32f>(sumB) / totalPixelsPerChannel);
            tensorMeanArr[idx] = meanR;
            tensorMeanArr[idx + 1] = meanG;
            tensorMeanArr[idx + 2] = meanB;
            tensorMeanArr[idx + 3] = mean;
        }

        // Tensor Mean without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64u sum;
            Rpp32u sumR = 0, sumG = 0, sumB = 0;
            Rpp32f mean, meanR = 0.0, meanG = 0.0, meanB = 0.0;
            Rpp32u sumAvxR[8] = {0};
            Rpp32u sumAvxG[8] = {0};
            Rpp32u sumAvxB[8] = {0};

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256i pSumR = avx_px0;
            __m256i pSumG = avx_px0;
            __m256i pSumB = avx_px0;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_u32pln3_avx, srcPtrTemp, p);
                    compute_sum_48_host(p, &pSumR, &pSumG, &pSumB);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    sumR += static_cast<Rpp32u>(srcPtrTemp[0]);
                    sumG += static_cast<Rpp32u>(srcPtrTemp[1]);
                    sumB += static_cast<Rpp32u>(srcPtrTemp[2]);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_store_si256((__m256i *)sumAvxR, pSumR);
            _mm256_store_si256((__m256i *)sumAvxG, pSumG);
            _mm256_store_si256((__m256i *)sumAvxB, pSumB);
            sumR += (sumAvxR[0] + sumAvxR[1] + sumAvxR[2] + sumAvxR[3] + sumAvxR[4] + sumAvxR[5] + sumAvxR[6] + sumAvxR[7]);
            sumG += (sumAvxG[0] + sumAvxG[1] + sumAvxG[2] + sumAvxG[3] + sumAvxG[4] + sumAvxG[5] + sumAvxG[6] + sumAvxG[7]);
            sumB += (sumAvxB[0] + sumAvxB[1] + sumAvxB[2] + sumAvxB[3] + sumAvxB[4] + sumAvxB[5] + sumAvxB[6] + sumAvxB[7]);
#endif
            sum = static_cast<Rpp64u>(sumR) + static_cast<Rpp64u>(sumG) + static_cast<Rpp64u>(sumB);
            mean = (static_cast<Rpp64f>(sum) / (totalPixelsPerChannel * 3));
            meanR = (static_cast<Rpp32f>(sumR) / totalPixelsPerChannel);
            meanG = (static_cast<Rpp32f>(sumG) / totalPixelsPerChannel);
            meanB = (static_cast<Rpp32f>(sumB) / totalPixelsPerChannel);
            tensorMeanArr[idx] = meanR;
            tensorMeanArr[idx + 1] = meanG;
            tensorMeanArr[idx + 2] = meanB;
            tensorMeanArr[idx + 3] = mean;
        }
    }

    return RPP_SUCCESS;
}

RppStatus tensor_mean_f32_f32_host(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *tensorMeanArr,
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

        // Tensor Mean without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            vectorIncrement = 8;
            Rpp32f mean = 0.0;
            Rpp64f sum = 0.0;
            Rpp64f sumAvx[4] = {0.0};

            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pSum = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_f32_to_f64_avx, srcPtrTemp, p1);
                    compute_sum_8_host(p1, &pSum);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    sum += static_cast<Rpp64f>(*srcPtrTemp++);
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(sumAvx, pSum);
            sum += (sumAvx[0] + sumAvx[1] + sumAvx[2] + sumAvx[3]);
#endif
            mean = static_cast<Rpp32f>(sum / totalPixelsPerChannel);
            tensorMeanArr[batchCount] = mean;
        }

        // Tensor Mean without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp32f mean, meanR = 0.0, meanG = 0.0, meanB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d pSumR = _mm256_setzero_pd();
            __m256d pSumG = _mm256_setzero_pd();
            __m256d pSumB = _mm256_setzero_pd();
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
                    compute_sum_24_host(p, &pSumR, &pSumG, &pSumB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sumR += static_cast<Rpp64f>(*srcPtrTempR++);
                    sumG += static_cast<Rpp64f>(*srcPtrTempG++);
                    sumB += static_cast<Rpp64f>(*srcPtrTempB++);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(sumAvxR, pSumR);
            _mm256_storeu_pd(sumAvxG, pSumG);
            _mm256_storeu_pd(sumAvxB, pSumB);
            sumR += (sumAvxR[0] + sumAvxR[1] + sumAvxR[2] + sumAvxR[3]);
            sumG += (sumAvxG[0] + sumAvxG[1] + sumAvxG[2] + sumAvxG[3]);
            sumB += (sumAvxB[0] + sumAvxB[1] + sumAvxB[2] + sumAvxB[3]);
#endif

            sum = sumR + sumG + sumB;
            mean = static_cast<Rpp32f>(sum / (totalPixelsPerChannel * 3));
            meanR = static_cast<Rpp32f>(sumR / totalPixelsPerChannel);
            meanG = static_cast<Rpp32f>(sumG / totalPixelsPerChannel);
            meanB = static_cast<Rpp32f>(sumB / totalPixelsPerChannel);
            tensorMeanArr[idx] = meanR;
            tensorMeanArr[idx + 1] = meanG;
            tensorMeanArr[idx + 2] = meanB;
            tensorMeanArr[idx + 3] = mean;
        }

        // Tensor Mean without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp32f mean, meanR = 0.0, meanG = 0.0, meanB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pSumR = _mm256_setzero_pd();
            __m256d pSumG = _mm256_setzero_pd();
            __m256d pSumB = _mm256_setzero_pd();
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
                    compute_sum_24_host(p, &pSumR, &pSumG, &pSumB);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    sumR += static_cast<Rpp64f>(srcPtrTemp[0]);
                    sumG += static_cast<Rpp64f>(srcPtrTemp[1]);
                    sumB += static_cast<Rpp64f>(srcPtrTemp[2]);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(sumAvxR, pSumR);
            _mm256_storeu_pd(sumAvxG, pSumG);
            _mm256_storeu_pd(sumAvxB, pSumB);
            sumR += (sumAvxR[0] + sumAvxR[1] + sumAvxR[2] + sumAvxR[3]);
            sumG += (sumAvxG[0] + sumAvxG[1] + sumAvxG[2] + sumAvxG[3]);
            sumB += (sumAvxB[0] + sumAvxB[1] + sumAvxB[2] + sumAvxB[3]);
#endif
            sum = sumR + sumG + sumB;
            mean = static_cast<Rpp32f>(sum / (totalPixelsPerChannel * 3));
            meanR = static_cast<Rpp32f>(sumR / totalPixelsPerChannel);
            meanG = static_cast<Rpp32f>(sumG / totalPixelsPerChannel);
            meanB = static_cast<Rpp32f>(sumB / totalPixelsPerChannel);
            tensorMeanArr[idx] = meanR;
            tensorMeanArr[idx + 1] = meanG;
            tensorMeanArr[idx + 2] = meanB;
            tensorMeanArr[idx + 3] = mean;
        }
    }

    return RPP_SUCCESS;
}

RppStatus tensor_mean_f16_f32_host(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *tensorMeanArr,
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

        // Tensor Mean without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            vectorIncrement = 8;
            Rpp32f mean = 0.0;
            Rpp64f sum = 0.0;
            Rpp64f sumAvx[4] = {0.0};

            Rpp16f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pSum = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = static_cast<Rpp32f>(srcPtrTemp[cnt]);
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_f32_to_f64_avx, srcPtrTemp_ps, p1);
                    compute_sum_8_host(p1, &pSum);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    sum += static_cast<Rpp64f>(*srcPtrTemp++);
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(sumAvx, pSum);
            sum += (sumAvx[0] + sumAvx[1] + sumAvx[2] + sumAvx[3]);
#endif
            mean = static_cast<Rpp32f>(sum / totalPixelsPerChannel);
            tensorMeanArr[batchCount] = mean;
        }

        // Tensor Mean without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp32f mean, meanR = 0.0, meanG = 0.0, meanB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d pSumR = _mm256_setzero_pd();
            __m256d pSumG = _mm256_setzero_pd();
            __m256d pSumB = _mm256_setzero_pd();
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
                    compute_sum_24_host(p, &pSumR, &pSumG, &pSumB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sumR += static_cast<Rpp64f>(*srcPtrTempR++);
                    sumG += static_cast<Rpp64f>(*srcPtrTempG++);
                    sumB += static_cast<Rpp64f>(*srcPtrTempB++);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(sumAvxR, pSumR);
            _mm256_storeu_pd(sumAvxG, pSumG);
            _mm256_storeu_pd(sumAvxB, pSumB);
            sumR += (sumAvxR[0] + sumAvxR[1] + sumAvxR[2] + sumAvxR[3]);
            sumG += (sumAvxG[0] + sumAvxG[1] + sumAvxG[2] + sumAvxG[3]);
            sumB += (sumAvxB[0] + sumAvxB[1] + sumAvxB[2] + sumAvxB[3]);
#endif
            sum = sumR + sumG + sumB;
            mean = static_cast<Rpp32f>(sum / (totalPixelsPerChannel * 3));
            meanR = static_cast<Rpp32f>(sumR / totalPixelsPerChannel);
            meanG = static_cast<Rpp32f>(sumG / totalPixelsPerChannel);
            meanB = static_cast<Rpp32f>(sumB / totalPixelsPerChannel);
            tensorMeanArr[idx] = meanR;
            tensorMeanArr[idx + 1] = meanG;
            tensorMeanArr[idx + 2] = meanB;
            tensorMeanArr[idx + 3] = mean;
        }

        // Tensor Mean without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp32f mean, meanR = 0.0, meanG = 0.0, meanB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp16f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pSumR = _mm256_setzero_pd();
            __m256d pSumG = _mm256_setzero_pd();
            __m256d pSumB = _mm256_setzero_pd();
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
                    compute_sum_24_host(p, &pSumR, &pSumG, &pSumB);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    sumR += static_cast<Rpp64f>(srcPtrTemp[0]);
                    sumG += static_cast<Rpp64f>(srcPtrTemp[1]);
                    sumB += static_cast<Rpp64f>(srcPtrTemp[2]);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_storeu_pd(sumAvxR, pSumR);
            _mm256_storeu_pd(sumAvxG, pSumG);
            _mm256_storeu_pd(sumAvxB, pSumB);
            sumR += (sumAvxR[0] + sumAvxR[1] + sumAvxR[2] + sumAvxR[3]);
            sumG += (sumAvxG[0] + sumAvxG[1] + sumAvxG[2] + sumAvxG[3]);
            sumB += (sumAvxB[0] + sumAvxB[1] + sumAvxB[2] + sumAvxB[3]);
#endif
            sum = sumR + sumG + sumB;
            mean = static_cast<Rpp32f>(sum / (totalPixelsPerChannel * 3));
            meanR = static_cast<Rpp32f>(sumR / totalPixelsPerChannel);
            meanG = static_cast<Rpp32f>(sumG / totalPixelsPerChannel);
            meanB = static_cast<Rpp32f>(sumB / totalPixelsPerChannel);
            tensorMeanArr[idx] = meanR;
            tensorMeanArr[idx + 1] = meanG;
            tensorMeanArr[idx + 2] = meanB;
            tensorMeanArr[idx + 3] = mean;
        }
    }

    return RPP_SUCCESS;
}

RppStatus tensor_mean_i8_f32_host(Rpp8s *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *tensorMeanArr,
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
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32f totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;
        int idx = batchCount * 4;

        // Tensor Mean without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~15;
            vectorIncrement = 16;
            Rpp32f mean = 0.0;
            Rpp32s sum = 0;
            Rpp32s sumAvx[8] = {0};

            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256i pSum = avx_px0;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1[2];
                    rpp_simd_load(rpp_load16_i8_to_i32_avx, srcPtrTemp, p1);
                    compute_sum_16_host(p1, &pSum);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    sum += static_cast<Rpp32s>(*srcPtrTemp++);
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_store_si256((__m256i *)sumAvx, pSum);
            sum += (sumAvx[0] + sumAvx[1] + sumAvx[2] + sumAvx[3] + sumAvx[4] + sumAvx[5] + sumAvx[6] + sumAvx[7]);
#endif
            mean = static_cast<Rpp32f>(sum)  / totalPixelsPerChannel;
            tensorMeanArr[batchCount] = mean;
        }

        // Tensor Mean without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64s sum;
            Rpp32s sumR = 0, sumG = 0, sumB = 0;
            Rpp32f mean, meanR = 0.0, meanG = 0.0, meanB = 0.0;
            Rpp32s sumAvxR[8] = {0};
            Rpp32s sumAvxG[8] = {0};
            Rpp32s sumAvxB[8] = {0};

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256i pSumR = avx_px0;
            __m256i pSumG = avx_px0;
            __m256i pSumB = avx_px0;
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
                    __m256i p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_i32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_sum_48_host(p, &pSumR, &pSumG, &pSumB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sumR += static_cast<Rpp32s>(*srcPtrTempR++);
                    sumG += static_cast<Rpp32s>(*srcPtrTempG++);
                    sumB += static_cast<Rpp32s>(*srcPtrTempB++);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_store_si256((__m256i *)sumAvxR, pSumR);
            _mm256_store_si256((__m256i *)sumAvxG, pSumG);
            _mm256_store_si256((__m256i *)sumAvxB, pSumB);
            sumR += (sumAvxR[0] + sumAvxR[1] + sumAvxR[2] + sumAvxR[3] + sumAvxR[4] + sumAvxR[5] + sumAvxR[6] + sumAvxR[7]);
            sumG += (sumAvxG[0] + sumAvxG[1] + sumAvxG[2] + sumAvxG[3] + sumAvxG[4] + sumAvxG[5] + sumAvxG[6] + sumAvxG[7]);
            sumB += (sumAvxB[0] + sumAvxB[1] + sumAvxB[2] + sumAvxB[3] + sumAvxB[4] + sumAvxB[5] + sumAvxB[6] + sumAvxB[7]);
#endif

            sum = static_cast<Rpp64u>(sum) + static_cast<Rpp64u>(sumG) + static_cast<Rpp64u>(sumB);
            mean = (static_cast<Rpp64f>(sum) / (totalPixelsPerChannel * 3));
            meanR = (static_cast<Rpp32f>(sumR) / totalPixelsPerChannel);
            meanG = (static_cast<Rpp32f>(sumG) / totalPixelsPerChannel);
            meanB = (static_cast<Rpp32f>(sumB) / totalPixelsPerChannel);
            tensorMeanArr[idx] = meanR;
            tensorMeanArr[idx + 1] = meanG;
            tensorMeanArr[idx + 2] = meanB;
            tensorMeanArr[idx + 3] = mean;
        }

        // Tensor Mean without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64s sum;
            Rpp32s sumR = 0, sumG = 0, sumB = 0;
            Rpp32f mean, meanR = 0.0, meanG = 0.0, meanB = 0.0;
            Rpp32s sumAvxR[8] = {0};
            Rpp32s sumAvxG[8] = {0};
            Rpp32s sumAvxB[8] = {0};

            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256i pSumR = avx_px0;
            __m256i pSumG = avx_px0;
            __m256i pSumB = avx_px0;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_i32pln3_avx, srcPtrTemp, p);
                    compute_sum_48_host(p, &pSumR, &pSumG, &pSumB);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    sumR += static_cast<Rpp32s>(srcPtrTemp[0]);
                    sumG += static_cast<Rpp32s>(srcPtrTemp[1]);
                    sumB += static_cast<Rpp32s>(srcPtrTemp[2]);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            _mm256_store_si256((__m256i *)sumAvxR, pSumR);
            _mm256_store_si256((__m256i *)sumAvxG, pSumG);
            _mm256_store_si256((__m256i *)sumAvxB, pSumB);
            sumR += (sumAvxR[0] + sumAvxR[1] + sumAvxR[2] + sumAvxR[3] + sumAvxR[4] + sumAvxR[5] + sumAvxR[6] + sumAvxR[7]);
            sumG += (sumAvxG[0] + sumAvxG[1] + sumAvxG[2] + sumAvxG[3] + sumAvxG[4] + sumAvxG[5] + sumAvxG[6] + sumAvxG[7]);
            sumB += (sumAvxB[0] + sumAvxB[1] + sumAvxB[2] + sumAvxB[3] + sumAvxB[4] + sumAvxB[5] + sumAvxB[6] + sumAvxB[7]);
#endif
            sum = static_cast<Rpp64u>(sumR) + static_cast<Rpp64u>(sumG) + static_cast<Rpp64u>(sumB);
            mean = (static_cast<Rpp64f>(sum) / (totalPixelsPerChannel * 3));
            meanR = (static_cast<Rpp32f>(sumR) / totalPixelsPerChannel);
            meanG = (static_cast<Rpp32f>(sumG) / totalPixelsPerChannel);
            meanB = (static_cast<Rpp32f>(sumB) / totalPixelsPerChannel);
            tensorMeanArr[idx] = meanR;
            tensorMeanArr[idx + 1] = meanG;
            tensorMeanArr[idx + 2] = meanB;
            tensorMeanArr[idx + 3] = mean;
        }
    }

    return RPP_SUCCESS;
}
