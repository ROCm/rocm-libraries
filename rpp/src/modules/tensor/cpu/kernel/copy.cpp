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

RppStatus copy_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    // Copy without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    if ((srcDescPtr->c == 1) || (srcDescPtr->layout == dstDescPtr->layout))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp8u *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
            memcpy(dstPtrImage, srcPtrImage, dstDescPtr->strides.nStride * sizeof(Rpp8u));
        }
    }

    // Copy with fused output-layout toggle (NHWC -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp8u *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

            Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp, px);    // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR++ = srcPtrTemp[0];
                    *dstPtrTempG++ = srcPtrTemp[1];
                    *dstPtrTempB++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    // Copy with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp8u *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

            Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, px);    // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);    // simd stores
                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTempR++;
                    dstPtrTemp[1] = *srcPtrTempG++;
                    dstPtrTemp[2] = *srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus copy_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    // Copy without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    if ((srcDescPtr->c == 1) || (srcDescPtr->layout == dstDescPtr->layout))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp32f *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
            memcpy(dstPtrImage, srcPtrImage, dstDescPtr->strides.nStride * sizeof(Rpp32f));
        }
    }

    // Copy with fused output-layout toggle (NHWC -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp32f *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

            Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR++ = srcPtrTemp[0];
                    *dstPtrTempG++ = srcPtrTemp[1];
                    *dstPtrTempB++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }

        }
    }

    // Copy with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp32f *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

            Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTempR++;
                    dstPtrTemp[1] = *srcPtrTempG++;
                    dstPtrTemp[2] = *srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

        }
    }

    return RPP_SUCCESS;
}

RppStatus copy_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    // Copy without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    if ((srcDescPtr->c == 1) || (srcDescPtr->layout == dstDescPtr->layout))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp16f *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
            memcpy(dstPtrImage, srcPtrImage, dstDescPtr->strides.nStride * sizeof(Rpp16f));
        }
    }

    // Copy with fused output-layout toggle (NHWC -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp16f *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

            Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
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
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR++ = srcPtrTemp[0];
                    *dstPtrTempG++ = srcPtrTemp[1];
                    *dstPtrTempB++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }

        }
    }

    // Copy with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp16f *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

            Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
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
                    dstPtrTemp[0] = *srcPtrTempR++;
                    dstPtrTemp[1] = *srcPtrTempG++;
                    dstPtrTemp[2] = *srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

        }
    }

    return RPP_SUCCESS;
}

RppStatus copy_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    // Copy without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
    if ((srcDescPtr->c == 1) || (srcDescPtr->layout == dstDescPtr->layout))
    {
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp8s *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
            memcpy(dstPtrImage, srcPtrImage, dstDescPtr->strides.nStride * sizeof(Rpp8s));
        }
    }

    // Copy with fused output-layout toggle (NHWC -> NCHW)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp8s *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

            Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp, px);    // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR++ = srcPtrTemp[0];
                    *dstPtrTempG++ = srcPtrTemp[1];
                    *dstPtrTempB++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    // Copy with fused output-layout toggle (NCHW -> NHWC)
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
        {
            Rpp8s *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
            dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

            Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, px);    // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);    // simd stores
                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTempR++;
                    dstPtrTemp[1] = *srcPtrTempG++;
                    dstPtrTemp[2] = *srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
