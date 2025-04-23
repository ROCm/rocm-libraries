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

RppStatus resample_host_tensor(Rpp32f *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *inRateTensor,
                               Rpp32f *outRateTensor,
                               Rpp32s *srcDimsTensor,
                               RpptResamplingWindow &window,
                               rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f inRate = inRateTensor[batchCount];
        Rpp32f outRate = outRateTensor[batchCount];
        Rpp32s srcLength = srcDimsTensor[batchCount * 2];
        Rpp32s numChannels = srcDimsTensor[batchCount * 2 + 1];
        if(outRate == inRate)
        {
            // No need of Resampling, do a direct memcpy
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcLength * numChannels * sizeof(Rpp32f)));
        }
        else
        {
            Rpp32s outEnd = std::ceil(srcLength * outRate / inRate);
            Rpp32s inPos = 0;
            Rpp32s block = 1 << 8;
            Rpp64f scale = static_cast<Rpp64f>(inRate) / outRate;
            Rpp32f fscale = scale;
            if(numChannels == 1)
            {
                for (int outBlock = 0; outBlock < outEnd; outBlock += block)
                {
                    Rpp32s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp32s inBlockRounded = static_cast<int>(inBlockRaw);
                    Rpp32f inPos = inBlockRaw - inBlockRounded;
                    const Rpp32f *inBlockPtr = srcPtrTemp + inBlockRounded;
                    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
                    {
                        Rpp32s loc0, loc1;
                        window.input_range(inPos, &loc0, &loc1);
                        if (loc0 + inBlockRounded < 0)
                            loc0 = -inBlockRounded;
                        if (loc1 + inBlockRounded > srcLength)
                            loc1 = srcLength - inBlockRounded;
                        Rpp32s locInWindow = loc0;
                        Rpp32f locBegin = locInWindow - inPos;
                        __m128 pLocInWindow = _mm_add_ps(_mm_set1_ps(locBegin), xmm_pDstLocInit);

                        Rpp32f accum = 0.0f;
                        __m128 pAccum = xmm_p0;
                        for (; locInWindow + 3 < loc1; locInWindow += 4)
                        {
                            __m128 w4 = window(pLocInWindow);
                            pAccum = _mm_add_ps(pAccum, _mm_mul_ps(_mm_loadu_ps(inBlockPtr + locInWindow), w4));
                            pLocInWindow = _mm_add_ps(pLocInWindow, xmm_p4);
                        }
                        // sum all 4 values in the pAccum vector and store in accum
                        pAccum = _mm_add_ps(pAccum, _mm_shuffle_ps(pAccum, pAccum, _MM_SHUFFLE(1, 0, 3, 2)));
                        pAccum = _mm_add_ps(pAccum, _mm_shuffle_ps(pAccum, pAccum, _MM_SHUFFLE(0, 1, 0, 1)));
                        accum = _mm_cvtss_f32(pAccum);

                        Rpp32f x = locInWindow - inPos;
                        for (; locInWindow < loc1; locInWindow++, x++) {
                            Rpp32f w = window(x);
                            accum += inBlockPtr[locInWindow] * w;
                        }
                        dstPtrTemp[outPos] = accum;
                    }
                }
            }
            else
            {
                std::vector<Rpp32f> tempBuf;
                tempBuf.resize(numChannels);
                for (int outBlock = 0; outBlock < outEnd; outBlock += block)
                {
                    Rpp32s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp32s inBlockRounded = static_cast<int>(inBlockRaw);
                    Rpp32f inPos = inBlockRaw - inBlockRounded;
                    const Rpp32f *inBlockPtr = srcPtrTemp + inBlockRounded * numChannels;
                    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
                    {
                        Rpp32s loc0, loc1;
                        window.input_range(inPos, &loc0, &loc1);
                        if (loc0 + inBlockRounded < 0)
                            loc0 = -inBlockRounded;
                        if (loc1 + inBlockRounded > srcLength)
                            loc1 = srcLength - inBlockRounded;

                        std::fill(tempBuf.begin(), tempBuf.end(), 0.0f);
                        Rpp32f locInWindow = loc0 - inPos;
                        Rpp32s ofs0 = loc0 * numChannels;
                        Rpp32s ofs1 = loc1 * numChannels;
                        for (int inOfs = ofs0; inOfs < ofs1; inOfs += numChannels, locInWindow++)
                        {
                            Rpp32f w = window(locInWindow);
                            for (int c = 0; c < numChannels; c++)
                                tempBuf[c] += inBlockPtr[inOfs + c] * w;
                        }

                        Rpp32s dstLoc = outPos * numChannels;
                        for (int c = 0; c < numChannels; c++)
                            dstPtrTemp[dstLoc + c] = tempBuf[c];
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}
