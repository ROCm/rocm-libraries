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

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcDimsTensor,
                                  bool normalizeWeights,
                                  rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s samples = srcDimsTensor[batchCount * 2];
        Rpp32s channels = srcDimsTensor[batchCount * 2 + 1];
        bool flagAVX = 0;

        if(channels == 1)
        {
            // No need of downmixing, do a direct memcpy
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(samples * sizeof(Rpp32f)));
        }
        else
        {
            Rpp32f *weights = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + batchCount * channels;
            std::fill(weights, weights + channels, 1.f / channels);

            if(normalizeWeights)
            {
                // Compute sum of the weights
                Rpp32f sum = 0.0;
                for(int i = 0; i < channels; i++)
                    sum += weights[i];

                // Normalize the weights
                Rpp32f invSum = 1.0 / sum;
                for(int i = 0; i < channels; i++)
                    weights[i] *= invSum;
            }

            Rpp32s channelIncrement = 4;
            Rpp32s alignedChannels = (channels / 4) * 4;
            if(channels > 7)
            {
                flagAVX = 1;
                channelIncrement = 8;
                alignedChannels = (channels / 8) * 8;
            }

            // use weights to downmix to mono
            for(int64_t dstIdx = 0; dstIdx < samples; dstIdx++)
            {
                Rpp32s channelLoopCount = 0;
                // if number of channels are greater than or equal to 8, use AVX implementation
                if(flagAVX)
                {
                    __m256 pDst = avx_p0;
                    for(; channelLoopCount < alignedChannels; channelLoopCount += channelIncrement)
                    {
                        __m256 pSrc, pWeights;
                        pWeights = _mm256_setr_ps(weights[channelLoopCount], weights[channelLoopCount + 1], weights[channelLoopCount + 2], weights[channelLoopCount + 3],
                                weights[channelLoopCount + 4], weights[channelLoopCount + 5], weights[channelLoopCount + 6], weights[channelLoopCount + 7]);
                        pSrc = _mm256_loadu_ps(srcPtrTemp);
                        pSrc = _mm256_mul_ps(pSrc, pWeights);
                        pDst = _mm256_add_ps(pDst, pSrc);
                        srcPtrTemp += channelIncrement;
                    }
                    dstPtrTemp[dstIdx] = rpp_hsum_ps(pDst);
                }
                else
                {
                    __m128 pDst = xmm_p0;
                    for(; channelLoopCount < alignedChannels; channelLoopCount += channelIncrement)
                    {
                        __m128 pSrc, pWeights;
                        pWeights = _mm_setr_ps(weights[channelLoopCount], weights[channelLoopCount + 1], weights[channelLoopCount + 2], weights[channelLoopCount + 3]);
                        pSrc = _mm_loadu_ps(srcPtrTemp);
                        pSrc = _mm_mul_ps(pSrc, pWeights);
                        pDst = _mm_add_ps(pDst, pSrc);
                        srcPtrTemp += channelIncrement;
                    }
                    dstPtrTemp[dstIdx] = rpp_hsum_ps(pDst);
                }
                for(; channelLoopCount < channels; channelLoopCount++)
                    dstPtrTemp[dstIdx] += ((*srcPtrTemp++) * weights[channelLoopCount]);
            }
        }
    }

    return RPP_SUCCESS;
}
