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

RppStatus pre_emphasis_filter_host_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s *srcLengthTensor,
                                          Rpp32f *coeffTensor,
                                          Rpp32u borderType,
                                          rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        Rpp32f coeff = coeffTensor[batchCount];
        Rpp32f border = (borderType == RpptAudioBorderType::CLAMP) ? srcPtrTemp[0] :
                        (borderType == RpptAudioBorderType::REFLECT) ? srcPtrTemp[1] : 0;
        dstPtrTemp[0] = srcPtrTemp[0] - coeff * border;

        Rpp32s vectorIncrement = 8;
        Rpp32s alignedLength = (bufferLength / 8) * 8 - 8;
        __m256 pCoeff = _mm256_set1_ps(coeff);

        Rpp32s vectorLoopCount = 1;
        dstPtrTemp++;
        srcPtrTemp++;
        for(; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
        {
            __m256 pSrc[2];
            pSrc[0] = _mm256_loadu_ps(srcPtrTemp);
            pSrc[1] = _mm256_loadu_ps(srcPtrTemp - 1);
            pSrc[1] = _mm256_sub_ps(pSrc[0], _mm256_mul_ps(pSrc[1], pCoeff));
            _mm256_storeu_ps(dstPtrTemp, pSrc[1]);
            srcPtrTemp += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
        for(; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            *dstPtrTemp++ = *srcPtrTemp - coeff * (*(srcPtrTemp - 1));
            srcPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}
