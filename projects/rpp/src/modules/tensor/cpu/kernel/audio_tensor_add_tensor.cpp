/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

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

RppStatus audio_tensor_add_tensor_host(Rpp32f *srcPtr1,
                                       Rpp32f *srcPtr2,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32s *srcLengthTensor,
                                       rpp::Handle& handle)
{
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtr1Temp = srcPtr1 + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];

        // Scalar per batch broadcasting: srcPtr2 has shape (batchSize, 1)
        // Each batch has a single scalar value that gets added to all elements
        Rpp32f scalarValue = srcPtr2[batchCount];

        Rpp32s vectorLoopCount = 0;
        Rpp32s vectorIncrement = 8;
        Rpp32s alignedLength = (bufferLength / 8) * 8;
#if __AVX2__
        __m256 pScalar = _mm256_set1_ps(scalarValue);
        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
        {
            __m256 pSrc = _mm256_loadu_ps(srcPtr1Temp);
            __m256 pDst = _mm256_add_ps(pSrc, pScalar);
            _mm256_storeu_ps(dstPtrTemp, pDst);
            srcPtr1Temp += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
#endif
        // Process remaining elements
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            *dstPtrTemp++ = *srcPtr1Temp++ + scalarValue;
    }

    return RPP_SUCCESS;
}
