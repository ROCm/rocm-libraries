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

RppStatus audio_tensor_mul_scalar_host(Rpp32f *srcPtr,
                                       Rpp32f scalarValue,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32s *srcLengthTensor,
                                       rpp::Handle& handle)
{
    // Broadcast the scalar value for SIMD operations
    __m256 pScalar = _mm256_set1_ps(scalarValue);

    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (Rpp32u batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        
        Rpp32s vectorLoopCount = 0;
        Rpp32s vectorIncrement = 8;
        Rpp32s alignedLength = (bufferLength / 8) * 8;
#if __AVX2__       
        // Vectorized multiplication using AVX2 (8 floats at a time)
        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
        {
            __m256 pSrc = _mm256_loadu_ps(srcPtrTemp);
            __m256 pDst = _mm256_mul_ps(pSrc, pScalar);
            _mm256_storeu_ps(dstPtrTemp, pDst);
            srcPtrTemp += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
#endif        
        // Handle remaining elements
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            *dstPtrTemp++ = *srcPtrTemp++ * scalarValue;
    }

    return RPP_SUCCESS;
}
