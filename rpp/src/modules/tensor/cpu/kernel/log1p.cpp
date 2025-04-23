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

// 1 pixel log1p helper functions
// Also negative values are converted to positive by taking absolute of inputs
inline void compute_log1p(Rpp16s *src, Rpp32f *dst)
{
    *dst =  std::log1p(std::abs(*src));
}

// Computes ND log recursively
void log1p_recursive(Rpp16s *src, Rpp32u *srcStrides, Rpp32f *dst, Rpp32u *dstStrides, Rpp32u *dstShape, Rpp32u nDim)
{
    if (!nDim)
        compute_log1p(src, dst);
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            log1p_recursive(src, srcStrides + 1, dst, dstStrides + 1, dstShape + 1, nDim - 1);
            dst += *dstStrides;
            src += *srcStrides;
        }
    }
}

//log(1+x) or log1p(x) for input I16 and output F32
RppStatus log1p_i16_f32_host_tensor(Rpp16s *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = srcGenericDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        Rpp16s *srcPtr1 = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        Rpp32f *dstPtr1 = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        for (int i = 0; i < nDim; i++)
            srcPtr1 += begin[i] * srcGenericDescPtr->strides[i + 1];
        Rpp32u alignedLength;
        Rpp32u vectorIncrement = 16;
        if (nDim == 1)
        {
            alignedLength = length[0] & ~15;
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
            {
                __m256 p[2];

                rpp_simd_load(rpp_load16_i16_to_f32_abs_avx, srcPtr1, p);   // simd loads
                p[0] = _mm256_add_ps(p[0], avx_p1);
                p[1] = _mm256_add_ps(p[1], avx_p1);
                compute_log_16_host(p);
                rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtr1, p); // simd stores
                srcPtr1 += vectorIncrement;
                dstPtr1 += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < length[0]; vectorLoopCount++)
            {
                compute_log1p(srcPtr1, dstPtr1);
                srcPtr1++;
                dstPtr1++;
            }
        }
        else if (nDim == 2)
        {
            alignedLength = length[1] & ~15;
            for (int i = 0; i < length[0]; i++)
            {
                Rpp16s *srcPtrTemp = srcPtr1;
                Rpp32f *dstPtrTemp = dstPtr1;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[2];

                    rpp_simd_load(rpp_load16_i16_to_f32_abs_avx, srcPtrTemp, p);    // simd loads
                    p[0] = _mm256_add_ps(p[0], avx_p1);
                    p[1] = _mm256_add_ps(p[1], avx_p1);
                    compute_log_16_host(p);
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[1]; vectorLoopCount++)
                {
                    compute_log1p(srcPtrTemp, dstPtrTemp);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else if (nDim == 3)
        {
            alignedLength = (length[0] * length[1]) & ~15;
            for (int i = 0; i < length[2]; i++)
            {
                Rpp16s *srcPtrTemp = srcPtr1;
                Rpp32f *dstPtrTemp = dstPtr1;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[2];

                    rpp_simd_load(rpp_load16_i16_to_f32_abs_avx, srcPtrTemp, p);    // simd loads
                    p[0] = _mm256_add_ps(p[0], avx_p1);
                    p[1] = _mm256_add_ps(p[1], avx_p1);
                    compute_log_16_host(p);  // log compute
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[0] * length[1]; vectorLoopCount++)
                {
                    compute_log1p(srcPtrTemp, dstPtrTemp);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtr1 += (length[0] * length[1]);
                dstPtr1 += (length[0] * length[1]);
            }
        }
        else if (nDim == 4)
        {
            alignedLength = (length[0] * length[1]) & ~15;
            for (int i = 0; i < length[3]; i++)
            {
                Rpp16s *srcPtrCol = srcPtr1;
                Rpp32f *dstPtrCol = dstPtr1;
                for (int j = 0; j < length[2]; j++)
                {
                    Rpp16s *srcPtrTemp = srcPtrCol;
                    Rpp32f *dstPtrTemp = dstPtrCol;
                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[2];

                        rpp_simd_load(rpp_load16_i16_to_f32_abs_avx, srcPtrTemp, p);    // simd loads
                        p[0] = _mm256_add_ps(p[0], avx_p1);
                        p[1] = _mm256_add_ps(p[1], avx_p1);
                        compute_log_16_host(p);  // log compute
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[0] * length[1]; vectorLoopCount++)
                    {
                        compute_log1p(srcPtrTemp, dstPtrTemp);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrCol += (length[0] * length[1]);
                    dstPtrCol += (length[0] * length[1]);
                }
                srcPtr1 += (length[0] * length[1] * length[2]);
                dstPtr1 += (length[0] * length[1] * length[2]);
            }

        }
         else
             log1p_recursive(srcPtr1, srcGenericDescPtr->strides, dstPtr1, dstGenericDescPtr->strides, length, nDim);
    }

    return RPP_SUCCESS;
}