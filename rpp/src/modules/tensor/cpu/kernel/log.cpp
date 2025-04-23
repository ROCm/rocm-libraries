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

// 1 pixel log helper functions
// NOTE: log(0) leads to undefined thus using nextafter() to avoid this result
//       Also negative values are converted to positive by taking absolute of inputs
inline void compute_log(Rpp8u *src, Rpp32f *dst) { *dst = (!*src) ? std::log(std::nextafter(0.0f, 1.0f)) : std::log(*src); }
inline void compute_log(Rpp8s *src, Rpp32f *dst) { *dst = (!*src) ? std::log(std::nextafter(0.0f, 1.0f)) : std::log(*src + 128); }
inline void compute_log(Rpp16f *src, Rpp16f *dst) { *dst = (!*src) ? log(std::nextafter(0.0f, 1.0f)) : log(abs(*src)); }
inline void compute_log(Rpp32f *src, Rpp32f *dst) { *dst = (!*src) ? std::log(std::nextafter(0.0f, 1.0f)) : std::log(abs(*src)); }

// Computes ND log recursively
template<typename T1, typename T2>
void log_recursive(T1 *src, Rpp32u *srcStrides, T2 *dst, Rpp32u *dstStrides, Rpp32u *dstShape, Rpp32u nDim)
{
    if (!nDim)
        compute_log(src, dst);
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            log_recursive(src, srcStrides + 1, dst, dstStrides + 1, dstShape + 1, nDim - 1);
            dst += *(dstStrides + 1);
            src += *(srcStrides + 1);
        }
    }
}

RppStatus log_generic_host_tensor(Rpp8u *srcPtr,
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
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        Rpp8u *srcPtr1 = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        Rpp32f *dstPtr1 = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        for(int i = 0; i < nDim; i++)
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

                rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr1, p);    // simd loads
                compute_log_16_host(p);  // log compute
                rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtr1, p);    // simd stores
                srcPtr1 += vectorIncrement;
                dstPtr1 += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < length[0]; vectorLoopCount++)
            {
                compute_log(srcPtr1, dstPtr1);
                srcPtr1++;
                dstPtr1++;
            }
        }
        else if(nDim == 2)
        {
            alignedLength = length[1] & ~15;
            for(int i = 0; i < length[0]; i++)
            {
                Rpp8u *srcPtrTemp = srcPtr1;
                Rpp32f *dstPtrTemp = dstPtr1;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[2];

                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                    compute_log_16_host(p);  // log compute
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[1]; vectorLoopCount++)
                {
                    compute_log(srcPtrTemp, dstPtrTemp);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else if(nDim == 3)
        {
            alignedLength = length[2] & ~15;
            for(int i = 0; i < length[0]; i++)
            {
                Rpp8u *srcPtrRow = srcPtr1;
                Rpp32f *dstPtrRow = dstPtr1;

                for(int j = 0; j < length[1]; j++)
                {
                    Rpp8u *srcPtrTemp = srcPtrRow;
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[2];

                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_log_16_host(p);  // log compute
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[2]; vectorLoopCount++)
                    {
                        compute_log(srcPtrTemp, dstPtrTemp);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcGenericDescPtr->strides[2];
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else
            log_recursive(srcPtr1, srcGenericDescPtr->strides, dstPtr1, dstGenericDescPtr->strides, length, nDim);
    }

    return RPP_SUCCESS;
}

RppStatus log_generic_host_tensor(Rpp8s *srcPtr,
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
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        Rpp8s *srcPtr1 = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        Rpp32f *dstPtr1 = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        for(int i = 0; i < nDim; i++)
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

                rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr1, p);    // simd loads
                compute_log_16_host(p);  // log compute
                rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtr1, p);    // simd stores
                srcPtr1 += vectorIncrement;
                dstPtr1 += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < length[0]; vectorLoopCount++)
            {
                compute_log(srcPtr1, dstPtr1);
                srcPtr1++;
                dstPtr1++;
            }
        }
        else if(nDim == 2)
        {
            alignedLength = length[1] & ~15;
            for(int i = 0; i < length[0]; i++)
            {
                Rpp8s *srcPtrTemp = srcPtr1;
                Rpp32f *dstPtrTemp = dstPtr1;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[2];

                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                    compute_log_16_host(p);  // log compute
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[1]; vectorLoopCount++)
                {
                    compute_log(srcPtrTemp, dstPtrTemp);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else if(nDim == 3)
        {
            alignedLength = length[2] & ~15;
            for(int i = 0; i < length[0]; i++)
            {
                Rpp8s *srcPtrRow = srcPtr1;
                Rpp32f *dstPtrRow = dstPtr1;

                for(int j = 0; j < length[1]; j++)
                {
                    Rpp8s *srcPtrTemp = srcPtrRow;
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[2];

                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_log_16_host(p);  // log compute
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[2]; vectorLoopCount++)
                    {
                        compute_log(srcPtrTemp, dstPtrTemp);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcGenericDescPtr->strides[2];
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else
            log_recursive(srcPtr1, srcGenericDescPtr->strides, dstPtr1, dstGenericDescPtr->strides, length, nDim);
    }

    return RPP_SUCCESS;
}

RppStatus log_generic_host_tensor(Rpp32f *srcPtr,
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
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        Rpp32f *srcPtr1 = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        Rpp32f *dstPtr1 = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        for(int i = 0; i < nDim; i++)
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

                rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtr1, p);    // simd loads
                compute_log_16_host(p);  // log compute
                rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtr1, p);    // simd stores
                srcPtr1 += vectorIncrement;
                dstPtr1 += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < length[0]; vectorLoopCount++)
            {
                compute_log(srcPtr1, dstPtr1);
                srcPtr1++;
                dstPtr1++;
            }
        }
        else if(nDim == 2)
        {
            alignedLength = length[1] & ~15;
            for(int i = 0; i < length[0]; i++)
            {
                Rpp32f *srcPtrTemp = srcPtr1;
                Rpp32f *dstPtrTemp = dstPtr1;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[2];

                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                    compute_log_16_host(p);  // log compute
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[1]; vectorLoopCount++)
                {
                    compute_log(srcPtrTemp, dstPtrTemp);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else if(nDim == 3)
        {
            alignedLength = length[2] & ~15;
            for(int i = 0; i < length[0]; i++)
            {
                Rpp32f *srcPtrRow = srcPtr1;
                Rpp32f *dstPtrRow = dstPtr1;

                for(int j = 0; j < length[1]; j++)
                {
                    Rpp32f *srcPtrTemp = srcPtrRow;
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[2];

                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_log_16_host(p);  // log compute
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[2]; vectorLoopCount++)
                    {
                        compute_log(srcPtrTemp, dstPtrTemp);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcGenericDescPtr->strides[2];
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else
            log_recursive(srcPtr1, srcGenericDescPtr->strides, dstPtr1, dstGenericDescPtr->strides, length, nDim);
    }

    return RPP_SUCCESS;
}

RppStatus log_generic_host_tensor(Rpp16f *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp16f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = srcGenericDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        Rpp16f *srcPtr1 = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        Rpp16f *dstPtr1 = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        for(int i = 0; i < nDim; i++)
            srcPtr1 += begin[i] * srcGenericDescPtr->strides[i + 1];
        Rpp32u alignedLength;
        Rpp32u vectorIncrement = 16;
        if (nDim == 1)
        {
            int vectorLoopCount = 0;
#if __AVX2__
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
            {
                Rpp32f srcPtrTemp_ps[16];
                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                    srcPtrTemp_ps[cnt] = static_cast<Rpp32f>(srcPtr1[cnt]);

                __m256 p[2];
                rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
                compute_log_16_host(p);  // log compute
                rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtr1, p);    // simd stores
                srcPtr1 += vectorIncrement;
                dstPtr1 += vectorIncrement;
            }
#endif
            for (; vectorLoopCount < length[0]; vectorLoopCount++)
            {
                compute_log(srcPtr1, dstPtr1);
                srcPtr1++;
                dstPtr1++;
            }
        }
        else if(nDim == 2)
        {
            alignedLength = length[1] & ~15;
            for(int i = 0; i < length[0]; i++)
            {
                Rpp16f *srcPtrTemp = srcPtr1;
                Rpp16f *dstPtrTemp = dstPtr1;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[16];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = static_cast<Rpp32f>(srcPtrTemp[cnt]);

                    __m256 p[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_log_16_host(p);  // log compute
                    rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < length[1]; vectorLoopCount++)
                {
                    compute_log(srcPtrTemp, dstPtrTemp);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else if(nDim == 3)
        {
            alignedLength = length[2] & ~15;
            for(int i = 0; i < length[0]; i++)
            {
                Rpp16f *srcPtrRow = srcPtr1;
                Rpp16f *dstPtrRow = dstPtr1;

                for(int j = 0; j < length[1]; j++)
                {
                    Rpp16f *srcPtrTemp = srcPtrRow;
                    Rpp16f *dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        Rpp32f srcPtrTemp_ps[16];
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            srcPtrTemp_ps[cnt] = static_cast<Rpp32f>(srcPtrTemp[cnt]);

                        __m256 p[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
                        compute_log_16_host(p);  // log compute
                        rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p);    // simd stores
                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < length[2]; vectorLoopCount++)
                    {
                        compute_log(srcPtrTemp, dstPtrTemp);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcGenericDescPtr->strides[2];
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtr1 += srcGenericDescPtr->strides[1];
                dstPtr1 += dstGenericDescPtr->strides[1];
            }
        }
        else
            log_recursive(srcPtr1, srcGenericDescPtr->strides, dstPtr1, dstGenericDescPtr->strides, length, nDim);
    }

    return RPP_SUCCESS;
}
