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

using namespace std;

inline void increment_ndim_ptr(Rpp32f **dstPtr, Rpp32u tensorDims, Rpp32u increment)
{
    for(int i = 0; i < tensorDims; i++)
        dstPtr[i] += increment;
}

inline void rpp_store16_f32_f32_channelwise(Rpp32f **dstPtr, __m128 *p)
{
    _mm_storeu_ps(dstPtr[0], p[0]);
    _mm_storeu_ps(dstPtr[1], p[1]);
    _mm_storeu_ps(dstPtr[2], p[2]);
    _mm_storeu_ps(dstPtr[3], p[3]);
}

inline void compute_transpose4x8_avx(__m256 *pSrc, __m128 *pDst)
{
    __m256 tmp0, tmp1, tmp2, tmp3;
    tmp0 = _mm256_shuffle_ps(pSrc[0], pSrc[1], 0x44);   /* shuffle to get [P01|P02|P09|P10|P05|P06|P13|P14] */
    tmp2 = _mm256_shuffle_ps(pSrc[0], pSrc[1], 0xEE);   /* shuffle to get [P03|P04|P11|P12|P07|P08|P15|P16] */
    tmp1 = _mm256_shuffle_ps(pSrc[2], pSrc[3], 0x44);   /* shuffle to get [P17|P18|P25|P26|P21|P22|P29|P30] */
    tmp3 = _mm256_shuffle_ps(pSrc[2], pSrc[3], 0xEE);   /* shuffle to get [P19|P20|P27|P28|P23|P24|P31|P32] */
    pSrc[0] = _mm256_shuffle_ps(tmp0, tmp1, 0x88);  /* shuffle to get [P01|P09|P17|P25|P05|P13|P21|P29] */
    pSrc[1] = _mm256_shuffle_ps(tmp0, tmp1, 0xDD);  /* shuffle to get [P02|P10|P18|P26|P06|P14|P22|P30] */
    pSrc[2] = _mm256_shuffle_ps(tmp2, tmp3, 0x88);  /* shuffle to get [P03|P11|P19|P27|P07|P15|P23|P31] */
    pSrc[3] = _mm256_shuffle_ps(tmp2, tmp3, 0xDD);  /* shuffle to get [P04|P12|P20|P28|P08|P16|P24|P32] */

    pDst[0] = _mm256_castps256_ps128(pSrc[0]);  /* extract [P01|P09|P17|P25] */
    pDst[1] = _mm256_castps256_ps128(pSrc[1]);  /* extract [P02|P10|P18|P26] */
    pDst[2] = _mm256_castps256_ps128(pSrc[2]);  /* extract [P03|P11|P19|P27] */
    pDst[3] = _mm256_castps256_ps128(pSrc[3]);  /* extract [P04|P12|P20|P28] */
    pDst[4] = _mm256_extractf128_ps(pSrc[0], 1);    /* extract [P05|P13|P21|P29] */
    pDst[5] = _mm256_extractf128_ps(pSrc[1], 1);    /* extract [P06|P14|P22|P30] */
    pDst[6] = _mm256_extractf128_ps(pSrc[2], 1);    /* extract [P07|P15|P23|P31] */
    pDst[7] = _mm256_extractf128_ps(pSrc[3], 1);    /* extract [P08|P16|P24|P32] */
}

inline void compute_2d_pln1_transpose(Rpp32f *srcPtrTemp, Rpp32f *dstPtrTemp, Rpp32u height, Rpp32u width, Rpp32u srcRowStride, Rpp32u dstRowStride)
{
    Rpp32u alignedRows = height & ~3;
    Rpp32u alignedCols = width & ~7;
    Rpp32u vectorIncrement = 8;
    Rpp32u dstRowVectorStride = vectorIncrement * dstRowStride;

    Rpp32s i = 0;
    for(Rpp32s k = 0; i < alignedRows; i += 4, k++)
    {
        Rpp32f *srcPtrRow[4], *dstPtrRow[8];
        for(int j = 0; j < 4; j++)
            srcPtrRow[j] = srcPtrTemp + (i + j) * srcRowStride;
        for(int j = 0; j < 8; j++)
            dstPtrRow[j] = dstPtrTemp + j * dstRowStride + i;

        Rpp32u vectorLoopCount = 0;
#if __AVX2__
        for(; vectorLoopCount < alignedCols; vectorLoopCount += vectorIncrement)
        {
            __m256 pSrc[4];
            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow[0], &pSrc[0]);
            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow[1], &pSrc[1]);
            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow[2], &pSrc[2]);
            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow[3], &pSrc[3]);

            __m128 pDst[8];
            compute_transpose4x8_avx(pSrc, pDst);
            rpp_simd_store(rpp_store4_f32_to_f32, dstPtrRow[0], &pDst[0]);
            rpp_simd_store(rpp_store4_f32_to_f32, dstPtrRow[1], &pDst[1]);
            rpp_simd_store(rpp_store4_f32_to_f32, dstPtrRow[2], &pDst[2]);
            rpp_simd_store(rpp_store4_f32_to_f32, dstPtrRow[3], &pDst[3]);
            rpp_simd_store(rpp_store4_f32_to_f32, dstPtrRow[4], &pDst[4]);
            rpp_simd_store(rpp_store4_f32_to_f32, dstPtrRow[5], &pDst[5]);
            rpp_simd_store(rpp_store4_f32_to_f32, dstPtrRow[6], &pDst[6]);
            rpp_simd_store(rpp_store4_f32_to_f32, dstPtrRow[7], &pDst[7]);

            srcPtrRow[0] += vectorIncrement;
            srcPtrRow[1] += vectorIncrement;
            srcPtrRow[2] += vectorIncrement;
            srcPtrRow[3] += vectorIncrement;
            dstPtrRow[0] += dstRowVectorStride;
            dstPtrRow[1] += dstRowVectorStride;
            dstPtrRow[2] += dstRowVectorStride;
            dstPtrRow[3] += dstRowVectorStride;
            dstPtrRow[4] += dstRowVectorStride;
            dstPtrRow[5] += dstRowVectorStride;
            dstPtrRow[6] += dstRowVectorStride;
            dstPtrRow[7] += dstRowVectorStride;
        }
#endif
    }

    // handle remaining columns
    for(Rpp32s k = 0; k < alignedRows; k++)
    {
        Rpp32f *srcPtrRowTemp = srcPtrTemp + k * srcRowStride + alignedCols;
        Rpp32f *dstPtrRowTemp = dstPtrTemp + alignedCols * dstRowStride + k;
        for(Rpp32s j = alignedCols; j < width; j++)
        {
            *dstPtrRowTemp = *srcPtrRowTemp++;
            dstPtrRowTemp += dstRowStride;
        }
    }

    // handle remaining rows
    for( ; i < height; i++)
    {
        Rpp32f *srcPtrRowTemp = srcPtrTemp + i * srcRowStride;
        Rpp32f *dstPtrRowTemp = dstPtrTemp + i;
        for(Rpp32s j = 0; j < width; j++)
        {
            *dstPtrRowTemp = *srcPtrRowTemp;
            srcPtrRowTemp++;
            dstPtrRowTemp += dstRowStride;
        }
    }
}

template<typename T>
void transpose_generic_nd_recursive(T *dst, Rpp32u *dstStrides, T *src, Rpp32u *srcStrides, Rpp32u *dstShape, Rpp32u tensorDims)
{
    // exit case for recursion
    if (tensorDims == 0)
    {
        *dst = *src;
    }
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            transpose_generic_nd_recursive(dst, dstStrides + 1, src, srcStrides + 1, dstShape + 1, tensorDims - 1);
            dst += *dstStrides;
            src += *srcStrides;
        }
    }
}

template<typename T>
void transpose_generic_setup_and_run(T *srcPtrTemp, T *dstPtrTemp, Rpp32u *length, Rpp32u *perm, Rpp32u tensorDims)
{
    Rpp32u dstShape[RPPT_MAX_DIMS];
    Rpp32u srcStrides[RPPT_MAX_DIMS];
    Rpp32u dstStrides[RPPT_MAX_DIMS];

    // compute output shape
    for(Rpp32u i = 0; i < tensorDims; i++)
        dstShape[i] = length[perm[i]];

    // compute output strides
    compute_strides(dstStrides, dstShape, tensorDims);

    // compute input strides and update as per the permute order
    Rpp32u tempStrides[RPPT_MAX_DIMS];
    compute_strides(tempStrides, length, tensorDims);
    for(int i = 0; i < tensorDims; i++)
        srcStrides[i] = tempStrides[perm[i]];

    // perform transpose as per the permute order
    transpose_generic_nd_recursive(dstPtrTemp, dstStrides, srcPtrTemp, srcStrides, dstShape, tensorDims);
}

RppStatus transpose_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = dstGenericDescPtr->numDims - 1;  // exclude batchsize from input dims
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32f *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        // get the starting address of begin and length values from roiTensor
        Rpp32u *roi = roiTensor + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *perm = permTensor;

        bool copyInput = true;
        for(int i = 0; i < tensorDims; i++)
            copyInput *= (perm[i] == i);

        // do memcpy of input to output since output order is same as input order
        if(copyInput)
        {
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcGenericDescPtr->strides[0] * sizeof(Rpp32f)));
        }
        else
        {
            for(int i = 1; i < tensorDims; i++)
                srcPtrTemp += begin[i - 1] * srcGenericDescPtr->strides[i];

            if (tensorDims == 2 && perm[0] == 1 && perm[1] == 0)
            {
                // Optimized AVX version for 2D PLN1 inputs
                compute_2d_pln1_transpose(srcPtrTemp, dstPtrTemp, length[0], length[1], srcGenericDescPtr->strides[1], dstGenericDescPtr->strides[1]);
            }
            else if (tensorDims == 3)
            {
                // Optimized AVX version for 3D inputs of shape(x, y, 16) and permutation order (2, 0, 1) (usecases : Deepcam training)
                if(perm[0] == 2 && perm[1] == 0 && perm[2] == 1 && length[2] == 16)
                {
                    Rpp32u height = length[0];
                    Rpp32u width = length[1];
                    Rpp32u channels = 16;
                    Rpp32u bufferLength = width * channels;
                    Rpp32u alignedLength = bufferLength & ~63;
                    Rpp32u vectorIncrement = 64;
                    Rpp32u vectorIncrementPerChannel = 4;

                    // initialize pointers for 16 channel
                    Rpp32f *dstPtrChannel[16];
                    for(int i = 0; i < 16; i++)
                        dstPtrChannel[i] = dstPtrTemp + i * dstGenericDescPtr->strides[1];

                    // loop over rows
                    for(int i = 0; i < height; i++)
                    {
                        Rpp32f *srcPtrRow = srcPtrTemp;

                        // update temporary pointers for 16 channel
                        Rpp32f *dstPtrTempChannel[16];
                        for(int k = 0; k < 16; k++)
                            dstPtrTempChannel[k] = dstPtrChannel[k];

                        Rpp32u vectorLoopCount = 0;
#if __AVX2__
                        for( ; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 pSrc[8];
                            // load 64 values for source
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow, &pSrc[0]);
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow + 16, &pSrc[1]);
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow + 32, &pSrc[2]);
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow + 48, &pSrc[3]);
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow + 8, &pSrc[4]);
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow + 24, &pSrc[5]);
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow + 40, &pSrc[6]);
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRow + 56, &pSrc[7]);

                            __m128 pDst[16];
                            compute_transpose4x8_avx(&pSrc[0], &pDst[0]);
                            compute_transpose4x8_avx(&pSrc[4], &pDst[8]);

                            // store 4 values in output per channel
                            rpp_store16_f32_f32_channelwise(&dstPtrTempChannel[0], &pDst[0]);
                            rpp_store16_f32_f32_channelwise(&dstPtrTempChannel[4], &pDst[4]);
                            rpp_store16_f32_f32_channelwise(&dstPtrTempChannel[8], &pDst[8]);
                            rpp_store16_f32_f32_channelwise(&dstPtrTempChannel[12], &pDst[12]);

                            srcPtrRow += vectorIncrement;
                            increment_ndim_ptr(dstPtrTempChannel, 16, vectorIncrementPerChannel);
                        }
#endif
                        for( ; vectorLoopCount < bufferLength; vectorLoopCount += 16)
                        {
                            for(int k = 0; k < 16; k++)
                                *dstPtrTempChannel[k] = srcPtrRow[k];

                            srcPtrRow += 16;
                            increment_ndim_ptr(dstPtrTempChannel, 16, 1);
                        }
                        srcPtrTemp += srcGenericDescPtr->strides[1];
                        increment_ndim_ptr(dstPtrChannel, 16, dstGenericDescPtr->dims[3]);
                    }
                }
                // Optimized AVX version for 3D inputs and permutation order (1, 0, 2)
                else if(perm[0] == 1 && perm[1] == 0 && perm[2] == 2)
                {
                    Rpp32f *srcPtrRow = srcPtrTemp;
                    Rpp32f *dstPtrRow = dstPtrTemp;
                    Rpp32u height = length[0];
                    Rpp32u width = length[1];
                    Rpp32u channels = length[2];
                    Rpp32u copySizeInBytes = channels * sizeof(Rpp32f);
                    for(int i = 0; i < height; i++)
                    {
                        Rpp32f *srcPtrRowTemp = srcPtrRow;
                        Rpp32f *dstPtrRowTemp = dstPtrRow;
                        for(int j = 0; j < width; j++)
                        {
                            memcpy(dstPtrRowTemp, srcPtrRowTemp, copySizeInBytes);
                            srcPtrRowTemp += srcGenericDescPtr->strides[2];
                            dstPtrRowTemp += dstGenericDescPtr->strides[1];
                        }
                        srcPtrRow += srcGenericDescPtr->strides[1];
                        dstPtrRow += dstGenericDescPtr->strides[2];
                    }
                }
                // Optimized AVX version for 3D inputs and permutation order (0, 2, 1)
                else if(perm[0] == 0 && perm[1] == 2 && perm[2] == 1)
                {
                    Rpp32f *srcPtrRow = srcPtrTemp;
                    Rpp32f *dstPtrRow = dstPtrTemp;
                    for(int i = 0; i < length[0]; i++)
                    {
                        compute_2d_pln1_transpose(srcPtrTemp, dstPtrTemp, length[1], length[2], srcGenericDescPtr->strides[2], dstGenericDescPtr->strides[2]);

                        // increment src and dst pointers
                        srcPtrTemp += srcGenericDescPtr->strides[1];
                        dstPtrTemp += dstGenericDescPtr->strides[1];
                    }
                }
                else
                {
                    transpose_generic_setup_and_run(srcPtrTemp, dstPtrTemp, length, perm, tensorDims);
                }
            }
            else if (tensorDims == 4)
            {
                // Optimized AVX version for 4D inputs and permutation order (1, 2, 3, 0)
                Rpp32u vectorIncrement = 8;
                if(perm[0] == 1 && perm[1] == 2 && perm[2] == 3 && perm[3] == 0)
                {
                    Rpp32u bufferLength = length[perm[3]];
                    Rpp32u alignedLength = bufferLength & ~7;
                    Rpp32f *srcPtr0 = srcPtrTemp;
                    Rpp32f *dstPtr0 = dstPtrTemp;
                    Rpp32u stridesIncrement[8] = {0, srcGenericDescPtr->strides[1], 2 * srcGenericDescPtr->strides[1], 3 * srcGenericDescPtr->strides[1],
                                                  4 * srcGenericDescPtr->strides[1], 5 * srcGenericDescPtr->strides[1], 6 * srcGenericDescPtr->strides[1], 7 * srcGenericDescPtr->strides[1]};
                    Rpp32u srcIncrement = vectorIncrement * srcGenericDescPtr->strides[1];
                    for(int i = 0; i < length[perm[0]]; i++)
                    {
                        Rpp32f *srcPtr1 = srcPtr0;
                        Rpp32f *dstPtr1 = dstPtr0;
                        for(int j = 0; j < length[perm[1]]; j++)
                        {
                            Rpp32f *srcPtr2 = srcPtr1;
                            Rpp32f *dstPtr2 = dstPtr1;
                            for(int k = 0; k < length[perm[2]]; k++)
                            {
                                Rpp32f *srcPtr3 = srcPtr2;
                                Rpp32f *dstPtr3 = dstPtr2;

                                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                                for( ; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                                {
                                    __m256 pSrc = _mm256_setr_ps(srcPtr3[stridesIncrement[0]], srcPtr3[stridesIncrement[1]], srcPtr3[stridesIncrement[2]], srcPtr3[stridesIncrement[3]],
                                                                 srcPtr3[stridesIncrement[4]], srcPtr3[stridesIncrement[5]], srcPtr3[stridesIncrement[6]], srcPtr3[stridesIncrement[7]]);
                                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtr3, &pSrc);
                                    srcPtr3 += srcIncrement;
                                    dstPtr3 += vectorIncrement;
                                }
#endif
                                for( ; vectorLoopCount < bufferLength; vectorLoopCount++)
                                {
                                    *dstPtr3++ = *srcPtr3;
                                    srcPtr3 += srcGenericDescPtr->strides[1];
                                }
                                srcPtr2 += 1;
                                dstPtr2 += dstGenericDescPtr->strides[3];
                            }
                            srcPtr1 += srcGenericDescPtr->strides[3];
                            dstPtr1 += dstGenericDescPtr->strides[2];
                        }
                        srcPtr0 += srcGenericDescPtr->strides[2];
                        dstPtr0 += dstGenericDescPtr->strides[1];
                    }
                }
                else
                {
                    transpose_generic_setup_and_run(srcPtrTemp, dstPtrTemp, length, perm, tensorDims);
                }
            }
            else
            {
                transpose_generic_setup_and_run(srcPtrTemp, dstPtrTemp, length, perm, tensorDims);
            }
        }
    }

    return RPP_SUCCESS;
}

template<typename T>
RppStatus transpose_generic_host_tensor(T *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = dstGenericDescPtr->numDims - 1;  // exclude batchsize from input dims
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        T *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        // get the starting address of begin and length values from roiTensor
        Rpp32u *roi = roiTensor + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *perm = permTensor;

        bool copyInput = true;
        for(int i = 0; i < tensorDims; i++)
            copyInput *= (perm[i] == i);

        // do memcpy of input to output since output order is same as input order
        if(copyInput)
        {
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcGenericDescPtr->strides[0] * sizeof(T)));
        }
        else
        {
            for(int i = 1; i < tensorDims; i++)
                srcPtrTemp += begin[i - 1] * srcGenericDescPtr->strides[i];
            transpose_generic_setup_and_run(srcPtrTemp, dstPtrTemp, length, perm, tensorDims);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus transpose_generic_host_tensor<Rpp8u>(Rpp8u*,
                                                        RpptGenericDescPtr,
                                                        Rpp8u*,
                                                        RpptGenericDescPtr,
                                                        Rpp32u*,
                                                        Rpp32u*,
                                                        rpp::Handle&);

template RppStatus transpose_generic_host_tensor<Rpp16f>(Rpp16f*,
                                                         RpptGenericDescPtr,
                                                         Rpp16f*,
                                                         RpptGenericDescPtr,
                                                         Rpp32u*,
                                                         Rpp32u*,
                                                         rpp::Handle&);

template RppStatus transpose_generic_host_tensor<Rpp8s>(Rpp8s*,
                                                        RpptGenericDescPtr,
                                                        Rpp8s*,
                                                        RpptGenericDescPtr,
                                                        Rpp32u*,
                                                        Rpp32u*,
                                                        rpp::Handle&);
