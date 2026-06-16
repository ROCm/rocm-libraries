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

// Adjust permutation tensor for NHWC - NCHW layout conversion
inline void adjust_perm_tensor_pkd3_to_pln3(Rpp32u* permutationOrder) {
    Rpp32u count = 0;
    for (Rpp32u i = 0; i < 3; i++) {
        if (permutationOrder[i] != i) count++;
    }

    if (count == 3) {
        Rpp32u permutationOrderTemp[3];
        permutationOrderTemp[0] = permutationOrder[permutationOrder[0]];
        permutationOrderTemp[1] = permutationOrder[permutationOrder[1]];
        permutationOrderTemp[2] = permutationOrder[permutationOrder[2]];
        permutationOrder[0] = permutationOrderTemp[0];
        permutationOrder[1] = permutationOrderTemp[1];
        permutationOrder[2] = permutationOrderTemp[2];
    }
}

RppStatus channel_permute_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                            RpptDescPtr dstDescPtr, Rpp32u* permutationTensor,
                                            RppLayoutParams layoutParams, rpp::Handle& handle) {
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;

        // Load the 3-channel permutation order (e.g., {0, 2, 1} for R-B-G) for the current image in
        // the batch
        Rpp32u* permutationOrder = &permutationTensor[batchCount * 3];

        // Channel Permute with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            adjust_perm_tensor_pkd3_to_pln3(permutationOrder);
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp8u* dstPtrTemp[3] = {dstPtrTempR, dstPtrTempG, dstPtrTempB};
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256i p[6];
                    rpp_simd_load(rpp_load96_u8pkd3_to_u8pln3, srcPtrTemp, p);  // simd loads
                    rpp_simd_store(rpp_store96_u8pln3_to_u8pln3, dstPtrTemp[permutationOrder[0]],
                                   dstPtrTemp[permutationOrder[1]], dstPtrTemp[permutationOrder[2]],
                                   p);  // simd stores with channel permute
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp[0] += vectorIncrementPerChannel;
                    dstPtrTemp[1] += vectorIncrementPerChannel;
                    dstPtrTemp[2] += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    *dstPtrTemp[permutationOrder[0]]++ = srcPtrTemp[0];
                    *dstPtrTemp[permutationOrder[1]]++ = srcPtrTemp[1];
                    *dstPtrTemp[permutationOrder[2]]++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                Rpp8u* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256i p[6];
                    rpp_simd_load(rpp_load96_u8_avx, srcPtrTemp[permutationOrder[0]],
                                  srcPtrTemp[permutationOrder[1]], srcPtrTemp[permutationOrder[2]],
                                  p);  // simd loads with channel permute
                    rpp_simd_store(rpp_store96_u8pln3_to_u8pkd3, dstPtrTemp, p);  // simd stores
                    srcPtrTemp[permutationOrder[0]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[1]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[2]] += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    dstPtrTemp[0] = *srcPtrTemp[permutationOrder[0]]++;
                    dstPtrTemp[1] = *srcPtrTemp[permutationOrder[1]]++;
                    dstPtrTemp[2] = *srcPtrTemp[permutationOrder[2]]++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256i p[3], pPermute[3];
                    rpp_simd_load(rpp_load96_u8pkd3_to_u8pln3, srcPtrTemp, p);  // simd loads
                    pPermute[0] = p[permutationOrder[0]];
                    pPermute[1] = p[permutationOrder[1]];
                    pPermute[2] = p[permutationOrder[2]];
                    rpp_simd_store(rpp_store96_u8pln3_to_u8pkd3, dstPtrTemp,
                                   pPermute);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    dstPtrTemp[0] = srcPtrTemp[permutationOrder[0]];
                    dstPtrTemp[1] = srcPtrTemp[permutationOrder[1]];
                    dstPtrTemp[2] = srcPtrTemp[permutationOrder[2]];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp8u* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256i p[6];
                    rpp_simd_load(rpp_load96_u8_avx, srcPtrTemp[permutationOrder[0]],
                                  srcPtrTemp[permutationOrder[1]], srcPtrTemp[permutationOrder[2]],
                                  p);  // simd loads with channel permute
                    rpp_simd_store(rpp_store96_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp[permutationOrder[0]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[1]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[2]] += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTempR++ = *srcPtrTemp[permutationOrder[0]]++;
                    *dstPtrTempG++ = *srcPtrTemp[permutationOrder[1]]++;
                    *dstPtrTempB++ = *srcPtrTemp[permutationOrder[2]]++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus channel_permute_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr,
                                              Rpp32f* dstPtr, RpptDescPtr dstDescPtr,
                                              Rpp32u* permutationTensor,
                                              RppLayoutParams layoutParams, rpp::Handle& handle) {
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Load the 3-channel permutation order (e.g., {0, 2, 1} for R-B-G) for the current image in
        // the batch
        Rpp32u* permutationOrder = &permutationTensor[batchCount * 3];

        // Channel Permute with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            adjust_perm_tensor_pkd3_to_pln3(permutationOrder);
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f* dstPtrTemp[3] = {dstPtrTempR, dstPtrTempG, dstPtrTempB};
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx,
                                   dstPtrTemp[permutationOrder[0]], dstPtrTemp[permutationOrder[1]],
                                   dstPtrTemp[permutationOrder[2]],
                                   p);  // simd stores with channel permute
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp[0] += vectorIncrementPerChannel;
                    dstPtrTemp[1] += vectorIncrementPerChannel;
                    dstPtrTemp[2] += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    *dstPtrTemp[permutationOrder[0]]++ = srcPtrTemp[0];
                    *dstPtrTemp[permutationOrder[1]]++ = srcPtrTemp[1];
                    *dstPtrTemp[permutationOrder[2]]++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                Rpp32f* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx,
                                  srcPtrTemp[permutationOrder[0]], srcPtrTemp[permutationOrder[1]],
                                  srcPtrTemp[permutationOrder[2]],
                                  p);  // simd loads with channel permute
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTemp[permutationOrder[0]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[1]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[2]] += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    dstPtrTemp[0] = *srcPtrTemp[permutationOrder[0]]++;
                    dstPtrTemp[1] = *srcPtrTemp[permutationOrder[1]]++;
                    dstPtrTemp[2] = *srcPtrTemp[permutationOrder[2]]++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32u alignedLength = ((bufferLength - 1) / 24) * 24;
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3], pPermute[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    pPermute[0] = p[permutationOrder[0]];  // channel permute
                    pPermute[1] = p[permutationOrder[1]];  // channel permute
                    pPermute[2] = p[permutationOrder[2]];  // channel permute
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp,
                                   pPermute);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    dstPtrTemp[0] = srcPtrTemp[permutationOrder[0]];
                    dstPtrTemp[1] = srcPtrTemp[permutationOrder[1]];
                    dstPtrTemp[2] = srcPtrTemp[permutationOrder[2]];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp32f* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx,
                                  srcPtrTemp[permutationOrder[0]], srcPtrTemp[permutationOrder[1]],
                                  srcPtrTemp[permutationOrder[2]],
                                  p);  // simd loads with channel permute
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp[permutationOrder[0]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[1]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[2]] += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTempR++ = *srcPtrTemp[permutationOrder[0]]++;
                    *dstPtrTempG++ = *srcPtrTemp[permutationOrder[1]]++;
                    *dstPtrTempB++ = *srcPtrTemp[permutationOrder[2]]++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus channel_permute_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr,
                                              Rpp16f* dstPtr, RpptDescPtr dstDescPtr,
                                              Rpp32u* permutationTensor,
                                              RppLayoutParams layoutParams, rpp::Handle& handle) {
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Load the 3-channel permutation order (e.g., {0, 2, 1} for R-B-G) for the current image in
        // the batch
        Rpp32u* permutationOrder = &permutationTensor[batchCount * 3];

        // Channel Permute with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            adjust_perm_tensor_pkd3_to_pln3(permutationOrder);
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp16f* dstPtrTemp[3] = {dstPtrTempR, dstPtrTempG, dstPtrTempB};
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx,
                                   dstPtrTemp[permutationOrder[0]], dstPtrTemp[permutationOrder[1]],
                                   dstPtrTemp[permutationOrder[2]],
                                   p);  // simd stores with channel permute
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp[0] += vectorIncrementPerChannel;
                    dstPtrTemp[1] += vectorIncrementPerChannel;
                    dstPtrTemp[2] += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    *dstPtrTemp[permutationOrder[0]]++ = srcPtrTemp[0];
                    *dstPtrTemp[permutationOrder[1]]++ = srcPtrTemp[1];
                    *dstPtrTemp[permutationOrder[2]]++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                Rpp16f* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx,
                                  srcPtrTemp[permutationOrder[0]], srcPtrTemp[permutationOrder[1]],
                                  srcPtrTemp[permutationOrder[2]],
                                  p);  // simd loads with channel permute
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTemp[permutationOrder[0]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[1]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[2]] += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    dstPtrTemp[0] = *srcPtrTemp[permutationOrder[0]]++;
                    dstPtrTemp[1] = *srcPtrTemp[permutationOrder[1]]++;
                    dstPtrTemp[2] = *srcPtrTemp[permutationOrder[2]]++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[3], pPermute[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    pPermute[0] = p[permutationOrder[0]];  // channel permute
                    pPermute[1] = p[permutationOrder[1]];  // channel permute
                    pPermute[2] = p[permutationOrder[2]];  // channel permute
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp,
                                   pPermute);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    dstPtrTemp[0] = srcPtrTemp[permutationOrder[0]];
                    dstPtrTemp[1] = srcPtrTemp[permutationOrder[1]];
                    dstPtrTemp[2] = srcPtrTemp[permutationOrder[2]];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp16f* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx,
                                  srcPtrTemp[permutationOrder[0]], srcPtrTemp[permutationOrder[1]],
                                  srcPtrTemp[permutationOrder[2]],
                                  p);  // simd loads with channel permute
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores

                    srcPtrTemp[permutationOrder[0]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[1]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[2]] += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTempR++ = *srcPtrTemp[permutationOrder[0]]++;
                    *dstPtrTempG++ = *srcPtrTemp[permutationOrder[1]]++;
                    *dstPtrTempB++ = *srcPtrTemp[permutationOrder[2]]++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus channel_permute_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                            RpptDescPtr dstDescPtr, Rpp32u* permutationTensor,
                                            RppLayoutParams layoutParams, rpp::Handle& handle) {
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Load the 3-channel permutation order (e.g., {0, 2, 1} for R-B-G) for the current image in
        // the batch
        Rpp32u* permutationOrder = &permutationTensor[batchCount * 3];

        // Channel Permute with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            adjust_perm_tensor_pkd3_to_pln3(permutationOrder);
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp8s* dstPtrTemp[3] = {dstPtrTempR, dstPtrTempG, dstPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx,
                                   dstPtrTemp[permutationOrder[0]], dstPtrTemp[permutationOrder[1]],
                                   dstPtrTemp[permutationOrder[2]],
                                   p);  // simd stores with channel permute
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp[0] += vectorIncrementPerChannel;
                    dstPtrTemp[1] += vectorIncrementPerChannel;
                    dstPtrTemp[2] += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    *dstPtrTemp[permutationOrder[0]]++ = srcPtrTemp[0];
                    *dstPtrTemp[permutationOrder[1]]++ = srcPtrTemp[1];
                    *dstPtrTemp[permutationOrder[2]]++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                Rpp8s* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTemp[permutationOrder[0]],
                                  srcPtrTemp[permutationOrder[1]], srcPtrTemp[permutationOrder[2]],
                                  p);  // simd loads with channel permute
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                   p);  // simd stores
                    srcPtrTemp[permutationOrder[0]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[1]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[2]] += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    dstPtrTemp[0] = *srcPtrTemp[permutationOrder[0]]++;
                    dstPtrTemp[1] = *srcPtrTemp[permutationOrder[1]]++;
                    dstPtrTemp[2] = *srcPtrTemp[permutationOrder[2]]++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32u permIdx[6] = {permutationOrder[0] * 2, permutationOrder[0] * 2 + 1,
                                 permutationOrder[1] * 2, permutationOrder[1] * 2 + 1,
                                 permutationOrder[2] * 2, permutationOrder[2] * 2 + 1};
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement) {
                    __m256 p[6], pPermute[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);  // simd loads
                    pPermute[0] = p[permIdx[0]];
                    pPermute[1] = p[permIdx[1]];
                    pPermute[2] = p[permIdx[2]];
                    pPermute[3] = p[permIdx[3]];
                    pPermute[4] = p[permIdx[4]];
                    pPermute[5] = p[permIdx[5]];
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp,
                                   pPermute);  // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3) {
                    dstPtrTemp[0] = srcPtrTemp[permutationOrder[0]];
                    dstPtrTemp[1] = srcPtrTemp[permutationOrder[1]];
                    dstPtrTemp[2] = srcPtrTemp[permutationOrder[2]];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Permute without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < srcDescPtr->h; i++) {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG,
                    *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp8s* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTemp[permutationOrder[0]],
                                  srcPtrTemp[permutationOrder[1]], srcPtrTemp[permutationOrder[2]],
                                  p);  // simd loads with channel permute
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, p);  // simd stores
                    srcPtrTemp[permutationOrder[0]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[1]] += vectorIncrementPerChannel;
                    srcPtrTemp[permutationOrder[2]] += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    *dstPtrTempR++ = *srcPtrTemp[permutationOrder[0]]++;
                    *dstPtrTempG++ = *srcPtrTemp[permutationOrder[1]]++;
                    *dstPtrTempB++ = *srcPtrTemp[permutationOrder[2]]++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
