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
#include "rpp_cpu_interpolation.hpp"
#include "rpp_cpu_simd_math.hpp"

#if __AVX2__
inline void compute_fisheye_src_loc_avx(__m256& pDstY, __m256& pDstX, __m256& pSrcY, __m256& pSrcX,
                                        __m256& pHeight, __m256& pWidth) {
    __m256 pNormX, pNormY, pDist;
    pNormX = _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(avx_p2, pDstX), pWidth),
                           avx_p1);  //  (static_cast<Rpp32f>((2.0 * dstX)) / width) - 1;
    pNormY = _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(avx_p2, pDstY), pHeight),
                           avx_p1);  //  (static_cast<Rpp32f>((2.0 * dstY)) / height) - 1;
    pDist = rpp_host_math_inverse_sqrt_8_avx(_mm256_fmadd_ps(
        pNormX, pNormX,
        _mm256_mul_ps(pNormY, pNormY)));  //  std::sqrt((normX * normX) + (normY * normY));
    pDist = _mm256_div_ps(avx_p1, pDist);

    __m256 pDistNew, pTheta, pSinFactor, pCosFactor;
    pDistNew = _mm256_sqrt_ps(
        _mm256_sub_ps(avx_p1, _mm256_mul_ps(pDist, pDist)));  //  std::sqrt(1.0 - dist * dist);
    pDistNew = _mm256_mul_ps(_mm256_add_ps(pDist, _mm256_sub_ps(avx_p1, pDistNew)),
                             avx_p1op2);  //  (dist + (1.0 - distNew)) * 0.5f;
    pTheta = atan2_ps(pNormY, pNormX);    //  std::atan2(normY, normX);
    pTheta = _mm256_blendv_ps(avx_p0, pTheta, _mm256_cmp_ps(pTheta, pTheta, _CMP_ORD_Q));
    sincos_ps(pTheta, &pSinFactor, &pCosFactor);

    pSrcX = _mm256_mul_ps(_mm256_mul_ps(_mm256_fmadd_ps(pDistNew, pCosFactor, avx_p1), pWidth),
                          avx_p1op2);
    pSrcY = _mm256_mul_ps(_mm256_mul_ps(_mm256_fmadd_ps(pDistNew, pSinFactor, avx_p1), pHeight),
                          avx_p1op2);
    pSrcX = _mm256_blendv_ps(avx_pMinus1, pSrcX, _mm256_cmp_ps(pSrcX, pSrcX, _CMP_ORD_Q));
    pSrcY = _mm256_blendv_ps(avx_pMinus1, pSrcY, _mm256_cmp_ps(pSrcY, pSrcY, _CMP_ORD_Q));

    __m256 pMask1, pMask2;
    pMask1 = _mm256_and_ps(_mm256_cmp_ps(pDist, avx_p0, _CMP_GE_OQ),
                           _mm256_cmp_ps(pDist, avx_p1, _CMP_LE_OQ));
    pMask2 = _mm256_and_ps(pMask1, _mm256_cmp_ps(pDistNew, avx_p1, _CMP_LE_OQ));
    pSrcX = _mm256_blendv_ps(avx_pMinus1, pSrcX, pMask2);
    pSrcY = _mm256_blendv_ps(avx_pMinus1, pSrcY, pMask2);
    pDstX = _mm256_add_ps(pDstX, avx_p8);
}
#endif

inline void compute_fisheye_src_loc(Rpp32f dstY, Rpp32f dstX, Rpp32f& srcY, Rpp32f& srcX,
                                    Rpp32s& height, Rpp32s& width) {
    Rpp32f normX = (static_cast<Rpp32f>((2.0 * dstX)) / width) - 1.0f;
    Rpp32f normY = (static_cast<Rpp32f>((2.0 * dstY)) / height) - 1.0f;
    Rpp32f dist = 1.0f / rpp_host_math_inverse_sqrt_1((normX * normX) + (normY * normY));
    srcX = -1;
    srcY = -1;
    if ((dist >= 0.0) && (dist <= 1.0)) {
        Rpp32f distNew = std::sqrt((1.0 - dist * dist));
        distNew = (dist + (1.0 - distNew)) * 0.5f;
        if (distNew <= 1.0) {
            Rpp32f theta = std::atan2(normY, normX);
            Rpp32f newX = distNew * std::cos(theta);
            Rpp32f newY = distNew * std::sin(theta);
            srcX = (((newX + 1) * width) * 0.5f);
            srcY = (((newY + 1) * height) * 0.5f);
        }
    }
}

RppStatus fisheye_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                    RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType, RppLayoutParams layoutParams,
                                    rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u alignedLength =
            bufferLength & ~7;        // Align dst width to process 8 dst pixels per iteration
        Rpp32s srcLocArray[8] = {0};  // Since 8 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};  // Since 8 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pRoiLTRB[4], pWidth, pHeight;
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        pWidth = _mm256_set1_ps(roi.xywhROI.roiWidth);
        pHeight = _mm256_set1_ps(roi.xywhROI.roiHeight);
#endif

        // fisheye with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256i pRow;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3_avx, srcPtrChannel, srcLocArray,
                                  invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_u8pkd3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pln3(
                        srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++,
                        srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // fisheye with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256i pRow[4];
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelR, srcLocArray,
                                  invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelG, srcLocArray,
                                  invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelB, srcLocArray,
                                  invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_u8pln3_to_u8pkd3_avx, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp,
                                                                  srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // fisheye without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8u* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256i pRow;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3_avx, srcPtrChannel, srcLocArray,
                                  invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_u8_to_u8_avx, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp,
                                                                  srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // fisheye without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8u* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8u* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad);
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        __m256i pRow;
                        rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrTempChn, srcLocArray,
                                      invalidLoad, pRow);
                        rpp_storeu_si64(reinterpret_cast<__m128i*>(dstPtrTempChn),
                                        _mm256_castsi256_si128(pRow));
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pln_to_pln(
                        srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fisheye_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr, Rpp16f* dstPtr,
                                      RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType, RppLayoutParams layoutParams,
                                      rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u alignedLength =
            bufferLength & ~7;        // Align dst width to process 8 dst pixels per iteration
        Rpp32s srcLocArray[8] = {0};  // Since 8 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};  // Since 8 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pRoiLTRB[4], pWidth, pHeight;
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        pWidth = _mm256_set1_ps(roi.xywhROI.roiWidth);
        pHeight = _mm256_set1_ps(roi.xywhROI.roiHeight);
#endif

        // fisheye with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256 pRow[3];
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f16pkd3_to_f32pln3_avx, srcPtrChannel,
                                  srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pln3(
                        srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++,
                        srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // fisheye with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp16f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256 pRow[3];
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_f16pln1_avx, srcPtrChannelR, srcLocArray,
                                  invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_f16pln1_avx, srcPtrChannelG, srcLocArray,
                                  invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_f16pln1_avx, srcPtrChannelB, srcLocArray,
                                  invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp,
                                                                  srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // fisheye without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp16f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256 pRow[3];
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f16pkd3_to_f32pkd3_avx, srcPtrChannel,
                                  srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_f32pkd3_to_f16pkd3_avx, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp,
                                                                  srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // fisheye without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp16f* dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp16f* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad);
                    Rpp16f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        __m256 pRow;
                        rpp_simd_load(rpp_generic_nn_load_f16pln1_avx, srcPtrTempChn, srcLocArray,
                                      invalidLoad, pRow);
                        rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pln_to_pln(
                        srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fisheye_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr, Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType, RppLayoutParams layoutParams,
                                      rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u alignedLength =
            bufferLength & ~7;        // Align dst width to process 8 dst pixels per iteration
        Rpp32s srcLocArray[8] = {0};  // Since 8 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};  // Since 8 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pRoiLTRB[4], pWidth, pHeight;
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        pWidth = _mm256_set1_ps(roi.xywhROI.roiWidth);
        pHeight = _mm256_set1_ps(roi.xywhROI.roiHeight);
#endif

        // fisheye with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256 pRow[3];
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pln3_avx, srcPtrChannel,
                                  srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pln3(
                        srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++,
                        srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // fisheye with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256 pRow[3];
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1_avx, srcPtrChannelR, srcLocArray,
                                  invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1_avx, srcPtrChannelG, srcLocArray,
                                  invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1_avx, srcPtrChannelB, srcLocArray,
                                  invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp,
                                                                  srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // fisheye without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp32f* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256 pRow[3];
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pkd3_avx, srcPtrChannel,
                                  srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_f32pkd3_to_f32pkd3_avx, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp,
                                                                  srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // fisheye without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp32f* dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp32f* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad);
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        __m256 pRow;
                        rpp_simd_load(rpp_generic_nn_load_f32pln1_avx, srcPtrTempChn, srcLocArray,
                                      invalidLoad, pRow);
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pln_to_pln(
                        srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus fisheye_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                    RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType, RppLayoutParams layoutParams,
                                    rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) +
                        (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u alignedLength =
            bufferLength & ~7;        // Align dst width to process 8 dst pixels per iteration
        Rpp32s srcLocArray[8] = {0};  // Since 8 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};  // Since 8 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pRoiLTRB[4], pWidth, pHeight;
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        pWidth = _mm256_set1_ps(roi.xywhROI.roiWidth);
        pHeight = _mm256_set1_ps(roi.xywhROI.roiHeight);
#endif

        // fisheye with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
            (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256i pRow;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3_avx, srcPtrChannel, srcLocArray,
                                  invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_i8pkd3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG,
                                   dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pln3(
                        srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++,
                        srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // fisheye with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256i pRow[4];
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1_avx, srcPtrChannelR, srcLocArray,
                                  invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1_avx, srcPtrChannelG, srcLocArray,
                                  invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1_avx, srcPtrChannelB, srcLocArray,
                                  invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_i8pln3_to_i8pkd3_avx, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp,
                                                                  srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // fisheye without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) &&
                 (dstDescPtr->layout == RpptLayout::NHWC)) {
            Rpp8s* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    __m256i pRow;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3_avx, srcPtrChannel, srcLocArray,
                                  invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_i8_to_i8_avx, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp,
                                                                  srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // fisheye without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                 (dstDescPtr->layout == RpptLayout::NCHW)) {
            Rpp8s* dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++) {
                Rpp8s* dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength;
                     vectorLoopCount += vectorIncrementPerChannel) {
                    __m256 pSrcX, pSrcY;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH,
                                                                srcLocArray, invalidLoad);
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for (int c = 0; c < srcDescPtr->c; c++) {
                        __m256i pRow;
                        rpp_simd_load(rpp_generic_nn_load_i8pln1_avx, srcPtrTempChn, srcLocArray,
                                      invalidLoad, pRow);
                        rpp_storeu_si64(reinterpret_cast<__m128i*>(dstPtrTempChn),
                                        _mm256_castsi256_si128(pRow));
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++) {
                    Rpp32f srcX, srcY;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight,
                                            roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pln_to_pln(
                        srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
