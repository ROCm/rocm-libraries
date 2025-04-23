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
#include "rpp_cpu_random.hpp"

inline void compute_jitter_src_loc_avx(__m256i *pxXorwowStateX, __m256i *pxXorwowStateCounter, __m256 &pRow, __m256 &pCol, __m256 &pKernelSize, __m256 &pBound, __m256 &pHeightLimit, __m256 &pWidthLimit, __m256 &pStride, __m256 &pChannel, Rpp32s *srcLoc)
{
    __m256 pRngX = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    __m256 pRngY = rpp_host_rng_xorwow_8_f32_avx(pxXorwowStateX, pxXorwowStateCounter);
    __m256 pX = _mm256_mul_ps(pRngX, pKernelSize);
    __m256 pY = _mm256_mul_ps(pRngY, pKernelSize);
    pX = _mm256_max_ps(_mm256_min_ps(_mm256_floor_ps(_mm256_add_ps(pRow, _mm256_sub_ps(pX, pBound))), pHeightLimit), avx_p0);
    pY = _mm256_max_ps(_mm256_min_ps(_mm256_floor_ps(_mm256_add_ps(pCol, _mm256_sub_ps(pY, pBound))), pWidthLimit), avx_p0);
    __m256i pxSrcLoc = _mm256_cvtps_epi32(_mm256_fmadd_ps(pX, pStride, _mm256_mul_ps(pY, pChannel)));
    _mm256_storeu_si256((__m256i*) srcLoc, pxSrcLoc);
}

inline void compute_jitter_src_loc(RpptXorwowStateBoxMuller *xorwowState, Rpp32s row, Rpp32s col, Rpp32s kSize, Rpp32s heightLimit, Rpp32s widthLimit, Rpp32s stride, Rpp32s bound, Rpp32s channels, Rpp32s &loc)
{
    Rpp32u heightIncrement = rpp_host_rng_xorwow_f32(xorwowState) * kSize;
    Rpp32u widthIncrement = rpp_host_rng_xorwow_f32(xorwowState) * kSize;
    loc = std::max(std::min(static_cast<int>(row + heightIncrement - bound), heightLimit), 0) * stride;
    loc += std::max(std::min(static_cast<int>(col + widthIncrement  - bound), (widthLimit - 1)), 0) * channels;
}

RppStatus jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *kernelSizeTensor,
                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize - 1) / 2;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[8] = {0};

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m256 pKernelSize = _mm256_set1_ps(kernelSize);
        __m256 pChannel = _mm256_set1_ps(layoutParams.bufferMultiplier);
        __m256 pHStride = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pHeightLimit = _mm256_set1_ps(heightLimit);
        __m256 pWidthLimit = _mm256_set1_ps(roi.xywhROI.roiWidth - 1);
        __m256 pBound = _mm256_set1_ps(bound);

        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pxRow;
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_resize_nn_extract_pkd3_avx(srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store24_u8pkd3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    *dstPtrTempR++ = *(srcPtrChannel + loc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pxRow[3];
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_resize_nn_extract_pln1_avx(srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_resize_nn_extract_pln1_avx(srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_resize_nn_extract_pln1_avx(srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store24_u8pln3_to_u8pkd3_avx, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTemp++ = *(srcPtrRowR + loc);
                    *dstPtrTemp++ = *(srcPtrRowG + loc);
                    *dstPtrTemp++ = *(srcPtrRowB + loc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pxRow;
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_resize_nn_extract_pkd3_avx(srcPtrRow, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store24_u8_to_u8_avx, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTemp++ = *(srcPtrRow + loc);
                    *dstPtrTemp++ = *(srcPtrRow + 1 + loc);
                    *dstPtrTemp++ = *(srcPtrRow + 2 + loc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256i pxRow;
                        rpp_resize_nn_extract_pln1_avx(srcPtrTempChn, srcLocArray, pxRow);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChn), _mm256_castsi256_si128(pxRow));
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (;vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn = dstPtrTemp;
                    Rpp8u *srcPtrTempChn = srcPtrChannel;
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *(srcPtrTempChn + loc);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u *kernelSizeTensor,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize - 1) / 2;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[8] = {0};

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m256 pKernelSize = _mm256_set1_ps(kernelSize);
        __m256 pChannel = _mm256_set1_ps(layoutParams.bufferMultiplier);
        __m256 pHStride = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pHeightLimit = _mm256_set1_ps(heightLimit);
        __m256 pWidthLimit = _mm256_set1_ps(roi.xywhROI.roiWidth-1);
        __m256 pBound = _mm256_set1_ps(bound);

        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pxRow[3];
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3_avx, srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTempR++ = *(srcPtrChannel + loc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pxRow[4];
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1_avx, srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1_avx, srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1_avx, srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTemp++ = *(srcPtrRowR + loc);
                    *dstPtrTemp++ = *(srcPtrRowG + loc);
                    *dstPtrTemp++ = *(srcPtrRowB + loc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    __m256 pRow;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, (srcPtrChannel + loc), &pRow);
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &pRow);
                    dstPtrTemp += 3;
                }
#endif
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Jitter with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        __m256 pxRow;
                        rpp_simd_load(rpp_resize_nn_load_f32pln1_avx, srcPtrTempChn, srcLocArray, pxRow);
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTempChn, &pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (;vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn = dstPtrTemp;
                    Rpp32f *srcPtrTempChn = srcPtrChannel;
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp32f)*(srcPtrTempChn + loc);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp16f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u *kernelSizeTensor,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize - 1) / 2;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[8] = {0};

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m256 pKernelSize = _mm256_set1_ps(kernelSize);
        __m256 pChannel = _mm256_set1_ps(layoutParams.bufferMultiplier);
        __m256 pHStride = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pHeightLimit = _mm256_set1_ps(heightLimit);
        __m256 pWidthLimit = _mm256_set1_ps(roi.xywhROI.roiWidth-1);
        __m256 pBound = _mm256_set1_ps(bound);

        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    __m256 pxRow[3];
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_f16pkd3_to_f32pln3_avx, srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, pxRow);
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTempR++ = *(srcPtrChannel + loc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f dstPtrTemp_ps[25];
                    __m256 pxRow[4];
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_f16pln1_avx, srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_f16pln1_avx, srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_f16pln1_avx, srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, pxRow);
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTemp++ = *(srcPtrRowR + loc);
                    *dstPtrTemp++ = *(srcPtrRowG + loc);
                    *dstPtrTemp++ = *(srcPtrRowB + loc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];
                    Rpp32s loc;
                    __m256 pRow;

                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTemp_ps[cnt] = (Rpp16f)srcPtrChannel[loc + cnt];
                    }

                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, &pRow);
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, &pRow);

                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    }
                    dstPtrTemp += 3;
                }
#endif
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Jitter with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp16f *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        Rpp32f dstPtrTemp_ps[8];
                        __m256 pxRow;
                        rpp_simd_load(rpp_resize_nn_load_f16pln1_avx, srcPtrTempChn, srcLocArray, pxRow);
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, &pxRow);
                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            dstPtrTempChn[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                        }
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (;vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp16f *dstPtrTempChn = dstPtrTemp;
                    Rpp16f *srcPtrTempChn = srcPtrChannel;
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp16f)*(srcPtrTempChn + loc);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *kernelSizeTensor,
                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize - 1) / 2;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[8] = {0};

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m256 pKernelSize = _mm256_set1_ps(kernelSize);
        __m256 pChannel = _mm256_set1_ps(layoutParams.bufferMultiplier);
        __m256 pHStride = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pHeightLimit = _mm256_set1_ps(heightLimit);
        __m256 pWidthLimit = _mm256_set1_ps(roi.xywhROI.roiWidth-1);
        __m256 pBound = _mm256_set1_ps(bound);

        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pxRow;
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_resize_nn_extract_pkd3_avx(srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store24_i8pkd3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTempR++ = *(srcPtrChannel + loc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pxRow[3];
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_resize_nn_extract_pln1_avx(srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_resize_nn_extract_pln1_avx(srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_resize_nn_extract_pln1_avx(srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store24_i8pln3_to_i8pkd3_avx, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTemp++ = *(srcPtrRowR + loc);
                    *dstPtrTemp++ = *(srcPtrRowG + loc);
                    *dstPtrTemp++ = *(srcPtrRowB + loc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pxRow;
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_resize_nn_extract_pkd3_avx(srcPtrRow, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store24_i8_to_i8_avx, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + loc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + 1 + loc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + 2 + loc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Jitter with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m256 pRow = _mm256_set1_ps(dstLocRow);
                __m256 pCol = avx_pDstLocInit;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_jitter_src_loc_avx(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256i pxRow;
                        rpp_resize_nn_extract_pln1_avx(srcPtrTempChn, srcLocArray, pxRow);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChn), _mm256_castsi256_si128(pxRow));
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    pCol = _mm256_add_ps(avx_p8, pCol);
                }
#endif
                for (;vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn = dstPtrTemp;
                    Rpp8s *srcPtrTempChn = srcPtrChannel;
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8s)*(srcPtrTempChn + loc);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
