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
#include "rpp_cpu_geometric.hpp"

/************* warp_perspective helpers *************/

#if __AVX2__
inline void compute_warp_perspective_src_loc_next_term_avx(__m256 &plocW, __m256 &plocY, __m256 &plocX, __m256 &pSrcY, __m256 &pSrcX, __m256 &pPerspectiveMatrixTerm6Incr, __m256 &pPerspectiveMatrixTerm3Incr, __m256 &pPerspectiveMatrixTerm0Incr, __m256 &pRoiHalfHeight, __m256 &pRoiHalfWidth)
{
    plocW = _mm256_add_ps(plocW, pPerspectiveMatrixTerm6Incr);
    plocY = _mm256_add_ps(plocY, pPerspectiveMatrixTerm3Incr);
    plocX = _mm256_add_ps(plocX, pPerspectiveMatrixTerm0Incr);
    pSrcY = _mm256_add_ps(_mm256_div_ps(plocY, plocW), pRoiHalfHeight);
    pSrcX = _mm256_add_ps(_mm256_div_ps(plocX, plocW), pRoiHalfWidth);
}
inline void compute_warp_perspective_src_loc_first_term_avx(Rpp32f locX, Rpp32f locY, Rpp32f locW, __m256 &plocW, __m256 &plocY, __m256 &plocX, __m256 &pSrcY, __m256 &pSrcX, __m256 &pPerspectiveMatrixTerm6, __m256 &pPerspectiveMatrixTerm3, __m256 &pPerspectiveMatrixTerm0, __m256 &pRoiHalfHeight, __m256 &pRoiHalfWidth) {
    plocX = _mm256_add_ps(_mm256_set1_ps(locX), pPerspectiveMatrixTerm0);
    plocY = _mm256_add_ps(_mm256_set1_ps(locY), pPerspectiveMatrixTerm3);
    plocW = _mm256_add_ps(_mm256_set1_ps(locW), pPerspectiveMatrixTerm6);
    pSrcY = _mm256_add_ps(_mm256_div_ps(plocY, plocW), pRoiHalfHeight);
    pSrcX = _mm256_add_ps(_mm256_div_ps(plocX, plocW), pRoiHalfWidth);
}
#endif

inline void compute_warp_perspective_src_loc_params(Rpp32s dstY, Rpp32s dstX, Rpp32f &locW, Rpp32f &locY, Rpp32f &locX, Rpp32f9 *perspectiveMatrix_f9, Rpp32s roiHalfHeight, Rpp32s roiHalfWidth)
{
    dstX -= roiHalfWidth;
    dstY -= roiHalfHeight;
    locX = std::fma(dstX, perspectiveMatrix_f9->data[0], std::fma(dstY, perspectiveMatrix_f9->data[1], perspectiveMatrix_f9->data[2]));
    locY = std::fma(dstX, perspectiveMatrix_f9->data[3], std::fma(dstY, perspectiveMatrix_f9->data[4], perspectiveMatrix_f9->data[5]));
    locW = std::fma(dstX, perspectiveMatrix_f9->data[6], std::fma(dstY, perspectiveMatrix_f9->data[7], perspectiveMatrix_f9->data[8]));
}

inline void compute_warp_perspective_src_loc_next_term(Rpp32s dstX, Rpp32f &locW, Rpp32f &locY, Rpp32f &locX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f9 *perspectiveMatrix_f9, Rpp32s roiHalfHeight, Rpp32s roiHalfWidth)
{
    locW += perspectiveMatrix_f9->data[6];
    locY += perspectiveMatrix_f9->data[3];   // Used in computation of next srcY locations by adding the delta from previous location
    locX += perspectiveMatrix_f9->data[0];   // Used in computation of next srcX locations by adding the delta from previous location
    srcX = ((locX / locW) + roiHalfWidth);
    srcY = ((locY / locW) + roiHalfHeight);
}

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus warp_perspective_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp8u *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32f *perspectiveTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams srcLayoutParams,
                                                rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = reinterpret_cast<Rpp32f9 *>(perspectiveTensor + batchCount * 9);

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration
        Rpp32s srcLoc[8] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};    // Since 4 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 pRoiHalfHeight = _mm256_set1_ps(roiHalfHeight);
        __m256 pRoiHalfWidth = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
#endif

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_u8pkd3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_u8pln3_to_u8pkd3_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_u8_to_u8_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for (int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256i pRow;
                        rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_storeu_si64(reinterpret_cast<__m128i *>(dstPtrTempChn), _mm256_castsi256_si128(pRow));
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_perspective_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32f *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *perspectiveTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams srcLayoutParams,
                                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = reinterpret_cast<Rpp32f9 *>(perspectiveTensor + batchCount * 9);

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration
        Rpp32s srcLoc[8] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};    // Since 4 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 pRoiHalfHeight = _mm256_set1_ps(roiHalfHeight);
        __m256 pRoiHalfWidth = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
#endif

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pln3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1_avx, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1_avx, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1_avx, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pkd3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_f32pkd3_to_f32pkd3_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for (int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256 pRow;
                        rpp_simd_load(rpp_generic_nn_load_f32pln1_avx, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_perspective_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp8s *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32f *perspectiveTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams srcLayoutParams,
                                                rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = reinterpret_cast<Rpp32f9 *>(perspectiveTensor + batchCount * 9);

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration
        Rpp32s srcLoc[8] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};    // Since 4 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 pRoiHalfHeight = _mm256_set1_ps(roiHalfHeight);
        __m256 pRoiHalfWidth = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
#endif

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_i8pkd3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1_avx, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1_avx, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1_avx, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_i8pln3_to_i8pkd3_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_i8_to_i8_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for (int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256i pRow;
                        rpp_simd_load(rpp_generic_nn_load_i8pln1_avx, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_storeu_si64(reinterpret_cast<__m128i *>(dstPtrTempChn), _mm256_castsi256_si128(pRow));
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_perspective_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp16f *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *perspectiveTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams srcLayoutParams,
                                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = reinterpret_cast<Rpp32f9 *>(perspectiveTensor + batchCount * 9);

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration
        Rpp32s srcLoc[8] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};    // Since 4 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 pRoiHalfHeight = _mm256_set1_ps(roiHalfHeight);
        __m256 pRoiHalfWidth = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
#endif

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f16pkd3_to_f32pln3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp16f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_f16pln1_avx, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_f16pln1_avx, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_f16pln1_avx, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f16pkd3_to_f32pkd3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_f32pkd3_to_f16pkd3_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp16f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for (int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m256 pRow;
                        rpp_simd_load(rpp_generic_nn_load_f16pln1_avx, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_perspective_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp8u *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      Rpp32f *perspectiveTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams srcLayoutParams,
                                                      rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = reinterpret_cast<Rpp32f9 *>(perspectiveTensor + batchCount * 9);

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration

#if __AVX2__
        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 pRoiHalfHeight = _mm256_set1_ps(roiHalfHeight);
        __m256 pRoiHalfWidth = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);

        __m256i pxSrcStridesCHW[3];
        pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
        pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
        pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;
#endif

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Warp Perspective with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Perspective without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_perspective_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp32f *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32f *perspectiveTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams srcLayoutParams,
                                                        rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = reinterpret_cast<Rpp32f9 *>(perspectiveTensor + batchCount * 9);

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration

#if __AVX2__
        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 pRoiHalfHeight = _mm256_set1_ps(roiHalfHeight);
        __m256 pRoiHalfWidth = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);

        __m256i pxSrcStridesCHW[3];
        pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
        pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
        pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;
#endif

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Warp Perspective with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Perspective without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_perspective_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp8s *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      Rpp32f *perspectiveTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams srcLayoutParams,
                                                      rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = reinterpret_cast<Rpp32f9 *>(perspectiveTensor + batchCount * 9);

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration

#if __AVX2__
        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 pRoiHalfHeight = _mm256_set1_ps(roiHalfHeight);
        __m256 pRoiHalfWidth = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);

        __m256i pxSrcStridesCHW[3];
        pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
        pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
        pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;
#endif

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Warp Perspective with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Perspective without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_1c_avx(pDst);
                    rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_perspective_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp16f *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32f *perspectiveTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams srcLayoutParams,
                                                        rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = reinterpret_cast<Rpp32f9 *>(perspectiveTensor + batchCount * 9);

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration

#if __AVX2__
        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 pRoiHalfHeight = _mm256_set1_ps(roiHalfHeight);
        __m256 pRoiHalfWidth = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);

        __m256i pxSrcStridesCHW[3];
        pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
        pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
        pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;
#endif

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Warp Perspective with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Warp perspective without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPkd;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Perspective without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f locX, locY, locW, srcX, srcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, locW, locY, locX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
#if __AVX2__
                __m256 plocX, plocY, plocW, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_first_term_avx(locX, locY, locW, plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6, pPerspectiveMatrixTerm3, pPerspectiveMatrixTerm0, pRoiHalfHeight, pRoiHalfWidth);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_perspective_src_loc_next_term_avx(plocW, plocY, plocX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, pRoiHalfHeight, pRoiHalfWidth);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                locW += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                locY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                locX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = ((locX / locW) + roiHalfWidth);
                srcY = ((locY / locW) + roiHalfHeight);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, locW, locY, locX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
