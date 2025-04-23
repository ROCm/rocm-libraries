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
#include "rpp_cpu_geometric.hpp"
#include "rpp_cpu_interpolation.hpp"

/************* warp_affine helpers *************/

inline void compute_warp_affine_src_loc_next_term_sse(__m128 &pSrcY, __m128 &pSrcX, __m128 &pAffineMatrixTerm3Incr, __m128 &pAffineMatrixTerm0Incr)
{
    pSrcY = _mm_add_ps(pSrcY, pAffineMatrixTerm3Incr);   // Vectorized computation of next 4 src Y locations by adding the delta from previous location
    pSrcX = _mm_add_ps(pSrcX, pAffineMatrixTerm0Incr);   // Vectorized computation of next 4 src X locations by adding the delta from previous location
}

inline void compute_warp_affine_src_loc_next_term_avx(__m256 &pSrcY, __m256 &pSrcX, __m256 &pAffineMatrixTerm3Incr, __m256 &pAffineMatrixTerm0Incr)
{
    pSrcY = _mm256_add_ps(pSrcY, pAffineMatrixTerm3Incr);   // Vectorized computation of next 8 src Y locations by adding the delta from previous location
    pSrcX = _mm256_add_ps(pSrcX, pAffineMatrixTerm0Incr);   // Vectorized computation of next 8 src X locations by adding the delta from previous location
}

inline void compute_warp_affine_src_loc(Rpp32s dstY, Rpp32s dstX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f6 *affineMatrix_f6, Rpp32s roiHalfHeight, Rpp32s roiHalfWidth)
{
    dstX -= roiHalfWidth;
    dstY -= roiHalfHeight;
    srcX = std::fma(dstX, affineMatrix_f6->data[0], std::fma(dstY, affineMatrix_f6->data[1], affineMatrix_f6->data[2])) + roiHalfWidth;
    srcY = std::fma(dstX, affineMatrix_f6->data[3], std::fma(dstY, affineMatrix_f6->data[4], affineMatrix_f6->data[5])) + roiHalfHeight;
}

inline void compute_warp_affine_src_loc_next_term(Rpp32s dstX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f6 *affineMatrix_f6)
{
    srcY += affineMatrix_f6->data[3];   // Computation of next src Y locations by adding the delta from previous location
    srcX += affineMatrix_f6->data[0];   // Computation of next src X locations by adding the delta from previous location
}

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus warp_affine_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *affineTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams,
                                           rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLoc[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pAffineMatrixTerm0 = _mm_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3);
        __m128 pAffineMatrixTerm3 = _mm_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3);
        __m128 pAffineMatrixTerm0Incr = _mm_set1_ps(affineMatrix_f6->data[0] * 4);
        __m128 pAffineMatrixTerm3Incr = _mm_set1_ps(affineMatrix_f6->data[3] * 4);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow[3];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_u8_to_u8, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_simd_store(rpp_storeu_si32, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *affineTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams srcLayoutParams,
                                             rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLoc[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pAffineMatrixTerm0 = _mm_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3);
        __m128 pAffineMatrixTerm3 = _mm_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3);
        __m128 pAffineMatrixTerm0Incr = _mm_set1_ps(affineMatrix_f6->data[0] * 4);
        __m128 pAffineMatrixTerm3Incr = _mm_set1_ps(affineMatrix_f6->data[3] * 4);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[3];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pln3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[4];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[4];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_f32pkd3_to_f32pkd3, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128 pRow;
                        rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *affineTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams,
                                           rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLoc[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pAffineMatrixTerm0 = _mm_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3);
        __m128 pAffineMatrixTerm3 = _mm_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3);
        __m128 pAffineMatrixTerm0Incr = _mm_set1_ps(affineMatrix_f6->data[0] * 4);
        __m128 pAffineMatrixTerm3Incr = _mm_set1_ps(affineMatrix_f6->data[3] * 4);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow[3];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_i8_to_i8, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_simd_store(rpp_storeu_si32, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *affineTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams srcLayoutParams,
                                             rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp16f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

// /************* BILINEAR INTERPOLATION *************/

RppStatus warp_affine_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp8u *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *affineTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams srcLayoutParams,
                                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 8 dst pixels per iteration

        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pAffineMatrixTerm0 = _mm256_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3, affineMatrix_f6->data[0] * 4, affineMatrix_f6->data[0] * 5, affineMatrix_f6->data[0] * 6, affineMatrix_f6->data[0] * 7);
        __m256 pAffineMatrixTerm3 = _mm256_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3, affineMatrix_f6->data[3] * 4, affineMatrix_f6->data[3] * 5, affineMatrix_f6->data[3] * 6, affineMatrix_f6->data[3] * 7);
        __m256 pAffineMatrixTerm0Incr = _mm256_set1_ps(affineMatrix_f6->data[0] * 8);
        __m256 pAffineMatrixTerm3Incr = _mm256_set1_ps(affineMatrix_f6->data[3] * 8);
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

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *affineTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams srcLayoutParams,
                                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 8 dst pixels per iteration

        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pAffineMatrixTerm0 = _mm256_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3, affineMatrix_f6->data[0] * 4, affineMatrix_f6->data[0] * 5, affineMatrix_f6->data[0] * 6, affineMatrix_f6->data[0] * 7);
        __m256 pAffineMatrixTerm3 = _mm256_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3, affineMatrix_f6->data[3] * 4, affineMatrix_f6->data[3] * 5, affineMatrix_f6->data[3] * 6, affineMatrix_f6->data[3] * 7);
        __m256 pAffineMatrixTerm0Incr = _mm256_set1_ps(affineMatrix_f6->data[0] * 8);
        __m256 pAffineMatrixTerm3Incr = _mm256_set1_ps(affineMatrix_f6->data[3] * 8);
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

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp8s *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *affineTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams srcLayoutParams,
                                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 8 dst pixels per iteration

        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pAffineMatrixTerm0 = _mm256_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3, affineMatrix_f6->data[0] * 4, affineMatrix_f6->data[0] * 5, affineMatrix_f6->data[0] * 6, affineMatrix_f6->data[0] * 7);
        __m256 pAffineMatrixTerm3 = _mm256_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3, affineMatrix_f6->data[3] * 4, affineMatrix_f6->data[3] * 5, affineMatrix_f6->data[3] * 6, affineMatrix_f6->data[3] * 7);
        __m256 pAffineMatrixTerm0Incr = _mm256_set1_ps(affineMatrix_f6->data[0] * 8);
        __m256 pAffineMatrixTerm3Incr = _mm256_set1_ps(affineMatrix_f6->data[3] * 8);
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

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_1c_avx(pDst);
                    rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp16f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *affineTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams srcLayoutParams,
                                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 8 dst pixels per iteration

        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pAffineMatrixTerm0 = _mm256_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3, affineMatrix_f6->data[0] * 4, affineMatrix_f6->data[0] * 5, affineMatrix_f6->data[0] * 6, affineMatrix_f6->data[0] * 7);
        __m256 pAffineMatrixTerm3 = _mm256_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3, affineMatrix_f6->data[3] * 4, affineMatrix_f6->data[3] * 5, affineMatrix_f6->data[3] * 6, affineMatrix_f6->data[3] * 7);
        __m256 pAffineMatrixTerm0Incr = _mm256_set1_ps(affineMatrix_f6->data[0] * 8);
        __m256 pAffineMatrixTerm3Incr = _mm256_set1_ps(affineMatrix_f6->data[3] * 8);
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

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    Rpp32f dstPtrTemp_ps[25];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    for(int cnt = 0; cnt < vectorIncrementPkd; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    Rpp32f dstPtrTemp_ps[25];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    for(int cnt = 0; cnt < vectorIncrementPkd; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm256_add_ps(_mm256_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm256_add_ps(_mm256_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    Rpp32f dstPtrTemp_ps[8];
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTemp_ps, pDst); // Store dst pixels
                    compute_warp_affine_src_loc_next_term_avx(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
