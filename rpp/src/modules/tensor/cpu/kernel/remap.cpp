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

inline void compute_remap_src_loc_sse(Rpp32f *rowRemapTablePtr, Rpp32f *colRemapTablePtr, Rpp32s *locArray, __m128 &pStride, __m128 &pWidthLimit, __m128 &pHeightLimit, const __m128 &pChannel = xmm_p1)
{
    __m128 pRowRemapVal = _mm_loadu_ps(rowRemapTablePtr);
    pRowRemapVal = _mm_max_ps(_mm_min_ps(pRowRemapVal, pHeightLimit), xmm_p0);
    __m128 pColRemapVal = _mm_loadu_ps(colRemapTablePtr);
    pColRemapVal = _mm_max_ps(_mm_min_ps(pColRemapVal, pWidthLimit), xmm_p0);
    __m128i pxRemappedSrcLoc = _mm_cvtps_epi32(_mm_fmadd_ps(pRowRemapVal, pStride, _mm_mul_ps(pColRemapVal, pChannel)));
    _mm_storeu_si128((__m128i*) locArray, pxRemappedSrcLoc);
}

inline void compute_remap_src_loc(Rpp32f rowLoc, Rpp32f colLoc, Rpp32s &srcLoc, Rpp32s stride, Rpp32f widthLimit, Rpp32f heightLimit, Rpp32s channels = 1)
{
    rowLoc = std::max(0.0f, std::min(rowLoc, heightLimit));
    colLoc = std::max(0.0f, std::min(colLoc, widthLimit));
    srcLoc = (rowLoc * stride) + colLoc * channels;
}

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus remap_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *rowRemapTable,
                                     Rpp32f *colRemapTable,
                                     RpptDescPtr remapTableDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);

omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrement = 12;
        Rpp32s remappedSrcLoc;
        Rpp32s remapSrcLocArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32f widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f heightLimit = roi.xywhROI.roiHeight - 1;
        __m128 pWidthLimit = _mm_set1_ps(widthLimit);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);

        // Remap with fused output-layout toggle (NHWC -> NCHW)
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
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow;
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit, pSrcChannel);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrChannel, remapSrcLocArray, pxRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit, srcDescPtr->c);
                    *dstPtrTempR++ = *(srcPtrChannel + remappedSrcLoc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + remappedSrcLoc);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow[3];
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowR, remapSrcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowG, remapSrcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowB, remapSrcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit);
                    *dstPtrTemp++ = *(srcPtrRowR + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrRowG + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrRowB + remappedSrcLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow;
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit, pSrcChannel);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrChannel, remapSrcLocArray, pxRow);
                    rpp_simd_store(rpp_store12_u8_to_u8, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit, srcDescPtr->c);
                    *dstPtrTemp++ = *(srcPtrChannel + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrChannel + 2 + remappedSrcLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        __m128i pxRow;
                        rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempChn, remapSrcLocArray, pxRow);
                        rpp_simd_store(rpp_storeu_si32, dstPtrTempChn, pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp8u * dstPtrTempChannel = dstPtrTemp;
                    Rpp8u * srcPtrTempChannel = srcPtrChannel;
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit);
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        *dstPtrTempChannel = *(srcPtrTempChannel + remappedSrcLoc);
                        dstPtrTempChannel += dstDescPtr->strides.cStride;
                        srcPtrTempChannel += srcDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus remap_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *rowRemapTable,
                                       Rpp32f *colRemapTable,
                                       RpptDescPtr remapTableDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);

omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrement = 12;
        Rpp32s remappedSrcLoc;
        Rpp32s remapSrcLocArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32f widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f heightLimit = roi.xywhROI.roiHeight - 1;
        __m128 pWidthLimit = _mm_set1_ps(widthLimit);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);

        // Remap with fused output-layout toggle (NHWC -> NCHW)
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
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[3];
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit, pSrcChannel);
                    rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3, srcPtrChannel, remapSrcLocArray, pRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit, srcDescPtr->c);
                    *dstPtrTempR++ = *(srcPtrChannel + remappedSrcLoc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + remappedSrcLoc);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[4];
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowR, remapSrcLocArray, pRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowG, remapSrcLocArray, pRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowB, remapSrcLocArray, pRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit);
                    *dstPtrTemp++ = *(srcPtrRowR + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrRowG + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrRowB + remappedSrcLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    __m128 pRow;
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit, pSrcChannel);
                    rpp_simd_load(rpp_load4_f32_to_f32, (srcPtrChannel + *remapSrcLocArray), &pRow);
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &pRow);
                    dstPtrTemp += 3;
                    rowRemapTableTemp++;
                    colRemapTableTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        __m128 pRow;
                        rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempChn, remapSrcLocArray, pRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32f * dstPtrTempChannel = dstPtrTemp;
                    Rpp32f * srcPtrTempChannel = srcPtrChannel;
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit);
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        *dstPtrTempChannel = *(srcPtrTempChannel + remappedSrcLoc);
                        dstPtrTempChannel += dstDescPtr->strides.cStride;
                        srcPtrTempChannel += srcDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus remap_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *rowRemapTable,
                                     Rpp32f *colRemapTable,
                                     RpptDescPtr remapTableDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);

omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrement = 12;
        Rpp32s remappedSrcLoc;
        Rpp32s remapSrcLocArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32f widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f heightLimit = roi.xywhROI.roiHeight - 1;
        __m128 pWidthLimit = _mm_set1_ps(widthLimit);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);

        // Remap with fused output-layout toggle (NHWC -> NCHW)
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
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow;
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit, pSrcChannel);
                    rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrChannel, remapSrcLocArray, pxRow);
                    rpp_simd_store(rpp_store12_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit, srcDescPtr->c);
                    *dstPtrTempR++ = *(srcPtrChannel + remappedSrcLoc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + remappedSrcLoc);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow[3];
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowR, remapSrcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowG, remapSrcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowB, remapSrcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit);
                    *dstPtrTemp++ = *(srcPtrRowR + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrRowG + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrRowB + remappedSrcLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow;
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit, pSrcChannel);
                    rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrChannel, remapSrcLocArray, pxRow);
                    rpp_simd_store(rpp_store12_i8_to_i8, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit, srcDescPtr->c);
                    *dstPtrTemp++ = *(srcPtrChannel + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrChannel + 2 + remappedSrcLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_remap_src_loc_sse(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pWidthLimit, pHeightLimit);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        __m128i pxRow;
                        rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempChn, remapSrcLocArray, pxRow);
                        rpp_simd_store(rpp_storeu_si32, dstPtrTempChn, pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp8s * dstPtrTempChannel = dstPtrTemp;
                    Rpp8s * srcPtrTempChannel = srcPtrChannel;
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit);
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        *dstPtrTempChannel = *(srcPtrTempChannel + remappedSrcLoc);
                        dstPtrTempChannel += dstDescPtr->strides.cStride;
                        srcPtrTempChannel += srcDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus remap_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *rowRemapTable,
                                       Rpp32f *colRemapTable,
                                       RpptDescPtr remapTableDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);

omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32f widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f heightLimit = roi.xywhROI.roiHeight - 1;
        // Remap with 3 channel inputs and outputs
        if (srcDescPtr->c == 3 && dstDescPtr->c == 3)
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
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
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s remappedSrcLoc;
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit, layoutParams.bufferMultiplier);
                    *dstPtrTempR = (Rpp16f)*(srcPtrRowR + remappedSrcLoc);
                    *dstPtrTempG = (Rpp16f)*(srcPtrRowG + remappedSrcLoc);
                    *dstPtrTempB = (Rpp16f)*(srcPtrRowB + remappedSrcLoc);
                    dstPtrTempR = dstPtrTempR + dstDescPtr->strides.wStride;
                    dstPtrTempG = dstPtrTempG + dstDescPtr->strides.wStride;
                    dstPtrTempB = dstPtrTempB + dstDescPtr->strides.wStride;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap with single channel inputs and outputs
        else
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s remappedSrcLoc;
                    compute_remap_src_loc(*rowRemapTableTemp++, *colRemapTableTemp++, remappedSrcLoc, srcDescPtr->strides.hStride, widthLimit, heightLimit);
                    *dstPtrTemp++ = (Rpp16f)*(srcPtrRow + remappedSrcLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

/************* BILINEAR INTERPOLATION *************/

RppStatus remap_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *rowRemapTable,
                                           Rpp32f *colRemapTable,
                                           RpptDescPtr remapTableDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();
#if __AVX2__
    __m256 pSrcChannel = _mm256_set1_ps(srcDescPtr->c);
    __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256i pxSrcStridesCHW[3];
    pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
    pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
    pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
#endif

omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;   // Align dst width to process 8 dst pixels per iteration
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrement = 24;

#if __AVX2__
        __m256 pBilinearCoeffs[4];
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;
#endif

        // Remap with fused output-layout toggle (NHWC -> NCHW)
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

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NCHW -> NCHW)
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

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pln_to_pln(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pln_to_pln(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus remap_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *rowRemapTable,
                                             Rpp32f *colRemapTable,
                                             RpptDescPtr remapTableDescPtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

#if __AVX2__
    __m256 pSrcChannel = _mm256_set1_ps(srcDescPtr->c);
    __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256i pxSrcStridesCHW[3];
    pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
    pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
    pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
#endif

omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;   // Align dst width to process 8 dst pixels per iteration
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrement = 24;

#if __AVX2__
        __m256 pBilinearCoeffs[4];
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;
#endif

        // Remap with fused output-layout toggle (NHWC -> NCHW)
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

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NCHW -> NCHW)
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

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pln_to_pln(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pln_to_pln(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus remap_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *rowRemapTable,
                                           Rpp32f *colRemapTable,
                                           RpptDescPtr remapTableDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams ,
                                           rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

#if __AVX2__
    __m256 pSrcChannel = _mm256_set1_ps(srcDescPtr->c);
    __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256i pxSrcStridesCHW[3];
    pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
    pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
    pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
#endif

omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;   // Align dst width to process 8 dst pixels per iteration
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrement = 24;

#if __AVX2__
        __m256 pBilinearCoeffs[4];
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;
#endif

        // Remap with fused output-layout toggle (NHWC -> NCHW)
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

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NCHW -> NCHW)
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

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pln_to_pln(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[4], pDst;
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_1c_avx(pDst);
                    rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pln_to_pln(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus remap_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *rowRemapTable,
                                             Rpp32f *colRemapTable,
                                             RpptDescPtr remapTableDescPtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

#if __AVX2__
    __m256 pSrcChannel = _mm256_set1_ps(srcDescPtr->c);
    __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
    __m256i pxSrcStridesCHW[3];
    pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
    pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
    pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
#endif

omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32f *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + batchCount * remapTableDescPtr->strides.nStride;
        colRemapTableImage = colRemapTable + batchCount * remapTableDescPtr->strides.nStride;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;   // Align dst width to process 8 dst pixels per iteration
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrement = 24;

#if __AVX2__
        __m256 pBilinearCoeffs[4];
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;
#endif

        // Remap with fused output-layout toggle (NHWC -> NCHW)
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

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, pDst); // Store dst pixels
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    Rpp32f dstPtrTemp_ps[25];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, pDst); // Store dst pixels
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    Rpp32f dstPtrTemp_ps[25];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, pDst); // Store dst pixels
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    dstPtrTemp += vectorIncrement;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NCHW -> NCHW)
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

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[12], pDst[3];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, pRoiLTRB, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, pDst); // Store dst pixels
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pln_to_pln(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage;
                colRemapTableTemp = colRemapTableImage;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX = _mm256_loadu_ps(colRemapTableTemp);
                    __m256 pSrcY = _mm256_loadu_ps(rowRemapTableTemp);
                    __m256 pSrc[4], pDst;
                    Rpp32f dstPtrTemp_ps[8];
                    compute_generic_bilinear_srclocs_1c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, pRoiLTRB);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTemp_ps, pDst); // Store dst pixels
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    dstPtrTemp += vectorIncrementPerChannel;
                    rowRemapTableTemp += vectorIncrementPerChannel;
                    colRemapTableTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                    compute_generic_bilinear_interpolation_pln_to_pln(*rowRemapTableTemp++, *colRemapTableTemp++, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);

                dstPtrRow += dstDescPtr->strides.hStride;
                rowRemapTableImage += remapTableDescPtr->strides.hStride;
                colRemapTableImage += remapTableDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
