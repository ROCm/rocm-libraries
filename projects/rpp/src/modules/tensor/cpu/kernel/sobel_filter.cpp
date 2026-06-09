/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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
#include "rpp_cpu_filter.hpp"

Rpp32f sobel3x3X[9] = {-1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1};
Rpp32f sobel3x3Y[9] = {-1, -2, -1,
                       0, 0, 0,
                       1, 2, 1};
Rpp32f sobel5x5X[25] = {-1,  -2,   0,   2,   1,
                        -4,  -8,   0,   8,   4,
                        -6, -12,   0,  12,   6,
                        -4,  -8,   0,   8,   4,
                        -1,  -2,   0,   2,   1};
Rpp32f sobel5x5Y[25] = {-1,  -4,  -6,  -4,  -1,
                        -2,  -8, -12,  -8,  -2,
                         0,   0,   0,   0,   0,
                         2,   8,  12,   8,   2,
                         1,   4,   6,   4,   1};
Rpp32f sobel7x7X[49] = {-1,   -4,   -5,    0,    5,    4,    1,
                        -6,  -24,  -30,    0,   30,   24,    6,
                        -15,  -60,  -75,    0,   75,   60,   15,
                        -20,  -80, -100,    0,  100,   80,   20,
                        -15,  -60,  -75,    0,   75,   60,   15,
                        -6,  -24,  -30,    0,   30,   24,    6,
                        -1,   -4,   -5,    0,    5,    4,    1};
Rpp32f sobel7x7Y[49] = {-1,   -6,  -15,  -20,  -15,   -6,   -1,
                        -4,  -24,  -60,  -80,  -60,  -24,   -4,
                        -5,  -30,  -75, -100,  -75,  -30,   -5,
                         0,    0,    0,    0,    0,    0,    0,
                         5,   30,   75,  100,   75,   30,    5,
                         4,   24,   60,   80,   60,   24,    4,
                         1,    6,   15,   20,   15,    6,    1};

namespace rpp_sobel_filter
{

template<typename T>
inline void sobel_filter_bidirection_generic_tensor(T **srcPtrTemp, T *dstPtrTemp, Rpp32s columnIndex,
                                                     Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit,
                                                     Rpp32f *filterXTensor, Rpp32f *filterYTensor, Rpp32u channels = 1,
                                                     RpptImageBorderEdge padVertical = RpptImageBorderEdge::BOTTOM_EDGE,
                                                     RpptImageBorderEdge padHorizontal = RpptImageBorderEdge::RIGHT_EDGE)
{
    Rpp32f accumX = 0.0f;
    Rpp32f accumY = 0.0f;
    Rpp32s columnKernelLoopLimit = kernelSize;

    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);

    if (rowKernelLoopLimit < kernelSize || columnKernelLoopLimit < kernelSize)
    {
        for (int i = 0; i < kernelSize; i++)
        {
            Rpp32s rowOffset = (padVertical == RpptImageBorderEdge::TOP_EDGE)
                                ? std::max(0, static_cast<Rpp32s>(i + rowKernelLoopLimit - kernelSize)) // clamp top padded region
                                : std::min(rowKernelLoopLimit - 1, i); // clamp bottom padded region

            Rpp32s filterRowOffset = i * kernelSize;
            for (int j = 0; j < kernelSize; j++)
            {
                Rpp32s colOffset = (padHorizontal == RpptImageBorderEdge::LEFT_EDGE)
                                    ? std::max(0, static_cast<Rpp32s>(j + columnKernelLoopLimit - kernelSize)) // clamp left padded region
                                    : std::min(static_cast<Rpp32s>(columnKernelLoopLimit - 1), j); // clamp right padded region

                Rpp32f pixel;
                if constexpr (std::is_same<T, Rpp8s>::value)
                    pixel = static_cast<Rpp32f>(srcPtrTemp[rowOffset][colOffset * channels] + 128);
                else
                    pixel = static_cast<Rpp32f>(srcPtrTemp[rowOffset][colOffset * channels]);

                accumX = std::fmaf(pixel, filterXTensor[filterRowOffset + j], accumX);
                accumY = std::fmaf(pixel, filterYTensor[filterRowOffset + j], accumY);
            }
        }
    }
    else
    {
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                Rpp32f pixel;
                if constexpr (std::is_same<T, Rpp8s>::value)
                    pixel = static_cast<Rpp32f>(srcPtrTemp[i][j * channels] + 128);
                else
                    pixel = static_cast<Rpp32f>(srcPtrTemp[i][j * channels]);

                accumX = std::fmaf(pixel, filterXTensor[i * kernelSize + j], accumX);
                accumY = std::fmaf(pixel, filterYTensor[i * kernelSize + j], accumY);
            }
        }
    }

    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
    {
        accumX = RPPPIXELCHECK(accumX);
        accumY = RPPPIXELCHECK(accumY);
    }
    else
    {
        accumX = RPPPIXELCHECKF32(accumX);
        accumY = RPPPIXELCHECKF32(accumY);
    }

    Rpp32f accum = std::sqrt(std::fmaf(accumX, accumX, accumY * accumY));
    saturate_pixel(accum, dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 kernel, process them separately
template<typename T>
inline void process_left_border_columns_pln_pln(T **srcPtrTemp, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f *filterXTensor, Rpp32f *filterYTensor, RpptImageBorderEdge padVertical)
{
    for (int k = 0; k < padLength; k++)
    {
        sobel_filter_bidirection_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterXTensor, filterYTensor, 1, padVertical, RpptImageBorderEdge::LEFT_EDGE);
        dstPtrTemp++;
    }
}

} // namespace rpp_sobel_filter

using namespace rpp_sobel_filter;

template<typename T>
RppStatus sobel_filter_host_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u sobelType,
                                   Rpp32u kernelSize,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rpp::Handle& handle)
{
    if (srcDescPtr->layout != RpptLayout::NCHW)
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if (srcDescPtr->c != 1)
        return RPP_ERROR_INVALID_SRC_CHANNELS;

    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
#if __AVX2__
    __m256i pxMaskPln[7] = {avx_pxMaskRotate0To1, avx_pxMaskRotate0To2, avx_pxMaskRotate0To3, avx_pxMaskRotate0To4, avx_pxMaskRotate0To5, avx_pxMaskRotate0To6, avx_pxMaskRotate0To7};
#endif
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
        Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;
        bool combined = (sobelType == 2);
        Rpp32f *filter, *filterX, *filterY;

#if __AVX2__
        __m256 pMax, pMin;
        if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
        {
            pMax = avx_p255;
            pMin = avx_p0;
        }
        else
        {
            pMax = avx_p1;
            pMin = avx_p0;
        }
#endif

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + roi.xywhROI.xy.x;
        dstPtrChannel = dstPtrImage;
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (srcDescPtr->c == 1))
        {
            if (kernelSize == 3)
            {
                T *srcPtrRow[3], *dstPtrRow;
                for (int i = 0; i < 3; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;

                if (combined)
                {
                    filterX = sobel3x3X;
                    filterY = sobel3x3Y;
#if __AVX2__
                    __m256 pFilterX[9], pFilterY[9];
                    for (int i = 0; i < 9; i++)
                    {
                        pFilterX[i] = _mm256_set1_ps(filterX[i]);
                        pFilterY[i] = _mm256_set1_ps(filterY[i]);
                    }
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 14) * 14;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY, padVertical);
                        dstPtrTemp += padLength;
#if __AVX2__
                        Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ?  rowKernelLoopLimit - 1 : 0;
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                        {
                            __m256 pRow[6], pDst[2], pDstX[2], pDstY[2];
                            rpp_load_filter_NxN_pln_host<3>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                            for (int k = 0; k < 2; k++)
                            {
                                pDstX[k] = avx_p0;
                                pDstY[k] = avx_p0;
                                pDst[k] = avx_p0;
                            }
                            for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3; k++, filterIndex += 3, rowIndex += 2)
                            {
                                permute_blend_add_3x3<1, 3, 0, 1>(pDstX[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilterX[filterIndex], pxMaskPln);
                                permute_blend_add_3x3<1, 3, 0, 1>(pDstY[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilterY[filterIndex], pxMaskPln);
                                permute_blend_add_3x3<1, 3, 0, 1>(pDstX[1], pRow[rowIndex + 1], avx_p0, &pFilterX[filterIndex], pxMaskPln);
                                permute_blend_add_3x3<1, 3, 0, 1>(pDstY[1], pRow[rowIndex + 1], avx_p0, &pFilterY[filterIndex], pxMaskPln);
                            }
                            pDstX[0] = _mm256_min_ps(_mm256_max_ps(pDstX[0], pMin), pMax);
                            pDstY[0] = _mm256_min_ps(_mm256_max_ps(pDstY[0], pMin), pMax);
                            pDstX[0] = _mm256_mul_ps(pDstX[0], pDstX[0]);
                            pDstY[0] = _mm256_mul_ps(pDstY[0], pDstY[0]);
                            pDst[0] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[0], pDstY[0]));
                            pDst[0] = _mm256_min_ps(_mm256_max_ps(pDst[0], pMin), pMax);

                            pDstX[1] = _mm256_min_ps(_mm256_max_ps(pDstX[1], pMin), pMax);
                            pDstY[1] = _mm256_min_ps(_mm256_max_ps(pDstY[1], pMin), pMax);
                            pDstX[1] = _mm256_mul_ps(pDstX[1], pDstX[1]);
                            pDstY[1] = _mm256_mul_ps(pDstY[1], pDstY[1]);
                            pDst[1] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[1], pDstY[1]));
                            pDst[1] = _mm256_min_ps(_mm256_max_ps(pDst[1], pMin), pMax);

                            if constexpr (std::is_same<T, Rpp32f>::value)
                                rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 14);
                            dstPtrTemp += 14;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_bidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY, 1, padVertical);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
                else
                {
                    filter = (!sobelType) ? sobel3x3X : sobel3x3Y;
#if __AVX2__
                    __m256 pFilter[9];
                    for (int i = 0; i < 9; i++)
                        pFilter[i] = _mm256_set1_ps(filter[i]);
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 14) * 14;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
                        for (int k = 0; k < padLength; k++)
                        {
                            convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter, 1, padVertical, RpptImageBorderEdge::LEFT_EDGE);
                            dstPtrTemp++;
                        }
#if __AVX2__
                         Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ?  rowKernelLoopLimit - 1 : 0;
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                        {
                            __m256 pRow[6], pDst[2];
                            rpp_load_filter_NxN_pln_host<3>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                            pDst[0] = avx_p0;
                            pDst[1] = avx_p0;
                            for (int k = 0, filterIndex = 0, rowIndex = 0; k < 3; k++, filterIndex += 3, rowIndex += 2)
                            {
                                permute_blend_add_3x3<1, 3, 0, 1>(pDst[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilter[filterIndex], pxMaskPln);
                                permute_blend_add_3x3<1, 3, 0, 1>(pDst[1], pRow[rowIndex + 1], avx_p0, &pFilter[filterIndex], pxMaskPln);
                            }
                            pDst[0] = _mm256_min_ps(_mm256_max_ps(pDst[0], pMin), pMax);
                            pDst[1] = _mm256_min_ps(_mm256_max_ps(pDst[1], pMin), pMax);

                            if constexpr (std::is_same<T, Rpp32f>::value)
                                rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 14);
                            dstPtrTemp += 14;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter, 1, padVertical);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
            }
            else if (kernelSize == 5)
            {
                T *srcPtrRow[5], *dstPtrRow;
                for (int i = 0; i < 5; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;

                if (combined)
                {
                    filterX = sobel5x5X;
                    filterY = sobel5x5Y;
#if __AVX2__
                    __m256 pFilterX[25], pFilterY[25];
                    for (int i = 0; i < 25; i++)
                    {
                        pFilterX[i] = _mm256_set1_ps(filterX[i]);
                        pFilterY[i] = _mm256_set1_ps(filterY[i]);
                    }
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2],  srcPtrRow[3],  srcPtrRow[4]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY, padVertical);
                        dstPtrTemp += padLength;
#if __AVX2__
                         Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ?  rowKernelLoopLimit - 1 : 0;
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                        {
                            __m256 pRow[10], pDst[2], pDstX[2], pDstY[2];
                            rpp_load_filter_NxN_pln_host<5>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                            for (int k = 0; k < 2; k++)
                            {
                                pDstX[k] = avx_p0;
                                pDstY[k] = avx_p0;
                                pDst[k] = avx_p0;
                            }
                            for (int k = 0, filterIndex = 0, rowIndex = 0; k < 5; k++, filterIndex += 5, rowIndex += 2)
                            {
                                permute_blend_add_5x5_pln(pDstX[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilterX[filterIndex]);
                                permute_blend_add_5x5_pln(pDstY[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilterY[filterIndex]);
                                permute_blend_add_5x5_pln(pDstX[1], pRow[rowIndex + 1], avx_p0, &pFilterX[filterIndex]);
                                permute_blend_add_5x5_pln(pDstY[1], pRow[rowIndex + 1], avx_p0, &pFilterY[filterIndex]);
                            }
                            pDstX[0] = _mm256_min_ps(_mm256_max_ps(pDstX[0], pMin), pMax);
                            pDstY[0] = _mm256_min_ps(_mm256_max_ps(pDstY[0], pMin), pMax);
                            pDstX[0] = _mm256_mul_ps(pDstX[0], pDstX[0]);
                            pDstY[0] = _mm256_mul_ps(pDstY[0], pDstY[0]);
                            pDst[0] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[0], pDstY[0]));
                            pDst[0] = _mm256_min_ps(_mm256_max_ps(pDst[0], pMin), pMax);

                            pDstX[1] = _mm256_min_ps(_mm256_max_ps(pDstX[1], pMin), pMax);
                            pDstY[1] = _mm256_min_ps(_mm256_max_ps(pDstY[1], pMin), pMax);
                            pDstX[1] = _mm256_mul_ps(pDstX[1], pDstX[1]);
                            pDstY[1] = _mm256_mul_ps(pDstY[1], pDstY[1]);
                            pDst[1] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[1], pDstY[1]));
                            pDst[1] = _mm256_min_ps(_mm256_max_ps(pDst[1], pMin), pMax);

                            if constexpr (std::is_same<T, Rpp32f>::value)
                                rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);

                            // In each pass, convolution filter is applied 12 times
                            increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                            dstPtrTemp += 12;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_bidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY, 1, padVertical);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
                else
                {
                    filter = (!sobelType) ? sobel5x5X : sobel5x5Y;
#if __AVX2__
                    __m256 pFilter[25];
                    for (int i = 0; i < 25; i++)
                        pFilter[i] = _mm256_set1_ps(filter[i]);
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
                        for (int k = 0; k < padLength; k++)
                        {
                            convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter, 1, padVertical, RpptImageBorderEdge::LEFT_EDGE);
                            dstPtrTemp++;
                        }
#if __AVX2__
                         Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ?  rowKernelLoopLimit - 1 : 0;
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                        {
                            __m256 pRow[10], pDst[2];
                            rpp_load_filter_NxN_pln_host<5>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                            pDst[0] = avx_p0;
                            pDst[1] = avx_p0;
                            for (int k = 0, filterIndex = 0, rowIndex = 0; k < 5; k++, filterIndex += 5, rowIndex += 2)
                            {
                                permute_blend_add_5x5_pln(pDst[0], pRow[rowIndex], pRow[rowIndex + 1], &pFilter[filterIndex]);
                                permute_blend_add_5x5_pln(pDst[1], pRow[rowIndex + 1], avx_p0, &pFilter[filterIndex]);
                            }
                            pDst[0] = _mm256_min_ps(_mm256_max_ps(pDst[0], pMin), pMax);
                            pDst[1] = _mm256_min_ps(_mm256_max_ps(pDst[1], pMin), pMax);
                            
                            if constexpr (std::is_same<T, Rpp32f>::value)
                                rpp_store16_f32_to_f32_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                rpp_store16_f32_to_f16_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                            dstPtrTemp += 12;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter, 1, padVertical);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
            }
            else if (kernelSize == 7)
            {
                T *srcPtrRow[7], *dstPtrRow;
                for (int i = 0; i < 7; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;

                if (combined)
                {
                    filterX = sobel7x7X;
                    filterY = sobel7x7Y;
#if __AVX2__
                    __m256 pFilterX[49], pFilterY[49];
                    for (int i = 0; i < 49; i++)
                    {
                        pFilterX[i] = _mm256_set1_ps(filterX[i]);
                        pFilterY[i] = _mm256_set1_ps(filterY[i]);
                    }
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[7] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2],  srcPtrRow[3],  srcPtrRow[4], srcPtrRow[5], srcPtrRow[6]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY, padVertical);
                        dstPtrTemp += padLength;
#if __AVX2__
                         Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ?  rowKernelLoopLimit - 1 : 0;
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            __m256 pRow[14], pDst, pDstX, pDstY;
                            rpp_load_filter_NxN_pln_host<7>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                            pDstX = avx_p0;
                            pDstY = avx_p0;
                            pDst = avx_p0;
                            for (int k = 0, filterIndex = 0, rowIndex = 0; k < 7; k++, filterIndex += 7, rowIndex += 2)
                            {
                                permute_blend_add_7x7_pln(pDstX, &pRow[rowIndex], &pFilterX[filterIndex]);
                                permute_blend_add_7x7_pln(pDstY, &pRow[rowIndex], &pFilterY[filterIndex]);
                            }
                            pDstX = _mm256_min_ps(_mm256_max_ps(pDstX, pMin), pMax);
                            pDstY = _mm256_min_ps(_mm256_max_ps(pDstY, pMin), pMax);
                            pDstX = _mm256_mul_ps(pDstX, pDstX);
                            pDstY = _mm256_mul_ps(pDstY, pDstY);
                            pDst =  _mm256_sqrt_ps(_mm256_add_ps(pDstX, pDstY));
                            pDst = _mm256_min_ps(_mm256_max_ps(pDst, pMin), pMax);

                            // convert result from pln to pkd format and store in output buffer
                            if constexpr (std::is_same<T, Rpp32f>::value)
                                _mm256_storeu_ps(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store8_f32_to_u8_avx(dstPtrTemp, &pDst);

                            // In each pass, convolution filter is applied 8 times
                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            dstPtrTemp += 8;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            sobel_filter_bidirection_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY, 1, padVertical);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
                else
                {
                    filter = (!sobelType) ? sobel7x7X : sobel7x7Y;
#if __AVX2__
                    __m256 pFilter[49];
                    for (int i = 0; i < 49; i++)
                        pFilter[i] = _mm256_set1_ps(filter[i]);
#endif
                    /* exclude 2 * padLength number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                    Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[7] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4], srcPtrRow[5], srcPtrRow[6]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        RpptImageBorderEdge padVertical = i < padLength ? RpptImageBorderEdge::TOP_EDGE : RpptImageBorderEdge::BOTTOM_EDGE;
                        for (int k = 0; k < padLength; k++)
                        {
                            convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter, 1, padVertical, RpptImageBorderEdge::LEFT_EDGE);
                            dstPtrTemp++;
                        }
#if __AVX2__
                         Rpp32s padIndex = (padVertical == RpptImageBorderEdge::BOTTOM_EDGE) ?  rowKernelLoopLimit - 1 : 0;
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            __m256 pRow[14], pDst;
                            rpp_load_filter_NxN_pln_host<7>(pRow, srcPtrTemp, rowKernelLoopLimit, padIndex);
                            pDst = avx_p0;
                            for (int k = 0, filterIndex = 0, rowIndex = 0; k < 7; k++, filterIndex += 7, rowIndex += 2)
                                permute_blend_add_7x7_pln(pDst, &pRow[rowIndex], &pFilter[filterIndex]);
                            pDst = _mm256_min_ps(_mm256_max_ps(pDst, pMin), pMax);
                            
                            // convert result from pln to pkd format and store in output buffer
                            if constexpr (std::is_same<T, Rpp32f>::value)
                                _mm256_storeu_ps(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store8_f32_to_u8_avx(dstPtrTemp, &pDst);

                            // In each pass, convolution filter is applied 8 times
                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            dstPtrTemp += 8;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            convolution_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter, 1, padVertical);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus sobel_filter_host_tensor<Rpp8u>(Rpp8u*,
                                                   RpptDescPtr,
                                                   Rpp8u*,
                                                   RpptDescPtr,
                                                   Rpp32u,
                                                   Rpp32u,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus sobel_filter_host_tensor<Rpp8s>(Rpp8s*,
                                                   RpptDescPtr,
                                                   Rpp8s*,
                                                   RpptDescPtr,
                                                   Rpp32u,
                                                   Rpp32u,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus sobel_filter_host_tensor<Rpp16f>(Rpp16f*,
                                                    RpptDescPtr,
                                                    Rpp16f*,
                                                    RpptDescPtr,
                                                    Rpp32u,
                                                    Rpp32u,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);

template RppStatus sobel_filter_host_tensor<Rpp32f>(Rpp32f*,
                                                    RpptDescPtr,
                                                    Rpp32f*,
                                                    RpptDescPtr,
                                                    Rpp32u,
                                                    Rpp32u,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);
