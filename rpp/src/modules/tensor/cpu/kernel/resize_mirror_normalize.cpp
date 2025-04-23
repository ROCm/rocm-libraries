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

inline void compute_rmn_24_host(__m256 *p, __m256 *pRMNParams)
{
    p[0] = _mm256_mul_ps(_mm256_sub_ps(p[0], pRMNParams[0]), pRMNParams[1]);
    p[1] = _mm256_mul_ps(_mm256_sub_ps(p[1], pRMNParams[2]), pRMNParams[3]);
    p[2] = _mm256_mul_ps(_mm256_sub_ps(p[2], pRMNParams[4]), pRMNParams[5]);
}

inline void compute_rmn_8_host(__m256 *p, __m256 *pRMNParams)
{
    p[0] = _mm256_mul_ps(_mm256_sub_ps(p[0], pRMNParams[0]), pRMNParams[1]);
}

RppStatus resize_mirror_normalize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8u *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptImagePatchPtr dstImgSize,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
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

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 2;
        Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 2) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        //Set non ROI pixels to zero
        memset(dstPtrImage, (Rpp8u)0, (size_t)dstDescPtr->strides.nStride);

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS], pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
        Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS], bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pRMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c];
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pRMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pRMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];
        Rpp32u width = dstImgSize[batchCount].width;

        __m256 pDstLocInit =  avx_pDstLocInit;
        auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;
        if(mirrorFlag)
        {
            pDstLocInit =  _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5, width - 6, width - 7, width - 8);
            computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
        }

        // Resize Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset); // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*dstPtrTempR - mean[0]) * invStdDev[0]))));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*dstPtrTempG - mean[1]) * invStdDev[1]))));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*dstPtrTempB - mean[2]) * invStdDev[2]))));
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (dstPtrTemp[0] - mean[0]) * invStdDev[0]))));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (dstPtrTemp[1] - mean[1]) * invStdDev[1]))));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (dstPtrTemp[2] - mean[2]) * invStdDev[2]))));
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (dstPtrTemp[0] - mean[0]) * invStdDev[1]))));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (dstPtrTemp[1] - mean[1]) * invStdDev[0]))));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (dstPtrTemp[2] - mean[2]) * invStdDev[2]))));
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                Rpp8u *dstPtrTempChn;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        compute_rmn_8_host(&pDst, &pRMNParams[2 * c]);
                        rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    dstPtrTempChn = dstPtrTemp;
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        *dstPtrTempChn = (Rpp8u) RPPPIXELCHECK(std::nearbyintf((((Rpp32f) (*dstPtrTempChn - mean[c]) * invStdDev[c]))));
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

RppStatus resize_mirror_normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp32f *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      RpptImagePatchPtr dstImgSize,
                                                      Rpp32f *meanTensor,
                                                      Rpp32f *stdDevTensor,
                                                      Rpp32u *mirrorTensor,
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

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        //Set non ROI pixels to zero
        memset(dstPtrImage, (Rpp32f)0, (size_t)dstDescPtr->strides.nStride);

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS], pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
        Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS], bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pRMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c] * ONE_OVER_255;
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pRMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pRMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];
        Rpp32u width = dstImgSize[batchCount].width;

        __m256 pDstLocInit =  avx_pDstLocInit;
        auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;

        if(mirrorFlag)
        {
            pDstLocInit =  _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5, width - 6, width - 7, width - 8);
            computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
        }

        // Resize Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    *dstPtrTempR = RPPPIXELCHECKF32((*dstPtrTempR - mean[0]) * invStdDev[0]);
                    *dstPtrTempG = RPPPIXELCHECKF32((*dstPtrTempG - mean[1]) * invStdDev[1]);
                    *dstPtrTempB = RPPPIXELCHECKF32((*dstPtrTempB - mean[2]) * invStdDev[2]);
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = RPPPIXELCHECKF32((dstPtrTemp[0] - mean[0]) * invStdDev[0]);
                    dstPtrTemp[1] = RPPPIXELCHECKF32((dstPtrTemp[1] - mean[1]) * invStdDev[1]);
                    dstPtrTemp[2] = RPPPIXELCHECKF32((dstPtrTemp[2] - mean[2]) * invStdDev[2]);
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the col row location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = RPPPIXELCHECKF32((dstPtrTemp[0] - mean[0]) * invStdDev[0]);
                    dstPtrTemp[1] = RPPPIXELCHECKF32((dstPtrTemp[1] - mean[1]) * invStdDev[1]);
                    dstPtrTemp[2] = RPPPIXELCHECKF32((dstPtrTemp[2] - mean[2]) * invStdDev[2]);
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);    // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                        compute_rmn_8_host(&pDst, &pRMNParams[2 * c]);
                        rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempChn, pDst); // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        *dstPtrTempChn = RPPPIXELCHECKF32((*dstPtrTempChn - mean[c]) * invStdDev[c]);
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

RppStatus resize_mirror_normalize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp16f *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      RpptImagePatchPtr dstImgSize,
                                                      Rpp32f *meanTensor,
                                                      Rpp32f *stdDevTensor,
                                                      Rpp32u *mirrorTensor,
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

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Set non ROI pixels to zero
        memset(dstPtrImage, (Rpp16f)0, (size_t)dstDescPtr->strides.nStride);

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS], pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
        Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS], bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pRMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c] * ONE_OVER_255;
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pRMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pRMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];
        Rpp32u width = dstImgSize[batchCount].width;

        __m256 pDstLocInit =  avx_pDstLocInit;
        auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;

        if(mirrorFlag)
        {
            pDstLocInit =  _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5, width - 6, width - 7, width - 8);
            computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
        }

        // Resize Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp16f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*dstPtrTempR - mean[0]) * invStdDev[0]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*dstPtrTempG - mean[1]) * invStdDev[1]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*dstPtrTempB - mean[2]) * invStdDev[2]);
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(dstPtrTemp[0] - mean[0]) * invStdDev[0]);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(dstPtrTemp[1] - mean[1]) * invStdDev[1]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(dstPtrTemp[2] - mean[2]) * invStdDev[2]);
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the col row location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(dstPtrTemp[0] - mean[0]) * invStdDev[0]);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(dstPtrTemp[1] - mean[1]) * invStdDev[1]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(dstPtrTemp[2] - mean[2]) * invStdDev[2]);
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);    // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                        compute_rmn_8_host(&pDst, &pRMNParams[2 * c]);
                        rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTempChn, pDst); // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        *dstPtrTempChn = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)(*dstPtrTempChn - mean[c]) * invStdDev[c]);
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

RppStatus resize_mirror_normalize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8s *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptImagePatchPtr dstImgSize,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
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

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        //Set non ROI pixels to zero
        memset(dstPtrImage, (Rpp8s)0, (size_t)dstDescPtr->strides.nStride);

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);

        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS], pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
        Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS], bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pRMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c];
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pRMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pRMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];
        Rpp32u width = dstImgSize[batchCount].width;

        __m256 pDstLocInit =  avx_pDstLocInit;
        auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;
        if(mirrorFlag)
        {
            pDstLocInit =  _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5, width - 6, width - 7, width - 8);
            computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
        }

        // Resize Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp8s *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset); // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*dstPtrTempR + 128 - mean[0]) * invStdDev[0] - 128)));
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*dstPtrTempG + 128 - mean[1]) * invStdDev[1] - 128)));
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*dstPtrTempB + 128 - mean[2]) * invStdDev[2] - 128)));
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8s *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (dstPtrTemp[0] + 128 - mean[0]) * invStdDev[0] - 128)));
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (dstPtrTemp[1] + 128 - mean[1]) * invStdDev[1] - 128)));
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (dstPtrTemp[2] + 128 - mean[2]) * invStdDev[2] - 128)));
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8s *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (dstPtrTemp[0] + 128 - mean[0]) * invStdDev[0] - 128)));
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (dstPtrTemp[1] + 128 - mean[1]) * invStdDev[1] - 128)));
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (dstPtrTemp[2] + 128 - mean[2]) * invStdDev[2] - 128)));
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp8s *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                Rpp8s *dstPtrTempChn;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        compute_rmn_8_host(&pDst, &pRMNParams[2 * c]);
                        rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }

                    dstPtrTemp += vectorIncrementPerChannel;
                }

                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    dstPtrTempChn = dstPtrTemp;
                    int dstLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(dstLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        *dstPtrTempChn = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*dstPtrTempChn + 128 - mean[c]) * invStdDev[c] - 128)));
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

RppStatus resize_mirror_normalize_u8_f32_host_tensor(Rpp8u *srcPtr,
                                                     RpptDescPtr srcDescPtr,
                                                     Rpp32f *dstPtr,
                                                     RpptDescPtr dstDescPtr,
                                                     RpptImagePatchPtr dstImgSize,
                                                     Rpp32f *meanTensor,
                                                     Rpp32f *stdDevTensor,
                                                     Rpp32u *mirrorTensor,
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

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8u *srcPtrChannel, *srcPtrImage;
        Rpp32f *dstPtrChannel, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

         //Set non ROI pixels to zero
        memset(dstPtrImage, (Rpp32f)0, (size_t)dstDescPtr->strides.nStride);

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS], pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
        Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS], bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pRMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c];
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pRMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pRMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];
        Rpp32u width = dstImgSize[batchCount].width;

        __m256 pDstLocInit =  avx_pDstLocInit;
        auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;
        if(mirrorFlag)
        {
            pDstLocInit =  _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5, width - 6, width - 7, width - 8);
            computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
        }

        // Resize Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset); // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int srcLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(srcLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    *dstPtrTempR = ((Rpp32f) (*dstPtrTempR - mean[0]) * invStdDev[0]);
                    *dstPtrTempG = ((Rpp32f) (*dstPtrTempG - mean[1]) * invStdDev[1]);
                    *dstPtrTempB = ((Rpp32f) (*dstPtrTempB - mean[2]) * invStdDev[2]);
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int srcLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(srcLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = ((Rpp32f) (dstPtrTemp[0] - mean[0]) * invStdDev[0]);
                    dstPtrTemp[1] = ((Rpp32f) (dstPtrTemp[1] - mean[1]) * invStdDev[1]);
                    dstPtrTemp[2] = ((Rpp32f) (dstPtrTemp[2] - mean[2]) * invStdDev[2]);
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int srcLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(srcLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = ((Rpp32f) (dstPtrTemp[0] - mean[0]) * invStdDev[1]);
                    dstPtrTemp[1] = ((Rpp32f) (dstPtrTemp[1] - mean[1]) * invStdDev[0]);
                    dstPtrTemp[2] = ((Rpp32f) (dstPtrTemp[2] - mean[2]) * invStdDev[2]);
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                Rpp32f *dstPtrTempChn;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256i pSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        __m256i pxSrcLoc;
                        rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        compute_rmn_8_host(&pDst, &pRMNParams[2 * c]);
                        rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    dstPtrTempChn = dstPtrTemp;
                    int srcLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(srcLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        *dstPtrTempChn = ((Rpp32f) (*dstPtrTempChn - mean[c]) * invStdDev[c]);
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

RppStatus resize_mirror_normalize_u8_f16_host_tensor(Rpp8u *srcPtr,
                                                     RpptDescPtr srcDescPtr,
                                                     Rpp16f *dstPtr,
                                                     RpptDescPtr dstDescPtr,
                                                     RpptImagePatchPtr dstImgSize,
                                                     Rpp32f *meanTensor,
                                                     Rpp32f *stdDevTensor,
                                                     Rpp32u *mirrorTensor,
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

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32f hOffset = (hRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - INTERP_BILINEAR_KERNEL_RADIUS;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8u *srcPtrChannel, *srcPtrImage;
        Rpp16f *dstPtrChannel, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Set non ROI pixels to zero
        memset(dstPtrImage, (Rpp16f)0, (size_t)dstDescPtr->strides.nStride);

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[INTERP_BILINEAR_NUM_COEFFS], pBilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS], pDstLoc;
        Rpp32f weightParams[INTERP_BILINEAR_NUM_COEFFS], bilinearCoeffs[INTERP_BILINEAR_NUM_COEFFS];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pRMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c];
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pRMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pRMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];
        Rpp32u width = dstImgSize[batchCount].width;

        __m256 pDstLocInit =  avx_pDstLocInit;
        auto computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_avx;
        if(mirrorFlag)
        {
            pDstLocInit =  _mm256_setr_ps(width - 1, width - 2, width - 3, width - 4, width - 5, width - 6, width - 7, width - 8);
            computeFnSrcLocAvx = &compute_resize_bilinear_src_loc_and_weights_mirror_avx;
        }

        // Resize Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset); // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int srcLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(srcLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    *dstPtrTempR = (Rpp16f) (((Rpp32f) (*dstPtrTempR - mean[0]) * invStdDev[0]));
                    *dstPtrTempG = (Rpp16f) (((Rpp32f) (*dstPtrTempG - mean[1]) * invStdDev[1]));
                    *dstPtrTempB = (Rpp16f) (((Rpp32f) (*dstPtrTempB - mean[2]) * invStdDev[2]));
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int srcLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(srcLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp16f) (((Rpp32f) (dstPtrTemp[0] - mean[0]) * invStdDev[0]));
                    dstPtrTemp[1] = (Rpp16f) (((Rpp32f) (dstPtrTemp[1] - mean[1]) * invStdDev[1]));
                    dstPtrTemp[2] = (Rpp16f) (((Rpp32f) (dstPtrTemp[2] - mean[2]) * invStdDev[2]));
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    compute_rmn_24_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    int srcLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(srcLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp16f) (((Rpp32f) (dstPtrTemp[0] - mean[0]) * invStdDev[1]));
                    dstPtrTemp[1] = (Rpp16f) (((Rpp32f) (dstPtrTemp[1] - mean[1]) * invStdDev[0]));
                    dstPtrTemp[2] = (Rpp16f) (((Rpp32f) (dstPtrTemp[2] - mean[2]) * invStdDev[2]));
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                Rpp16f *dstPtrTempChn;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    computeFnSrcLocAvx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, false); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        compute_rmn_8_host(&pDst, &pRMNParams[2 * c]);
                        rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    dstPtrTempChn = dstPtrTemp;
                    int srcLoc = (mirrorFlag) ? width - 1 - vectorLoopCount : vectorLoopCount;
                    compute_resize_bilinear_src_loc_and_weights(srcLoc, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * INTERP_BILINEAR_KERNEL_SIZE], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        *dstPtrTempChn = (Rpp16f) (((Rpp32f) (*dstPtrTempChn - mean[c]) * invStdDev[c]));
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
