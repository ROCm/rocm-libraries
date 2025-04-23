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

alignas(64) const Rpp32f sch_mat[16] = {0.701f, -0.299f, -0.300f, 0.0f, -0.587f, 0.413f, -0.588f, 0.0f, -0.114f, -0.114f, 0.886f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
alignas(64) const Rpp32f ssh_mat[16] = {0.168f, -0.328f, 1.250f, 0.0f, 0.330f, 0.035f, -1.050f, 0.0f, -0.497f, 0.292f, -0.203f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

inline void compute_color_jitter_ctm_host(Rpp32f brightnessParam, Rpp32f contrastParam, Rpp32f hueParam, Rpp32f saturationParam, Rpp32f *ctm)
{
    contrastParam += 1.0f;

    alignas(64) Rpp32f hue_saturation_matrix[16] = {RGB_TO_GREY_WEIGHT_RED, RGB_TO_GREY_WEIGHT_RED, RGB_TO_GREY_WEIGHT_RED, 0.0f, RGB_TO_GREY_WEIGHT_GREEN, RGB_TO_GREY_WEIGHT_GREEN, RGB_TO_GREY_WEIGHT_GREEN, 0.0f, RGB_TO_GREY_WEIGHT_BLUE, RGB_TO_GREY_WEIGHT_BLUE, RGB_TO_GREY_WEIGHT_BLUE, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    alignas(64) Rpp32f brightness_contrast_matrix[16] = {contrastParam, 0.0f, 0.0f, 0.0f, 0.0f, contrastParam, 0.0f, 0.0f, 0.0f, 0.0f, contrastParam, 0.0f, brightnessParam, brightnessParam, brightnessParam, 1.0f};

    Rpp32f sch = saturationParam * cos(hueParam * PI_OVER_180);
    Rpp32f ssh = saturationParam * sin(hueParam * PI_OVER_180);

    __m128 psch = _mm_set1_ps(sch);
    __m128 pssh = _mm_set1_ps(ssh);
    __m128 p0, p1, p2;

    for (int i = 0; i < 16; i+=4)
    {
        p0 = _mm_loadu_ps(hue_saturation_matrix + i);
        p1 = _mm_loadu_ps(sch_mat + i);
        p2 = _mm_loadu_ps(ssh_mat + i);
        p0 = _mm_fmadd_ps(psch, p1, _mm_fmadd_ps(pssh, p2, p0));
        _mm_storeu_ps(hue_saturation_matrix + i, p0);
    }

    fast_matmul4x4_sse(hue_saturation_matrix, brightness_contrast_matrix, ctm);
}

inline void compute_color_jitter_48_host(__m128 *p, __m128 *pCtm)
{
    __m128 pResult[3];

    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[0], _mm_fmadd_ps(p[4], pCtm[1], _mm_fmadd_ps(p[8], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R0-R3
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[4], _mm_fmadd_ps(p[4], pCtm[5], _mm_fmadd_ps(p[8], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G0-G3
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[0], pCtm[8], _mm_fmadd_ps(p[4], pCtm[9], _mm_fmadd_ps(p[8], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B0-B3
    p[0] = pResult[0];    // color_jitter adjustment R0-R3
    p[4] = pResult[1];    // color_jitter adjustment G0-G3
    p[8] = pResult[2];    // color_jitter adjustment B0-B3
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[0], _mm_fmadd_ps(p[5], pCtm[1], _mm_fmadd_ps(p[9], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R4-R7
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[4], _mm_fmadd_ps(p[5], pCtm[5], _mm_fmadd_ps(p[9], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G4-G7
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[1], pCtm[8], _mm_fmadd_ps(p[5], pCtm[9], _mm_fmadd_ps(p[9], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B4-B7
    p[1] = pResult[0];    // color_jitter adjustment R4-R7
    p[5] = pResult[1];    // color_jitter adjustment G4-G7
    p[9] = pResult[2];    // color_jitter adjustment B4-B7
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[0], _mm_fmadd_ps(p[6], pCtm[1], _mm_fmadd_ps(p[10], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R8-R11
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[4], _mm_fmadd_ps(p[6], pCtm[5], _mm_fmadd_ps(p[10], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G8-G11
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[2], pCtm[8], _mm_fmadd_ps(p[6], pCtm[9], _mm_fmadd_ps(p[10], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B8-B11
    p[2] = pResult[0];    // color_jitter adjustment R8-R11
    p[6] = pResult[1];    // color_jitter adjustment G8-G11
    p[10] = pResult[2];    // color_jitter adjustment B8-B11
    pResult[0] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[0], _mm_fmadd_ps(p[7], pCtm[1], _mm_fmadd_ps(p[11], pCtm[2], pCtm[3]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment R12-R15
    pResult[1] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[4], _mm_fmadd_ps(p[7], pCtm[5], _mm_fmadd_ps(p[11], pCtm[6], pCtm[7]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment G12-G15
    pResult[2] = _mm_round_ps(_mm_fmadd_ps(p[3], pCtm[8], _mm_fmadd_ps(p[7], pCtm[9], _mm_fmadd_ps(p[11], pCtm[10], pCtm[11]))), (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));    // color_jitter adjustment B12-B15
    p[3] = pResult[0];    // color_jitter adjustment R12-R15
    p[7] = pResult[1];    // color_jitter adjustment G12-G15
    p[11] = pResult[2];    // color_jitter adjustment B12-B15
}

inline void compute_color_jitter_12_host(__m128 *p, __m128 *pCtm)
{
    __m128 pResult[3];

    pResult[0] = _mm_fmadd_ps(p[0], pCtm[0], _mm_fmadd_ps(p[1], pCtm[1], _mm_fmadd_ps(p[2], pCtm[2], pCtm[3])));    // color_jitter adjustment R0-R3
    pResult[1] = _mm_fmadd_ps(p[0], pCtm[4], _mm_fmadd_ps(p[1], pCtm[5], _mm_fmadd_ps(p[2], pCtm[6], pCtm[7])));    // color_jitter adjustment G0-G3
    pResult[2] = _mm_fmadd_ps(p[0], pCtm[8], _mm_fmadd_ps(p[1], pCtm[9], _mm_fmadd_ps(p[2], pCtm[10], pCtm[11])));    // color_jitter adjustment B0-B3
    p[0] = pResult[0];    // color_jitter adjustment R0-R3
    p[1] = pResult[1];    // color_jitter adjustment G0-G3
    p[2] = pResult[2];    // color_jitter adjustment B0-B3
}

RppStatus color_jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp8u *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *brightnessTensor,
                                         Rpp32f *contrastTensor,
                                         Rpp32f *hueTensor,
                                         Rpp32f *saturationTensor,
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

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        alignas(64) Rpp32f ctm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::round(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::round(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::round(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]));

                    srcPtrTemp+=3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(std::round(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(std::round(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(std::round(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(std::round(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
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

RppStatus color_jitter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp32f *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *brightnessTensor,
                                           Rpp32f *contrastTensor,
                                           Rpp32f *hueTensor,
                                           Rpp32f *saturationTensor,
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

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        alignas(64) Rpp32f ctm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    *dstPtrTempG = RPPPIXELCHECKF32(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    *dstPtrTempB = RPPPIXELCHECKF32(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    *dstPtrTempG = RPPPIXELCHECKF32(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    *dstPtrTempB = RPPPIXELCHECKF32(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_jitter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp16f *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *brightnessTensor,
                                           Rpp32f *contrastTensor,
                                           Rpp32f *hueTensor,
                                           Rpp32f *saturationTensor,
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

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        alignas(64) Rpp32f ctm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    compute_color_jitter_12_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus color_jitter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp8s *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *brightnessTensor,
                                         Rpp32f *contrastTensor,
                                         Rpp32f *hueTensor,
                                         Rpp32f *saturationTensor,
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

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        alignas(64) Rpp32f ctm[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f srcPtrTempI8[3];
                    srcPtrTempI8[0] = (Rpp32f)srcPtrTemp[0] + 128;
                    srcPtrTempI8[1] = (Rpp32f)srcPtrTemp[1] + 128;
                    srcPtrTempI8[2] = (Rpp32f)srcPtrTemp[2] + 128;

                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[0] * srcPtrTempI8[0] + ctm[1] * srcPtrTempI8[1] + ctm[2] * srcPtrTempI8[2] + ctm[3]) - 128);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[4] * srcPtrTempI8[0] + ctm[5] * srcPtrTempI8[1] + ctm[6] * srcPtrTempI8[2] + ctm[7]) - 128);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[8] * srcPtrTempI8[0] + ctm[9] * srcPtrTempI8[1] + ctm[10] * srcPtrTempI8[2] + ctm[11]) - 128);

                    srcPtrTemp+=3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcPtrTempI8[3];
                    srcPtrTempI8[0] = (Rpp32f)*srcPtrTempR + 128;
                    srcPtrTempI8[1] = (Rpp32f)*srcPtrTempG + 128;
                    srcPtrTempI8[2] = (Rpp32f)*srcPtrTempB + 128;

                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[0] * srcPtrTempI8[0] + ctm[1] * srcPtrTempI8[1] + ctm[2] * srcPtrTempI8[2] + ctm[3]) - 128);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[4] * srcPtrTempI8[0] + ctm[5] * srcPtrTempI8[1] + ctm[6] * srcPtrTempI8[2] + ctm[7]) - 128);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[8] * srcPtrTempI8[0] + ctm[9] * srcPtrTempI8[1] + ctm[10] * srcPtrTempI8[2] + ctm[11]) - 128);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f srcPtrTempI8[3];
                    srcPtrTempI8[0] = (Rpp32f)srcPtrTemp[0] + 128;
                    srcPtrTempI8[1] = (Rpp32f)srcPtrTemp[1] + 128;
                    srcPtrTempI8[2] = (Rpp32f)srcPtrTemp[2] + 128;

                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[0] * srcPtrTempI8[0] + ctm[1] * srcPtrTempI8[1] + ctm[2] * srcPtrTempI8[2] + ctm[3]) - 128);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[4] * srcPtrTempI8[0] + ctm[5] * srcPtrTempI8[1] + ctm[6] * srcPtrTempI8[2] + ctm[7]) - 128);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[8] * srcPtrTempI8[0] + ctm[9] * srcPtrTempI8[1] + ctm[10] * srcPtrTempI8[2] + ctm[11]) - 128);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_color_jitter_48_host(p, pCtm);    // color_jitter adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcPtrTempI8[3];
                    srcPtrTempI8[0] = (Rpp32f)*srcPtrTempR + 128;
                    srcPtrTempI8[1] = (Rpp32f)*srcPtrTempG + 128;
                    srcPtrTempI8[2] = (Rpp32f)*srcPtrTempB + 128;

                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[0] * srcPtrTempI8[0] + ctm[1] * srcPtrTempI8[1] + ctm[2] * srcPtrTempI8[2] + ctm[3]) - 128);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[4] * srcPtrTempI8[0] + ctm[5] * srcPtrTempI8[1] + ctm[6] * srcPtrTempI8[2] + ctm[7]) - 128);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8(std::round(ctm[8] * srcPtrTempI8[0] + ctm[9] * srcPtrTempI8[1] + ctm[10] * srcPtrTempI8[2] + ctm[11]) - 128);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
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
