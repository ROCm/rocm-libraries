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

template <typename T>
RppStatus random_erase_host_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptRoiLtrb *anchorBoxInfoTensor,
                                   T *noiseBuffer,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        RpptRoiLtrb anchorBoxInfo = anchorBoxInfoTensor[batchCount];
        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier * sizeof(T);

        // Erase with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            T *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                for(int j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    *dstPtrTempR++ = srcPtrTemp[0];
                    *dstPtrTempG++ = srcPtrTemp[1];
                    *dstPtrTempB++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }

            Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
            Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
            Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.x, x1, roi.xywhROI.roiWidth));
            Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.y, y1, roi.xywhROI.roiHeight));

            // Erase only if the box is within the ROI
            if (x1 > x2 || y1 > y2)
                continue;

            Rpp32u boxHeight = y2 - y1 + 1;
            Rpp32u boxWidth = x2 - x1 + 1;
            Rpp32u pixelLocation = (y1 * dstDescPtr->strides.hStride) + (x1 * dstDescPtr->strides.wStride);

            T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrImage + pixelLocation;
            dstPtrTempG = dstPtrTempR + dstDescPtr->strides.cStride;
            dstPtrTempB = dstPtrTempG + dstDescPtr->strides.cStride;
            for (int i = 0; i < boxHeight; i++)
            {
                Rpp32u noiseRowOffset = ((y1 + i + batchCount) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * RANDOM_ERASE_NOISE_BUFFER_SIDE * 3;
                for (int j = 0; j < boxWidth; j++)
                {
                    Rpp32u noiseIdx = noiseRowOffset + ((x1 + j) % RANDOM_ERASE_NOISE_BUFFER_SIDE * 3);

                    dstPtrTempR[j] = noiseBuffer[noiseIdx];
                    dstPtrTempG[j] = noiseBuffer[noiseIdx + 1];
                    dstPtrTempB[j] = noiseBuffer[noiseIdx + 2];
                }
                dstPtrTempR += dstDescPtr->strides.hStride;
                dstPtrTempG += dstDescPtr->strides.hStride;
                dstPtrTempB += dstDescPtr->strides.hStride;
            }
        }
    
        // Erase with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            // To copy ROI region in Image
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcRowR = srcPtrRowR;
                T *srcRowG = srcPtrRowG;
                T *srcRowB = srcPtrRowB;
                T *dstPtrTemp = dstPtrRow;

                for (int j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    dstPtrTemp[0] = *srcRowR++;
                    dstPtrTemp[1] = *srcRowG++;
                    dstPtrTemp[2] = *srcRowB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
            Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
            Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.x, x1, roi.xywhROI.roiWidth));
            Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.y, y1, roi.xywhROI.roiHeight));

            if (x1 > x2 || y1 > y2)
                continue;

            Rpp32u boxHeight = y2 - y1 + 1;
            Rpp32u boxWidth = x2 - x1 + 1;
            Rpp32u pixelLocation = (y1 * dstDescPtr->strides.hStride) + (x1 * dstDescPtr->strides.wStride);
            T *dstPtrTemp;
            dstPtrTemp = dstPtrImage + pixelLocation;

            for (int i = 0; i < boxHeight; i++)
            {
                Rpp32u noiseRowOffset = ((y1 + i + batchCount) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * RANDOM_ERASE_NOISE_BUFFER_SIDE;
                T *dstPtrRow = dstPtrTemp;
                for (int j = 0; j < boxWidth; j++)
                {
                    Rpp32u noiseIdx = (noiseRowOffset + ((x1 + j) % RANDOM_ERASE_NOISE_BUFFER_SIDE)) * 3;
                    dstPtrRow[0] = noiseBuffer[noiseIdx];     // R
                    dstPtrRow[1] = noiseBuffer[noiseIdx + 1]; // G
                    dstPtrRow[2] = noiseBuffer[noiseIdx + 2]; // B
                    dstPtrRow += dstDescPtr->c;
                }
                dstPtrTemp += dstDescPtr->strides.hStride;
            }
        }
        // Erase without fused output-layout toggle 3 channel(NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            // To copy ROI region in Image
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                T *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrRow, srcPtrRow, bufferLength);
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }

            Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
            Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
            Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.x, x1, roi.xywhROI.roiWidth));
            Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.y, y1, roi.xywhROI.roiHeight));

            // Erase only if the box is within the ROI
            if (x1 > x2 || y1 > y2)
                continue;

            Rpp32u boxHeight = y2 - y1 + 1;
            Rpp32u boxWidth = x2 - x1 + 1;
            Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);

            T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
            dstPtrTempR = dstPtrImage + pixelLocation;
            dstPtrTempG = dstPtrTempR + dstDescPtr->strides.cStride;
            dstPtrTempB = dstPtrTempG + dstDescPtr->strides.cStride;
            for (int i = 0; i < boxHeight; i++)
            {
                Rpp32u noiseRowOffset = ((y1 + i + batchCount) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * RANDOM_ERASE_NOISE_BUFFER_SIDE * 3;
                for (int j = 0; j < boxWidth; j++)
                {
                    Rpp32u noiseIdx = noiseRowOffset + ((x1 + j) % RANDOM_ERASE_NOISE_BUFFER_SIDE * 3);
                    dstPtrTempR[j] = noiseBuffer[noiseIdx];
                    dstPtrTempG[j] = noiseBuffer[noiseIdx + 1];
                    dstPtrTempB[j] = noiseBuffer[noiseIdx + 2];
                }
                dstPtrTempR += dstDescPtr->strides.hStride;
                dstPtrTempG += dstDescPtr->strides.hStride;
                dstPtrTempB += dstDescPtr->strides.hStride;
            }
        }
        // Erase without fused output-layout toggle 1 channel(NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            // To copy ROI region in Image
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                memcpy(dstPtrChannel, srcPtrChannel, bufferLength);
                srcPtrChannel += srcDescPtr->strides.hStride;
                dstPtrChannel += dstDescPtr->strides.hStride;
            }

            Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
            Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
            Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.x, x1, roi.xywhROI.roiWidth));
            Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.y, y1, roi.xywhROI.roiHeight));

            // Erase only if the box is within the ROI
            if (x1 > x2 || y1 > y2)
                continue;

            Rpp32u boxHeight = y2 - y1 + 1;
            Rpp32u boxWidth = x2 - x1 + 1;
            Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);

            T *dstPtrTemp = dstPtrImage + pixelLocation;
            for (int i = 0; i < boxHeight; i++)
            {
                Rpp32u noiseRowOffset = ((y1 + i + batchCount) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * RANDOM_ERASE_NOISE_BUFFER_SIDE;
                for (int j = 0; j < boxWidth; j++)
                {
                    Rpp32u noiseIdx = noiseRowOffset + ((x1 + j) % RANDOM_ERASE_NOISE_BUFFER_SIDE);
                    dstPtrTemp[j] = noiseBuffer[noiseIdx];
                }
                dstPtrTemp += dstDescPtr->strides.hStride;
            }
        }
        // Erase without fused output-layout toggle 3 channel(NHWC -> NHWC)
        else
        {
            // To copy ROI region in Image
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                memcpy(dstPtrChannel, srcPtrChannel, bufferLength);
                srcPtrChannel += srcDescPtr->strides.hStride;
                dstPtrChannel += dstDescPtr->strides.hStride;
            }

            Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth));
            Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight));
            Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.x, x1, roi.xywhROI.roiWidth));
            Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo.rb.y, y1, roi.xywhROI.roiHeight));

            // Erase only if the box is within the ROI
            if (x1 > x2 || y1 > y2)
                continue;

            Rpp32u boxHeight = y2 - y1 + 1;
            Rpp32u boxWidth = x2 - x1 + 1;
            Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);
            T *dstPtrTemp;
            dstPtrTemp = dstPtrImage + pixelLocation;

            for (int i = 0; i < boxHeight; i++)
            {
                Rpp32u noiseRowOffset = ((y1 + i + batchCount) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * RANDOM_ERASE_NOISE_BUFFER_SIDE * 3;
                T *dstPtrRow = dstPtrTemp;
                for (int j = 0; j < boxWidth; j++)
                {
                    Rpp32u noiseXIdx = ((x1 + j) % RANDOM_ERASE_NOISE_BUFFER_SIDE) * 3;
                    Rpp32u noiseIdx = noiseRowOffset + noiseXIdx;
                    dstPtrRow[0] = noiseBuffer[noiseIdx];     // R
                    dstPtrRow[1] = noiseBuffer[noiseIdx + 1]; // G
                    dstPtrRow[2] = noiseBuffer[noiseIdx + 2]; // B
                    dstPtrRow += dstDescPtr->c;
                }
                dstPtrTemp += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus random_erase_host_tensor<Rpp8u>(Rpp8u*,
                                                   RpptDescPtr,
                                                   Rpp8u*,
                                                   RpptDescPtr,
                                                   RpptRoiLtrb*,
                                                   Rpp8u*,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   RppLayoutParams,
                                                   rpp::Handle&);

template RppStatus random_erase_host_tensor<Rpp16f>(Rpp16f*,
                                                    RpptDescPtr,
                                                    Rpp16f*,
                                                    RpptDescPtr,
                                                    RpptRoiLtrb*,
                                                    Rpp16f*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    RppLayoutParams,
                                                    rpp::Handle&);

template RppStatus random_erase_host_tensor<Rpp32f>(Rpp32f*,
                                                    RpptDescPtr,
                                                    Rpp32f*,
                                                    RpptDescPtr,
                                                    RpptRoiLtrb*,
                                                    Rpp32f*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    RppLayoutParams,
                                                    rpp::Handle&);
       
template RppStatus random_erase_host_tensor<Rpp8s>(Rpp8s*,
                                                   RpptDescPtr,
                                                   Rpp8s*,
                                                   RpptDescPtr,
                                                   RpptRoiLtrb*,
                                                   Rpp8s*,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   RppLayoutParams,
                                                   rpp::Handle&);
