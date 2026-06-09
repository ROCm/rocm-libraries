/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
RppStatus coarse_dropout_host_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     Rpp32u *numBoxesTensor,
                                     Rpp32u maxBoxesPerImage, 
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

        Rpp32u numBoxes = std::min(numBoxesTensor[batchCount], maxBoxesPerImage);
        RpptRoiLtrb *anchorBoxInfo = anchorBoxInfoTensor + batchCount * maxBoxesPerImage;

        // Compute dropout value once based on data type instead of using ternary operator repeatedly
        T dropoutValue = (std::is_same<T, Rpp8s>::value) ? static_cast<T>(-128) : static_cast<T>(0);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier * sizeof(T);

       // Coarse dropout with fused output-layout toggle (NHWC -> NCHW)
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            T *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            // Copy ROI region from source to destination while converting layout (NHWC -> NCHW)
            // This preserves the original image data before applying dropout to specific boxes
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                for(int j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    *dstPtrTempR++ = *srcPtrTemp++;
                    *dstPtrTempG++ = *srcPtrTemp++;
                    *dstPtrTempB++ = *srcPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
            for(int count = 0; count < numBoxes; count++)
            {
                // Clamp anchor box coordinates to ROI bounds in image space
                Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1));
                Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1));
                Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1));
                Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1));

                // Convert to ROI-local coordinates
                x1 -= roi.xywhROI.xy.x;
                y1 -= roi.xywhROI.xy.y;
                x2 -= roi.xywhROI.xy.x;
                y2 -= roi.xywhROI.xy.y;

                Rpp32u pixelLocation = (y1 * dstDescPtr->strides.hStride) + (x1 * dstDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxWidth = x2 - x1 + 1;

                T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrImage + pixelLocation;
                dstPtrTempG = dstPtrTempR + dstDescPtr->strides.cStride;
                dstPtrTempB = dstPtrTempG + dstDescPtr->strides.cStride;
                for (int i = 0; i < boxHeight; i++)
                {
                    std::fill_n(dstPtrTempR, boxWidth, dropoutValue);
                    std::fill_n(dstPtrTempG, boxWidth, dropoutValue);
                    std::fill_n(dstPtrTempB, boxWidth, dropoutValue);
                    dstPtrTempR += dstDescPtr->strides.hStride;
                    dstPtrTempG += dstDescPtr->strides.hStride;
                    dstPtrTempB += dstDescPtr->strides.hStride;
                }
            }
        }

        // Coarse dropout with fused output-layout toggle (NCHW -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            // Copy ROI region from source to destination while converting layout (NCHW -> NHWC)
            // This preserves the original image data before applying dropout to specific boxes
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                 T *srcRowR, *srcRowG, *srcRowB, *dstPtrTemp;
                srcRowR = srcPtrRowR;
                srcRowG = srcPtrRowG;
                srcRowB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

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

            for(int count = 0; count < numBoxes; count++)
            {
                // Clamp anchor box coordinates to ROI bounds in image space
                Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1));
                Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1));
                Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1));
                Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1));

                // Convert to ROI-local coordinates
                x1 -= roi.xywhROI.xy.x;
                y1 -= roi.xywhROI.xy.y;
                x2 -= roi.xywhROI.xy.x;
                y2 -= roi.xywhROI.xy.y;

                Rpp32u pixelLocation = (y1 * dstDescPtr->strides.hStride) + (x1 * dstDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxWidth = x2 - x1 + 1;
                T *dstPtrTemp;
                dstPtrTemp = dstPtrImage + pixelLocation;

                for(int i = 0; i < boxHeight; i++)
                {
                    std::fill_n(dstPtrTemp, boxWidth * 3, dropoutValue);
                    dstPtrTemp += dstDescPtr->strides.hStride;
                }
            }
        }

        // Coarse dropout without fused output-layout toggle (same src/dst layout with 3 channels)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == dstDescPtr->layout))
        {
            // Copy ROI region from source to destination (same layout)
            // This preserves the original image data before applying dropout to specific boxes
            if (srcDescPtr->layout == RpptLayout::NCHW)
            {
                // For NCHW layout, channels are in separate planes, requiring per-channel copy
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
            }
            else // NHWC layout
            {
                // For NHWC layout, all channels are interleaved, single memcpy per row
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    memcpy(dstPtrChannel, srcPtrChannel, bufferLength);
                    srcPtrChannel += srcDescPtr->strides.hStride;
                    dstPtrChannel += dstDescPtr->strides.hStride;
                }
            }

            for(int count = 0; count < numBoxes; count++)
            {
                // Clamp anchor box coordinates to ROI bounds in image space
                Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1));
                Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1));
                Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1));
                Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1));

                // Convert to ROI-local coordinates
                x1 -= roi.xywhROI.xy.x;
                y1 -= roi.xywhROI.xy.y;
                x2 -= roi.xywhROI.xy.x;
                y2 -= roi.xywhROI.xy.y;

                Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxWidth = x2 - x1 + 1;

                if (srcDescPtr->layout == RpptLayout::NCHW)
                {
                    T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    dstPtrTempR = dstPtrImage + pixelLocation;
                    dstPtrTempG = dstPtrTempR + dstDescPtr->strides.cStride;
                    dstPtrTempB = dstPtrTempG + dstDescPtr->strides.cStride;
                    for (int i = 0; i < boxHeight; i++)
                    {
                        std::fill_n(dstPtrTempR, boxWidth, dropoutValue);
                        std::fill_n(dstPtrTempG, boxWidth, dropoutValue);
                        std::fill_n(dstPtrTempB, boxWidth, dropoutValue);
                        dstPtrTempR += dstDescPtr->strides.hStride;
                        dstPtrTempG += dstDescPtr->strides.hStride;
                        dstPtrTempB += dstDescPtr->strides.hStride;
                    }
                }
                else // NHWC layout
                {
                    T *dstPtrTemp = dstPtrImage + pixelLocation;
                    for(int i = 0; i < boxHeight; i++)
                    {
                        std::fill_n(dstPtrTemp, boxWidth * 3, dropoutValue);
                        dstPtrTemp += dstDescPtr->strides.hStride;
                    }
                }
            }
        }
        // Coarse dropout without fused output-layout toggle 1 channel(NCHW -> NCHW)
        else if((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            // Copy ROI region from source to destination (same layout)
            // This preserves the original image data before applying dropout to specific boxes
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                memcpy(dstPtrChannel, srcPtrChannel, bufferLength);
                srcPtrChannel += srcDescPtr->strides.hStride;
                dstPtrChannel += dstDescPtr->strides.hStride;
            }

            for(int count = 0; count < numBoxes; count++)
            {
                // Clamp anchor box coordinates to ROI bounds in image space
                Rpp32u x1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1));
                Rpp32u y1 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1));
                Rpp32u x2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1));
                Rpp32u y2 = static_cast<Rpp32u>(RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1));

                // Convert to ROI-local coordinates
                x1 -= roi.xywhROI.xy.x;
                y1 -= roi.xywhROI.xy.y;
                x2 -= roi.xywhROI.xy.x;
                y2 -= roi.xywhROI.xy.y;

                Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxWidth = x2 - x1 + 1;

                T *dstPtrTemp;
                dstPtrTemp = dstPtrImage + pixelLocation;

                for(int i = 0; i < boxHeight; i++)
                {
                    std::fill_n(dstPtrTemp, boxWidth, dropoutValue);
                    dstPtrTemp += dstDescPtr->strides.hStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus coarse_dropout_host_tensor<Rpp8u>(Rpp8u*,
                                                     RpptDescPtr,
                                                     Rpp8u*,
                                                     RpptDescPtr,
                                                     RpptRoiLtrb*,
                                                     Rpp32u*,
                                                     Rpp32u,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     RppLayoutParams,
                                                     rpp::Handle&);

template RppStatus coarse_dropout_host_tensor<Rpp16f>(Rpp16f*,
                                                      RpptDescPtr,
                                                      Rpp16f*,
                                                      RpptDescPtr,
                                                      RpptRoiLtrb*,
                                                      Rpp32u*,
                                                      Rpp32u, 
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);

template RppStatus coarse_dropout_host_tensor<Rpp32f>(Rpp32f*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptDescPtr,
                                                      RpptRoiLtrb*,
                                                      Rpp32u*,
                                                      Rpp32u, 
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);

template RppStatus coarse_dropout_host_tensor<Rpp8s>(Rpp8s*,
                                                     RpptDescPtr,
                                                     Rpp8s*,
                                                     RpptDescPtr,
                                                     RpptRoiLtrb*,
                                                     Rpp32u*,
                                                     Rpp32u, 
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);
