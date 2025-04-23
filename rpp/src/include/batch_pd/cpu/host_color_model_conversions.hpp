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

#ifndef HOST_COLOR_MODER_CONVERSIONS_HPP
#define HOST_COLOR_MODER_CONVERSIONS_HPP

#include "rpp_cpu_batchpd.hpp"
#include "rpp_cpu_simd_math.hpp"

/**************** channel_extract ***************/

template <typename T>
RppStatus channel_extract_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                     Rpp32u *batch_extractChannelNumber,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u extractChannelNumber = batch_extractChannelNumber[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_srcSizeMax, batchCount, &dstLoc, 1);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            T *srcPtrChannel;
            srcPtrChannel = srcPtrImage + (extractChannelNumber * imageDimMax);

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width);

                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u extractChannelNumber = batch_extractChannelNumber[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, channel);
            compute_image_location_host(batch_srcSizeMax, batchCount, &dstLoc, 1);
            srcPtrImage = srcPtr + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            T *srcPtrChannel;
            srcPtrChannel = srcPtrImage + extractChannelNumber;

            Rpp32u elementsInRowSrc = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMaxSrc = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowDst = 1 * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMaxDst = 1 * batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrChannel + (i * elementsInRowMaxSrc);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMaxDst);

                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    *dstPtrTemp = *srcPtrTemp;
                    srcPtrTemp = srcPtrTemp + channel;
                    dstPtrTemp++;
                }
                srcPtrTemp += (elementsInRowMaxSrc - elementsInRowSrc);
                dstPtrTemp += (elementsInRowMaxDst - elementsInRowDst);
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus channel_extract_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u extractChannelNumber,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (extractChannelNumber != 0 && extractChannelNumber != 1 && extractChannelNumber != 2)
    {
        return RPP_ERROR;
    }

    T *srcPtrTemp, *dstPtrTemp;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrTemp = srcPtr + (extractChannelNumber * srcSize.height * srcSize.width);
        memcpy(dstPtrTemp, srcPtrTemp, srcSize.height * srcSize.width * sizeof(T));
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTemp = srcPtr + extractChannelNumber;
        for (int i = 0; i < srcSize.height * srcSize.width; i++)
        {
            *dstPtrTemp = *srcPtrTemp;
            srcPtrTemp = srcPtrTemp + channel;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}

/**************** channel_combine ***************/

template <typename T>
RppStatus channel_combine_host_batch(T* srcPtr1, T* srcPtr2, T* srcPtr3, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                     Rpp32u nbatchSize,
                                     RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T *srcPtr1Image, *srcPtr2Image, *srcPtr3Image, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, 1);
            compute_image_location_host(batch_srcSizeMax, batchCount, &dstLoc, channel);
            srcPtr1Image = srcPtr1 + srcLoc;
            srcPtr2Image = srcPtr2 + srcLoc;
            srcPtr3Image = srcPtr3 + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            T *dstPtrChannel;
            dstPtrChannel = dstPtrImage;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtr1Image + (i * batch_srcSizeMax[batchCount].width);
                dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
            }

            dstPtrChannel = dstPtrImage + (imageDimMax);


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtr2Image + (i * batch_srcSizeMax[batchCount].width);
                dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
            }

            dstPtrChannel = dstPtrImage + (2 * imageDimMax);


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtr3Image + (i * batch_srcSizeMax[batchCount].width);
                dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T *srcPtr1Image, *srcPtr2Image, *srcPtr3Image, *dstPtrImage;
            Rpp32u srcLoc = 0, dstLoc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &srcLoc, 1);
            compute_image_location_host(batch_srcSizeMax, batchCount, &dstLoc, channel);
            srcPtr1Image = srcPtr1 + srcLoc;
            srcPtr2Image = srcPtr2 + srcLoc;
            srcPtr3Image = srcPtr3 + srcLoc;
            dstPtrImage = dstPtr + dstLoc;

            Rpp32u elementsInRowSrc = 1 * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMaxSrc = 1 * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRowDst = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMaxDst = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtr1Temp, *srcPtr2Temp, *srcPtr3Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Image + (i * elementsInRowMaxSrc);
                srcPtr2Temp = srcPtr2Image + (i * elementsInRowMaxSrc);
                srcPtr3Temp = srcPtr3Image + (i * elementsInRowMaxSrc);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMaxDst);

                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    *dstPtrTemp = *srcPtr1Temp;
                    dstPtrTemp++;
                    srcPtr1Temp++;
                    *dstPtrTemp = *srcPtr2Temp;
                    dstPtrTemp++;
                    srcPtr2Temp++;
                    *dstPtrTemp = *srcPtr3Temp;
                    dstPtrTemp++;
                    srcPtr3Temp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus channel_combine_host(T* srcPtr1, T* srcPtr2, T* srcPtr3, RppiSize srcSize, T* dstPtr,
                               RppiChnFormat chnFormat, Rpp32u channel)
{

    T *srcPtr1Temp, *srcPtr2Temp, *srcPtr3Temp, *dstPtrTemp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    srcPtr3Temp = srcPtr3;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u increment = srcSize.height * srcSize.width;
        memcpy(dstPtrTemp, srcPtr1Temp, srcSize.height * srcSize.width * sizeof(T));
        dstPtrTemp += increment;
        memcpy(dstPtrTemp, srcPtr2Temp, srcSize.height * srcSize.width * sizeof(T));
        dstPtrTemp += increment;
        memcpy(dstPtrTemp, srcPtr3Temp, srcSize.height * srcSize.width * sizeof(T));
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height * srcSize.width; i++)
        {
            *dstPtrTemp = *srcPtr1Temp;
            dstPtrTemp++;
            srcPtr1Temp++;
            *dstPtrTemp = *srcPtr2Temp;
            dstPtrTemp++;
            srcPtr2Temp++;
            *dstPtrTemp = *srcPtr3Temp;
            dstPtrTemp++;
            srcPtr3Temp++;
        }
    }

    return RPP_SUCCESS;
}

/**************** look_up_table ***************/

// template <typename T>
// RppStatus look_up_table_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
//                                    T* batch_lutPtr,
//                                    RppiROI *roiPoints, Rpp32u nbatchSize,
//                                    RppiChnFormat chnFormat, Rpp32u channel)
// {
//     Rpp32u lutSize = channel * 256;
//     if(chnFormat == RPPI_CHN_PLANAR)
//     {
//         omp_set_dynamic(0);
// #pragma omp parallel for num_threads(numThreads)
//         for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
//         {
//             Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

//             Rpp32f x1 = roiPoints[batchCount].x;
//             Rpp32f y1 = roiPoints[batchCount].y;
//             Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
//             Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
//             if (x2 == -1)
//             {
//                 x2 = batch_srcSize[batchCount].width - 1;
//                 roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
//             }
//             if (y2 == -1)
//             {
//                 y2 = batch_srcSize[batchCount].height - 1;
//                 roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
//             }

//             T* lutPtr = (T*) calloc(lutSize, sizeof(T));
//             memcpy(lutPtr, (batch_lutPtr + (batchCount * lutSize)), lutSize * sizeof(T));

//             T *srcPtrImage, *dstPtrImage;
//             Rpp32u loc = 0;
//             compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
//             srcPtrImage = srcPtr + loc;
//             dstPtrImage = dstPtr + loc;

//             for(int c = 0; c < channel; c++)
//             {
//                 T *srcPtrChannel, *dstPtrChannel;
//                 srcPtrChannel = srcPtrImage + (c * imageDimMax);
//                 dstPtrChannel = dstPtrImage + (c * imageDimMax);

//                 T *lutPtrTemp;
//                 lutPtrTemp = lutPtr + (c * 256);


//                 for(int i = 0; i < batch_srcSize[batchCount].height; i++)
//                 {
//                     T *srcPtrTemp, *dstPtrTemp;
//                     srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
//                     dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

//                     if (!((y1 <= i) && (i <= y2)))
//                     {
//                         memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
//                         dstPtrTemp += batch_srcSizeMax[batchCount].width;
//                         srcPtrTemp += batch_srcSizeMax[batchCount].width;
//                     }
//                     else
//                     {
//                         for(int j = 0; j < batch_srcSize[batchCount].width; j++)
//                         {
//                             if((x1 <= j) && (j <= x2 ))
//                             {
//                                 *dstPtrTemp = *(lutPtrTemp + (Rpp32u)(*srcPtrTemp));

//                                 srcPtrTemp++;
//                                 dstPtrTemp++;
//                             }
//                             else
//                             {
//                                 *dstPtrTemp = *srcPtrTemp;
//                                 srcPtrTemp++;
//                                 dstPtrTemp++;
//                             }
//                         }
//                         srcPtrTemp += batch_srcSizeMax[batchCount].width - batch_srcSize[batchCount].width;
//                         dstPtrTemp += batch_srcSizeMax[batchCount].width - batch_srcSize[batchCount].width;
//                     }
//                 }
//             }

//             free(lutPtr);
//         }
//     }
//     else if (chnFormat == RPPI_CHN_PACKED)
//     {
//         omp_set_dynamic(0);
// #pragma omp parallel for num_threads(numThreads)
//         for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
//         {
//             Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

//             Rpp32f x1 = roiPoints[batchCount].x;
//             Rpp32f y1 = roiPoints[batchCount].y;
//             Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
//             Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
//             if (x2 == -1)
//             {
//                 x2 = batch_srcSize[batchCount].width - 1;
//                 roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
//             }
//             if (y2 == -1)
//             {
//                 y2 = batch_srcSize[batchCount].height - 1;
//                 roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
//             }

//             T* lutPtr = (T*) calloc(lutSize, sizeof(T));
//             memcpy(lutPtr, (batch_lutPtr + (batchCount * lutSize)), lutSize * sizeof(T));

//             T *srcPtrImage, *dstPtrImage;
//             Rpp32u loc = 0;
//             compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
//             srcPtrImage = srcPtr + loc;
//             dstPtrImage = dstPtr + loc;

//             Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
//             Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


//             for(int i = 0; i < batch_srcSize[batchCount].height; i++)
//             {
//                 T *srcPtrTemp, *dstPtrTemp;
//                 srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
//                 dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

//                 if (!((y1 <= i) && (i <= y2)))
//                 {
//                     memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
//                     dstPtrTemp += elementsInRowMax;
//                     srcPtrTemp += elementsInRowMax;
//                 }
//                 else
//                 {
//                     for(int j = 0; j < batch_srcSize[batchCount].width; j++)
//                     {
//                         if (!((x1 <= j) && (j <= x2 )))
//                         {
//                             memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));
//                             dstPtrTemp += channel;
//                             srcPtrTemp += channel;
//                         }
//                         else
//                         {
//                             for(int c = 0; c < channel; c++)
//                             {
//                                 *dstPtrTemp = *(lutPtr + c + (channel * (Rpp32u)(*srcPtrTemp)));

//                                 srcPtrTemp++;
//                                 dstPtrTemp++;
//                             }
//                         }
//                     }
//                     srcPtrTemp += (elementsInRowMax - elementsInRow);
//                     dstPtrTemp += (elementsInRowMax - elementsInRow);
//                 }
//             }

//             free(lutPtr);
//         }
//     }

//     return RPP_SUCCESS;
// }

template <typename T>
RppStatus look_up_table_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                        T *batch_lutPtr,
                        RppiROI *roiPoints, Rpp32u nbatchSize,
                        RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u lutSize = 256;
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T* lutPtr = (T*) calloc(lutSize, sizeof(T));
            memcpy(lutPtr, (batch_lutPtr + (batchCount * lutSize)), lutSize * sizeof(T));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);

                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                    {
                        *dstPtrTemp = *(lutPtr + (Rpp32s)(*srcPtrTemp));

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                }
            }
            free(lutPtr);
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            T* lutPtr = (T*) calloc(lutSize, sizeof(T));
            memcpy(lutPtr, (batch_lutPtr + (batchCount * lutSize)), lutSize * sizeof(T));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                for(int j = 0; j < batch_srcSize[batchCount].width; j++)
                {
                    for(int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *(lutPtr + (Rpp32s) *srcPtrTemp);

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                }

                srcPtrTemp += (elementsInRowMax - elementsInRow);
                dstPtrTemp += (elementsInRowMax - elementsInRow);
            }
            free(lutPtr);
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus look_up_table_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    U *lutPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    U *lutPtrTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            lutPtrTemp = lutPtr + (c * 256);
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                *dstPtrTemp = *(lutPtrTemp + (Rpp32u)(*srcPtrTemp));
                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        lutPtrTemp = lutPtr;
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            for (int c = 0; c < channel; c++)
            {
                *dstPtrTemp = *(lutPtrTemp + c + (channel * (Rpp32u)(*srcPtrTemp)));
                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** color_temperature ***************/

template <typename T>
RppStatus color_temperature_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                Rpp32s *batch_adjustmentValue,
                                RppiROI *roiPoints, Rpp32u nbatchSize,
                                RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if (channel == 1)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32s adjustmentValue = batch_adjustmentValue[batchCount];
            Rpp16s adjustmentValueShort = (Rpp16s) adjustmentValue;
            Rpp8s adjustmentValueChar = (Rpp8s) adjustmentValue;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32s pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        int bufferLength = roiPoints[batchCount].roiWidth;
                        int alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i pAdd = _mm_set1_epi8(adjustmentValueChar);
                        __m128i px0;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                            px0 = _mm_adds_epu8(px0, pAdd);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtrTemp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ + adjustmentValueShort);
                        }

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
        }
    }
    else if (channel == 3)
    {
        if(chnFormat == RPPI_CHN_PLANAR)
        {
            omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
            for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
            {
                Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

                Rpp32u x1 = roiPoints[batchCount].x;
                Rpp32u y1 = roiPoints[batchCount].y;
                Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
                Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
                if (x2 == -1)
                {
                    x2 = batch_srcSize[batchCount].width - 1;
                    roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
                }
                if (y2 == -1)
                {
                    y2 = batch_srcSize[batchCount].height - 1;
                    roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
                }

                Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

                Rpp32s adjustmentValue = batch_adjustmentValue[batchCount];
                Rpp16s adjustmentValueShort = (Rpp16s) adjustmentValue;
                Rpp8s adjustmentValueChar = (Rpp8s) adjustmentValue;

                T *srcPtrImage, *dstPtrImage;
                Rpp32u loc = 0;
                compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
                srcPtrImage = srcPtr + loc;
                dstPtrImage = dstPtr + loc;

                T *srcPtrChannel, *dstPtrChannel;

                srcPtrChannel = srcPtrImage;
                dstPtrChannel = dstPtrImage;


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32s pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        int bufferLength = roiPoints[batchCount].roiWidth;
                        int alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i pAdd = _mm_set1_epi8(adjustmentValueChar);
                        __m128i px0;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                            px0 = _mm_subs_epu8(px0, pAdd);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtrTemp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ - adjustmentValueShort);
                        }

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }

                srcPtrChannel = srcPtrImage + (imageDimMax);
                dstPtrChannel = dstPtrImage + (imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));
                }

                srcPtrChannel = srcPtrImage + (2 * imageDimMax);
                dstPtrChannel = dstPtrImage + (2 * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32s pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        int bufferLength = roiPoints[batchCount].roiWidth;
                        int alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i pAdd = _mm_set1_epi8(adjustmentValueChar);
                        __m128i px0;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                            px0 = _mm_adds_epu8(px0, pAdd);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtrTemp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ + adjustmentValueShort);
                        }

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
            for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
            {
                Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

                Rpp32f x1 = roiPoints[batchCount].x;
                Rpp32f y1 = roiPoints[batchCount].y;
                Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth - 1;
                Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight - 1;
                if (x2 == -1)
                {
                    x2 = batch_srcSize[batchCount].width - 1;
                    roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
                }
                if (y2 == -1)
                {
                    y2 = batch_srcSize[batchCount].height - 1;
                    roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
                }

                Rpp32u elementsBeforeROI = channel * x1;
                Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

                Rpp32s adjustmentValue = batch_adjustmentValue[batchCount];
                Rpp16s adjustmentValueShort = (Rpp16s) adjustmentValue;
                Rpp8s adjustmentValueChar = (Rpp8s) adjustmentValue;

                T *srcPtrImage, *dstPtrImage;
                Rpp32u loc = 0;
                compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
                srcPtrImage = srcPtr + loc;
                dstPtrImage = dstPtr + loc;

                Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
                Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    Rpp32s pixel;

                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                    dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                        dstPtrTemp += elementsInRowMax;
                        srcPtrTemp += elementsInRowMax;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                        srcPtrTemp += elementsBeforeROI;
                        dstPtrTemp += elementsBeforeROI;

                        Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;
                        int alignedLength = bufferLength & ~14;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i pAdd1 = _mm_set1_epi16(adjustmentValueShort);
                        __m128i pAdd2 = _mm_set1_epi16(adjustmentValueShort);
                        __m128i pMask1 = _mm_setr_epi16(-1, 0, 1, -1, 0, 1, -1, 0);
                        __m128i pMask2 = _mm_setr_epi16(1, -1, 0, 1, -1, 0, 1, 0);
                        pAdd1 = _mm_mullo_epi16(pAdd1, pMask1);
                        pAdd2 = _mm_mullo_epi16(pAdd2, pMask2);
                        __m128i px0, px1;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7

                            px0 = _mm_adds_epi16(px0, pAdd1);
                            px1 = _mm_adds_epi16(px1, pAdd2);

                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtrTemp +=15;
                            dstPtrTemp +=15;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ + adjustmentValueShort);
                            *dstPtrTemp++ = *srcPtrTemp++;
                            *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ - adjustmentValueShort);
                        }

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus color_temperature_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32s adjustmentValue,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (channel != 1 && channel !=  3)
    {
        return RPP_ERROR;
    }
    if (adjustmentValue < -100 || adjustmentValue > 100)
    {
        return RPP_ERROR;
    }

    Rpp16s adjustmentValueShort = (Rpp16s) adjustmentValue;
    Rpp8s adjustmentValueChar = (Rpp8s) adjustmentValue;

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    if (channel == 1)
    {
        int bufferLength = srcSize.height * srcSize.width;
        int alignedLength = bufferLength & ~15;

        __m128i const zero = _mm_setzero_si128();
        __m128i pAdd = _mm_set1_epi8(adjustmentValueChar);
        __m128i px0;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
            px0 = _mm_adds_epu8(px0, pAdd);
            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

            srcPtrTemp +=16;
            dstPtrTemp +=16;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ + adjustmentValueShort);
        }
    }
    else if (channel == 3)
    {
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            int bufferLength = srcSize.height * srcSize.width;
            int alignedLength = bufferLength & ~15;

            __m128i const zero = _mm_setzero_si128();
            __m128i pAdd = _mm_set1_epi8(adjustmentValueChar);
            __m128i px0;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                px0 = _mm_adds_epu8(px0, pAdd);
                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                srcPtrTemp +=16;
                dstPtrTemp +=16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ + adjustmentValueShort);
            }

            memcpy(dstPtrTemp, srcPtrTemp, bufferLength * sizeof(T));
            dstPtrTemp += bufferLength;
            srcPtrTemp += bufferLength;

            vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                px0 = _mm_subs_epu8(px0, pAdd);
                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                srcPtrTemp +=16;
                dstPtrTemp +=16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ - adjustmentValueShort);
            }

        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            int bufferLength = srcSize.height * srcSize.width * channel;
            int alignedLength = bufferLength & ~14;

            __m128i const zero = _mm_setzero_si128();
            __m128i pAdd = _mm_set1_epi8(adjustmentValueChar);
            __m128i pMask = _mm_setr_epi8(1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 0);
            pAdd = _mm_mullo_epi8(pAdd, pMask);
            __m128i px0;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                px0 = _mm_adds_epu8(px0, pAdd);
                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                srcPtrTemp +=15;
                dstPtrTemp +=15;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
            {
                *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ + adjustmentValueShort);
                dstPtrTemp++;
                srcPtrTemp++;
                *dstPtrTemp++ = (T) RPPPIXELCHECK((Rpp16s) *srcPtrTemp++ - adjustmentValueShort);
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** vignette ***************/

template <typename T>
RppStatus vignette_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              Rpp32f *batch_stdDev,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f multiplier = -1 / (2 * stdDev * stdDev);

            Rpp32u halfHeight = batch_srcSize[batchCount].height / 2;
            Rpp32u halfWidth = batch_srcSize[batchCount].width / 2;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        int iCenterRef = i - halfHeight;
                        Rpp32f iCenterRefSqr = iCenterRef * iCenterRef;

                        Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
                        Rpp32u alignedLength = bufferLength & ~15;

                        __m128i const zero = _mm_setzero_si128();
                        __m128i pMask0 = _mm_setr_epi32(0, 1, 2, 3);
                        __m128i pMask1 = _mm_setr_epi32(4, 5, 6, 7);
                        __m128i pMask2 = _mm_setr_epi32(8, 9, 10, 11);
                        __m128i pMask3 = _mm_setr_epi32(12, 13, 14, 15);
                        __m128 pICenterRefSqr = _mm_set1_ps((Rpp32f) (iCenterRef * iCenterRef));
                        __m128 pHalfWidth = _mm_set1_ps((Rpp32f) halfWidth);
                        __m128 pMul = _mm_set1_ps(multiplier);
                        __m128 p0, p1, p2, p3, q0, q1, q2, q3;
                        __m128i px0, px1, px2, px3, qx0, qx1, qx2, qx3;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                        {
                            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                            px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                            p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                            p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                            p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                            p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                            qx3 = _mm_set1_epi32(vectorLoopCount);

                            qx0 = _mm_add_epi32(qx3, pMask0);
                            qx1 = _mm_add_epi32(qx3, pMask1);
                            qx2 = _mm_add_epi32(qx3, pMask2);
                            qx3 = _mm_add_epi32(qx3, pMask3);

                            q0 = _mm_cvtepi32_ps(qx0);
                            q1 = _mm_cvtepi32_ps(qx1);
                            q2 = _mm_cvtepi32_ps(qx2);
                            q3 = _mm_cvtepi32_ps(qx3);

                            q0 = _mm_sub_ps(q0, pHalfWidth);
                            q1 = _mm_sub_ps(q1, pHalfWidth);
                            q2 = _mm_sub_ps(q2, pHalfWidth);
                            q3 = _mm_sub_ps(q3, pHalfWidth);

                            q0 = _mm_mul_ps(q0, q0);
                            q1 = _mm_mul_ps(q1, q1);
                            q2 = _mm_mul_ps(q2, q2);
                            q3 = _mm_mul_ps(q3, q3);

                            q0 = _mm_add_ps(pICenterRefSqr, q0);
                            q1 = _mm_add_ps(pICenterRefSqr, q1);
                            q2 = _mm_add_ps(pICenterRefSqr, q2);
                            q3 = _mm_add_ps(pICenterRefSqr, q3);

                            q0 = _mm_mul_ps(q0, pMul);
                            q1 = _mm_mul_ps(q1, pMul);
                            q2 = _mm_mul_ps(q2, pMul);
                            q3 = _mm_mul_ps(q3, pMul);

                            q0 = fast_exp_sse(q0);
                            q1 = fast_exp_sse(q1);
                            q2 = fast_exp_sse(q2);
                            q3 = fast_exp_sse(q3);

                            p0 = _mm_mul_ps(p0, q0);
                            p1 = _mm_mul_ps(p1, q1);
                            p2 = _mm_mul_ps(p2, q2);
                            p3 = _mm_mul_ps(p3, q3);

                            px0 = _mm_cvtps_epi32(p0);
                            px1 = _mm_cvtps_epi32(p1);
                            px2 = _mm_cvtps_epi32(p2);
                            px3 = _mm_cvtps_epi32(p3);

                            px0 = _mm_packus_epi32(px0, px1);
                            px1 = _mm_packus_epi32(px2, px3);
                            px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                            _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                            srcPtrTemp +=16;
                            dstPtrTemp +=16;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            int jCenterRef = vectorLoopCount - halfWidth;
                            Rpp32f jCenterRefSqr = jCenterRef * jCenterRef;

                            *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * (multiplier * (iCenterRefSqr + jCenterRefSqr)));
                        }

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32f stdDev = batch_stdDev[batchCount];
            Rpp32f multiplier = -1 / (2 * stdDev * stdDev);

            Rpp32u halfHeight = batch_srcSize[batchCount].height / 2;
            Rpp32u halfWidth = batch_srcSize[batchCount].width / 2;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                    srcPtrTemp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    int iCenterRef = i - halfHeight;
                    Rpp32f iCenterRefSqr = iCenterRef * iCenterRef;

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;
                    Rpp32u alignedLength = bufferLength & ~14;

                    __m128i const zero = _mm_setzero_si128();
                    __m128i pMask0 = _mm_setr_epi32(0, 0, 0, 1);
                    __m128i pMask1 = _mm_setr_epi32(1, 1, 2, 2);
                    __m128i pMask2 = _mm_setr_epi32(2, 3, 3, 3);
                    __m128i pMask3 = _mm_setr_epi32(4, 4, 4, 0);
                    __m128 pICenterRefSqr = _mm_set1_ps((Rpp32f) (iCenterRef * iCenterRef));
                    __m128 pHalfWidth = _mm_set1_ps((Rpp32f) halfWidth);
                    __m128 pMul = _mm_set1_ps(multiplier);
                    __m128 p0, p1, p2, p3, q0, q1, q2, q3;
                    __m128i px0, px1, px2, px3, qx0, qx1, qx2, qx3;

                    int vectorLoopCount = 0;
                    int jPos = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=15, jPos += 5)
                    {
                        px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                        px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                        px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                        p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                        p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                        p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                        p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                        qx3 = _mm_set1_epi32(jPos);

                        qx0 = _mm_add_epi32(qx3, pMask0);
                        qx1 = _mm_add_epi32(qx3, pMask1);
                        qx2 = _mm_add_epi32(qx3, pMask2);
                        qx3 = _mm_add_epi32(qx3, pMask3);

                        q0 = _mm_cvtepi32_ps(qx0);
                        q1 = _mm_cvtepi32_ps(qx1);
                        q2 = _mm_cvtepi32_ps(qx2);
                        q3 = _mm_cvtepi32_ps(qx3);

                        q0 = _mm_sub_ps(q0, pHalfWidth);
                        q1 = _mm_sub_ps(q1, pHalfWidth);
                        q2 = _mm_sub_ps(q2, pHalfWidth);
                        q3 = _mm_sub_ps(q3, pHalfWidth);

                        q0 = _mm_mul_ps(q0, q0);
                        q1 = _mm_mul_ps(q1, q1);
                        q2 = _mm_mul_ps(q2, q2);
                        q3 = _mm_mul_ps(q3, q3);

                        q0 = _mm_add_ps(pICenterRefSqr, q0);
                        q1 = _mm_add_ps(pICenterRefSqr, q1);
                        q2 = _mm_add_ps(pICenterRefSqr, q2);
                        q3 = _mm_add_ps(pICenterRefSqr, q3);

                        q0 = _mm_mul_ps(q0, pMul);
                        q1 = _mm_mul_ps(q1, pMul);
                        q2 = _mm_mul_ps(q2, pMul);
                        q3 = _mm_mul_ps(q3, pMul);

                        q0 = fast_exp_sse(q0);
                        q1 = fast_exp_sse(q1);
                        q2 = fast_exp_sse(q2);
                        q3 = fast_exp_sse(q3);

                        p0 = _mm_mul_ps(p0, q0);
                        p1 = _mm_mul_ps(p1, q1);
                        p2 = _mm_mul_ps(p2, q2);
                        p3 = _mm_mul_ps(p3, q3);

                        px0 = _mm_cvtps_epi32(p0);
                        px1 = _mm_cvtps_epi32(p1);
                        px2 = _mm_cvtps_epi32(p2);
                        px3 = _mm_cvtps_epi32(p3);

                        px0 = _mm_packus_epi32(px0, px1);
                        px1 = _mm_packus_epi32(px2, px3);
                        px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                        _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                        srcPtrTemp +=15;
                        dstPtrTemp +=15;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel, jPos++)
                    {
                        int jCenterRef = jPos - halfWidth;
                        Rpp32f jCenterRefSqr = jCenterRef * jCenterRef;

                        for (int c = 0; c < channel; c++)
                        {
                            *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * (multiplier * (iCenterRefSqr + jCenterRefSqr)));
                        }
                    }

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus vignette_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                          Rpp32f stdDev,
                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;

    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f multiplier = -1 / (2 * stdDev * stdDev);

    Rpp32u halfHeight = srcSize.height / 2;
    Rpp32u halfWidth = srcSize.width / 2;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);

            for (int i = 0; i < srcSize.height; i++)
            {
                int iCenterRef = i - halfHeight;
                Rpp32f iCenterRefSqr = iCenterRef * iCenterRef;

                int bufferLength = srcSize.width;
                int alignedLength = bufferLength & ~15;

                __m128i const zero = _mm_setzero_si128();
                __m128i pMask0 = _mm_setr_epi32(0, 1, 2, 3);
                __m128i pMask1 = _mm_setr_epi32(4, 5, 6, 7);
                __m128i pMask2 = _mm_setr_epi32(8, 9, 10, 11);
                __m128i pMask3 = _mm_setr_epi32(12, 13, 14, 15);
                __m128 pICenterRefSqr = _mm_set1_ps((Rpp32f) (iCenterRef * iCenterRef));
                __m128 pHalfWidth = _mm_set1_ps((Rpp32f) halfWidth);
                __m128 pMul = _mm_set1_ps(multiplier);
                __m128 p0, p1, p2, p3, q0, q1, q2, q3;
                __m128i px0, px1, px2, px3, qx0, qx1, qx2, qx3;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                    qx3 = _mm_set1_epi32(vectorLoopCount);

                    qx0 = _mm_add_epi32(qx3, pMask0);
                    qx1 = _mm_add_epi32(qx3, pMask1);
                    qx2 = _mm_add_epi32(qx3, pMask2);
                    qx3 = _mm_add_epi32(qx3, pMask3);

                    q0 = _mm_cvtepi32_ps(qx0);
                    q1 = _mm_cvtepi32_ps(qx1);
                    q2 = _mm_cvtepi32_ps(qx2);
                    q3 = _mm_cvtepi32_ps(qx3);

                    q0 = _mm_sub_ps(q0, pHalfWidth);
                    q1 = _mm_sub_ps(q1, pHalfWidth);
                    q2 = _mm_sub_ps(q2, pHalfWidth);
                    q3 = _mm_sub_ps(q3, pHalfWidth);

                    q0 = _mm_mul_ps(q0, q0);
                    q1 = _mm_mul_ps(q1, q1);
                    q2 = _mm_mul_ps(q2, q2);
                    q3 = _mm_mul_ps(q3, q3);

                    q0 = _mm_add_ps(pICenterRefSqr, q0);
                    q1 = _mm_add_ps(pICenterRefSqr, q1);
                    q2 = _mm_add_ps(pICenterRefSqr, q2);
                    q3 = _mm_add_ps(pICenterRefSqr, q3);

                    q0 = _mm_mul_ps(q0, pMul);
                    q1 = _mm_mul_ps(q1, pMul);
                    q2 = _mm_mul_ps(q2, pMul);
                    q3 = _mm_mul_ps(q3, pMul);

                    q0 = fast_exp_sse(q0);
                    q1 = fast_exp_sse(q1);
                    q2 = fast_exp_sse(q2);
                    q3 = fast_exp_sse(q3);

                    p0 = _mm_mul_ps(p0, q0);
                    p1 = _mm_mul_ps(p1, q1);
                    p2 = _mm_mul_ps(p2, q2);
                    p3 = _mm_mul_ps(p3, q3);

                    px0 = _mm_cvtps_epi32(p0);
                    px1 = _mm_cvtps_epi32(p1);
                    px2 = _mm_cvtps_epi32(p2);
                    px3 = _mm_cvtps_epi32(p3);

                    px0 = _mm_packus_epi32(px0, px1);
                    px1 = _mm_packus_epi32(px2, px3);
                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                    srcPtrTemp +=16;
                    dstPtrTemp +=16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    int jCenterRef = vectorLoopCount - halfWidth;
                    Rpp32f jCenterRefSqr = jCenterRef * jCenterRef;

                    *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * (multiplier * (iCenterRefSqr + jCenterRefSqr)));
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;

        for (int i = 0; i < srcSize.height; i++)
        {
            int iCenterRef = i - halfHeight;
            Rpp32f iCenterRefSqr = iCenterRef * iCenterRef;

            int bufferLength = elementsInRow;
            int alignedLength = bufferLength & ~14;

            __m128i const zero = _mm_setzero_si128();
            __m128i pMask0 = _mm_setr_epi32(0, 0, 0, 1);
            __m128i pMask1 = _mm_setr_epi32(1, 1, 2, 2);
            __m128i pMask2 = _mm_setr_epi32(2, 3, 3, 3);
            __m128i pMask3 = _mm_setr_epi32(4, 4, 4, 0);
            __m128 pICenterRefSqr = _mm_set1_ps((Rpp32f) (iCenterRef * iCenterRef));
            __m128 pHalfWidth = _mm_set1_ps((Rpp32f) halfWidth);
            __m128 pMul = _mm_set1_ps(multiplier);
            __m128 p0, p1, p2, p3, q0, q1, q2, q3;
            __m128i px0, px1, px2, px3, qx0, qx1, qx2, qx3;

            int vectorLoopCount = 0;
            int jPos = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=15, jPos += 5)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

                px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

                qx3 = _mm_set1_epi32(jPos);

                qx0 = _mm_add_epi32(qx3, pMask0);
                qx1 = _mm_add_epi32(qx3, pMask1);
                qx2 = _mm_add_epi32(qx3, pMask2);
                qx3 = _mm_add_epi32(qx3, pMask3);

                q0 = _mm_cvtepi32_ps(qx0);
                q1 = _mm_cvtepi32_ps(qx1);
                q2 = _mm_cvtepi32_ps(qx2);
                q3 = _mm_cvtepi32_ps(qx3);

                q0 = _mm_sub_ps(q0, pHalfWidth);
                q1 = _mm_sub_ps(q1, pHalfWidth);
                q2 = _mm_sub_ps(q2, pHalfWidth);
                q3 = _mm_sub_ps(q3, pHalfWidth);

                q0 = _mm_mul_ps(q0, q0);
                q1 = _mm_mul_ps(q1, q1);
                q2 = _mm_mul_ps(q2, q2);
                q3 = _mm_mul_ps(q3, q3);

                q0 = _mm_add_ps(pICenterRefSqr, q0);
                q1 = _mm_add_ps(pICenterRefSqr, q1);
                q2 = _mm_add_ps(pICenterRefSqr, q2);
                q3 = _mm_add_ps(pICenterRefSqr, q3);

                q0 = _mm_mul_ps(q0, pMul);
                q1 = _mm_mul_ps(q1, pMul);
                q2 = _mm_mul_ps(q2, pMul);
                q3 = _mm_mul_ps(q3, pMul);

                q0 = fast_exp_sse(q0);
                q1 = fast_exp_sse(q1);
                q2 = fast_exp_sse(q2);
                q3 = fast_exp_sse(q3);

                p0 = _mm_mul_ps(p0, q0);
                p1 = _mm_mul_ps(p1, q1);
                p2 = _mm_mul_ps(p2, q2);
                p3 = _mm_mul_ps(p3, q3);

                px0 = _mm_cvtps_epi32(p0);
                px1 = _mm_cvtps_epi32(p1);
                px2 = _mm_cvtps_epi32(p2);
                px3 = _mm_cvtps_epi32(p3);

                px0 = _mm_packus_epi32(px0, px1);
                px1 = _mm_packus_epi32(px2, px3);
                px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                srcPtrTemp +=15;
                dstPtrTemp +=15;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount+=channel, jPos++)
            {
                int jCenterRef = jPos - halfWidth;
                Rpp32f jCenterRefSqr = jCenterRef * jCenterRef;

                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp++ = (T) RPPPIXELCHECK(((Rpp32f) (*srcPtrTemp++)) * (multiplier * (iCenterRefSqr + jCenterRefSqr)));
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** hue ***************/

template <typename T>
RppStatus hueRGB_processBuffer_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                    Rpp32f hueShift, Rpp32f hueShiftAngle, Rpp64u bufferLength,
                                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
        T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u alignedLength = bufferLength & ~3;

        __m128i const zero = _mm_setzero_si128();
        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128i px0, px1, px2;
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        Rpp8u arrayR[4];
        Rpp8u arrayG[4];
        Rpp8u arrayB[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTempR);
            px1 =  _mm_loadu_si128((__m128i *)srcPtrTempG);
            px2 =  _mm_loadu_si128((__m128i *)srcPtrTempB);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);

            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);
            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));

            xG = _mm_sub_ps(xG, xH);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            px0 = _mm_cvtps_epi32(x1);
            px1 = _mm_cvtps_epi32(x2);
            px2 = _mm_cvtps_epi32(x3);

            px0 = _mm_packs_epi32(px0, px0);
            px0 = _mm_packus_epi16(px0, px0);
            *((int*)arrayR) = _mm_cvtsi128_si32(px0);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayG) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayB) = _mm_cvtsi128_si32(px2);

            memcpy(dstPtrTempR, arrayR, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempG, arrayG, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempB, arrayB, 4 * sizeof(Rpp8u));

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        T *srcPtrTempPx0, *srcPtrTempPx1, *srcPtrTempPx2, *srcPtrTempPx3;
        T *dstPtrTempPx0, *dstPtrTempPx1, *dstPtrTempPx2, *dstPtrTempPx3;

        srcPtrTempPx0 = srcPtr;
        srcPtrTempPx1 = srcPtr + 3;
        srcPtrTempPx2 = srcPtr + 6;
        srcPtrTempPx3 = srcPtr + 9;
        dstPtrTempPx0 = dstPtr;
        dstPtrTempPx1 = dstPtr + 3;
        dstPtrTempPx2 = dstPtr + 6;
        dstPtrTempPx3 = dstPtr + 9;

        Rpp64u alignedLength = (bufferLength / 12) * 12;

        __m128i const zero = _mm_setzero_si128();
        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pHueShift = _mm_set1_ps(hueShift);
        __m128i px0, px1, px2, px3;
        __m128 xR, xG, xB, xA;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        Rpp8u arrayPx0[4];
        Rpp8u arrayPx1[4];
        Rpp8u arrayPx2[4];
        Rpp8u arrayPx3[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
        {
            px0 = _mm_loadu_si128((__m128i *)srcPtrTempPx0);
            px1 = _mm_loadu_si128((__m128i *)srcPtrTempPx1);
            px2 = _mm_loadu_si128((__m128i *)srcPtrTempPx2);
            px3 = _mm_loadu_si128((__m128i *)srcPtrTempPx3);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7
            px3 = _mm_unpacklo_epi8(px3, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3
            xA = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px3, zero));    // pixels 0-3

            _MM_TRANSPOSE4_PS (xR, xG, xB, xA);

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Hue Values and re-normalize H to fraction
            xG = _mm_add_ps(xG, pHueShift);

            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);
            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));

            xG = _mm_sub_ps(xG, xH);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            x0 = _mm_shuffle_ps(x0,x0, _MM_SHUFFLE(0,3,2,1));
            x1 = _mm_shuffle_ps(x1,x1, _MM_SHUFFLE(0,3,2,1));
            x2 = _mm_shuffle_ps(x2,x2, _MM_SHUFFLE(0,3,2,1));
            x3 = _mm_shuffle_ps(x3,x3, _MM_SHUFFLE(0,3,2,1));

            x0 = _mm_mul_ps(x0, pFactor);
            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            px0 = _mm_cvtps_epi32(x0);
            px1 = _mm_cvtps_epi32(x1);
            px2 = _mm_cvtps_epi32(x2);
            px3 = _mm_cvtps_epi32(x3);

            px0 = _mm_packs_epi32(px0, px0);
            px0 = _mm_packus_epi16(px0, px0);
            *((int*)arrayPx0) = _mm_cvtsi128_si32(px0);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayPx1) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayPx2) = _mm_cvtsi128_si32(px2);

            px3 = _mm_packs_epi32(px3, px3);
            px3 = _mm_packus_epi16(px3, px3);
            *((int*)arrayPx3) = _mm_cvtsi128_si32(px3);

            memcpy(dstPtrTempPx0, arrayPx0, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx1, arrayPx1, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx2, arrayPx2, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx3, arrayPx3, 4 * sizeof(Rpp8u));

            srcPtrTempPx0 += 12;
            srcPtrTempPx1 += 12;
            srcPtrTempPx2 += 12;
            srcPtrTempPx3 += 12;
            dstPtrTempPx0 += 12;
            dstPtrTempPx1 += 12;
            dstPtrTempPx2 += 12;
            dstPtrTempPx3 += 12;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

            srcPtrTempR = srcPtrTempPx0;
            srcPtrTempG = srcPtrTempPx0 + 1;
            srcPtrTempB = srcPtrTempPx0 + 2;
            dstPtrTempR = dstPtrTempPx0;
            dstPtrTempG = dstPtrTempPx0 + 1;
            dstPtrTempB = dstPtrTempPx0 + 2;

            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Hue

            hue = hue + hueShiftAngle;
            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempPx0 += 3;
            dstPtrTempPx0 += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus hueRGB_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32f *batch_hueShift,
                         RppiROI *roiPoints, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f hueShiftAngle = hueShift;

            while (hueShiftAngle > 360)
            {
                hueShiftAngle = hueShiftAngle - 360;
            }
            while (hueShiftAngle < 0)
            {
                hueShiftAngle = 360 + hueShiftAngle;
            }

            hueShift = hueShiftAngle / 360;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                    dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp += batch_srcSizeMax[batchCount].width;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                    srcPtrTemp += x1;
                    dstPtrTemp += x1;

                    Rpp32u bufferLength = roiPoints[batchCount].roiWidth;

                    hueRGB_processBuffer_host(srcPtrTemp, batch_srcSizeMax[batchCount], dstPtrTemp, hueShift, hueShiftAngle, bufferLength, chnFormat, channel);

                    srcPtrTemp += bufferLength;
                    dstPtrTemp += bufferLength;

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32f hueShift = batch_hueShift[batchCount];
            Rpp32f hueShiftAngle = hueShift;

            while (hueShiftAngle > 360)
            {
                hueShiftAngle = hueShiftAngle - 360;
            }
            while (hueShiftAngle < 0)
            {
                hueShiftAngle = 360 + hueShiftAngle;
            }

            hueShift = hueShiftAngle / 360;

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f pixel;

                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                    srcPtrTemp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;

                    hueRGB_processBuffer_host(srcPtrTemp, batch_srcSizeMax[batchCount], dstPtrTemp, hueShift, hueShiftAngle, bufferLength, chnFormat, channel);

                    srcPtrTemp += bufferLength;
                    dstPtrTemp += bufferLength;
                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus hueRGB_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f hueShift,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f hueShiftAngle = hueShift;
    while (hueShiftAngle > 360)
    {
        hueShiftAngle = hueShiftAngle - 360;
    }
    while (hueShiftAngle < 0)
    {
        hueShiftAngle = 360 + hueShiftAngle;
    }

    hueShift = hueShiftAngle / 360;

    Rpp64u bufferLength;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        bufferLength = srcSize.height * srcSize.width;
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        bufferLength = channel * srcSize.height * srcSize.width;
    }

    hueRGB_processBuffer_host(srcPtr, srcSize, dstPtr, hueShift, hueShiftAngle, bufferLength, chnFormat, channel);

    return RPP_SUCCESS;
}

/**************** saturation ***************/

template <typename T>
RppStatus saturationRGB_processBuffer_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                           Rpp32f saturationFactor, Rpp64u bufferLength,
                                           RppiChnFormat chnFormat, Rpp32u channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
        T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u alignedLength = (bufferLength / 4) * 4;

        __m128i const zero = _mm_setzero_si128();
        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128i px0, px1, px2;
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        Rpp8u arrayR[4];
        Rpp8u arrayG[4];
        Rpp8u arrayB[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTempR);
            px1 =  _mm_loadu_si128((__m128i *)srcPtrTempG);
            px2 =  _mm_loadu_si128((__m128i *)srcPtrTempB);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            px0 = _mm_cvtps_epi32(x1);
            px1 = _mm_cvtps_epi32(x2);
            px2 = _mm_cvtps_epi32(x3);

            px0 = _mm_packs_epi32(px0, px0);
            px0 = _mm_packus_epi16(px0, px0);
            *((int*)arrayR) = _mm_cvtsi128_si32(px0);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayG) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayB) = _mm_cvtsi128_si32(px2);

            memcpy(dstPtrTempR, arrayR, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempG, arrayG, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempB, arrayB, 4 * sizeof(Rpp8u));

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Saturation

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        T *srcPtrTempPx0, *srcPtrTempPx1, *srcPtrTempPx2, *srcPtrTempPx3;
        T *dstPtrTempPx0, *dstPtrTempPx1, *dstPtrTempPx2, *dstPtrTempPx3;

        srcPtrTempPx0 = srcPtr;
        srcPtrTempPx1 = srcPtr + 3;
        srcPtrTempPx2 = srcPtr + 6;
        srcPtrTempPx3 = srcPtr + 9;
        dstPtrTempPx0 = dstPtr;
        dstPtrTempPx1 = dstPtr + 3;
        dstPtrTempPx2 = dstPtr + 6;
        dstPtrTempPx3 = dstPtr + 9;

        Rpp64u alignedLength = (bufferLength / 12) * 12;

        __m128i const zero = _mm_setzero_si128();
        __m128 pZeros = _mm_set1_ps(0.0);
        __m128 pOnes = _mm_set1_ps(1.0);
        __m128 pFactor = _mm_set1_ps(255.0);
        __m128 pSaturationFactor = _mm_set1_ps(saturationFactor);
        __m128i px0, px1, px2, px3;
        __m128 xR, xG, xB, xA;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;
        __m128 h0, h1, h2, h3;
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        h0 = _mm_set1_ps(1.0);

        Rpp8u arrayPx0[4];
        Rpp8u arrayPx1[4];
        Rpp8u arrayPx2[4];
        Rpp8u arrayPx3[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
        {
            px0 = _mm_loadu_si128((__m128i *)srcPtrTempPx0);
            px1 = _mm_loadu_si128((__m128i *)srcPtrTempPx1);
            px2 = _mm_loadu_si128((__m128i *)srcPtrTempPx2);
            px3 = _mm_loadu_si128((__m128i *)srcPtrTempPx3);

            px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
            px1 = _mm_unpacklo_epi8(px1, zero);    // pixels 0-7
            px2 = _mm_unpacklo_epi8(px2, zero);    // pixels 0-7
            px3 = _mm_unpacklo_epi8(px3, zero);    // pixels 0-7

            xR = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
            xG = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3
            xB = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3
            xA = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px3, zero));    // pixels 0-3

            _MM_TRANSPOSE4_PS (xR, xG, xB, xA);

            xR = _mm_div_ps(xR, pFactor);
            xG = _mm_div_ps(xG, pFactor);
            xB = _mm_div_ps(xB, pFactor);

            // Calculate Saturation, Value, Chroma
            xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
            xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

            xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
            xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

            xV = xS;                                               // xV <- [V    ]
            xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
            xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

            xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]

            // Calculate Hue
            xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
            xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

            xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
            xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

            xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
            xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]

            xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
            xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
            xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

            xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
            xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

            xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
            xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

            // G is the accumulator
            xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
            xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

            xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
            xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

            xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
            xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

            // Correct achromatic cases (H/S may be infinite due to zero division)
            xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
            xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
            xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

            xG = _mm_add_ps(xG, xH);

            // Normalize H to fraction
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Modify Saturation Values
            xS = _mm_mul_ps(xS, pSaturationFactor);
            xS = _mm_min_ps(xS, pOnes);
            xS = _mm_max_ps(xS, pZeros);

            _MM_TRANSPOSE4_PS (h0, xG, xS, xV);

            __m128 h1, h2, h3;

            h1 = xG;
            h2 = xS;
            h3 = xV;

            // Prepare HUE for RGB components (per pixel).
            x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
            x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
            x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
            x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

            // Calculate intervals from HUE.
            x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
            x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

            x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
            x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

            x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
            x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

            x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
            x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

            // Bound intervals.
            x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
            x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
            x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
            x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

            x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
            x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

            // Prepare S/V vectors.
            a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
            h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
            h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

            // Multiply with 'S*V' and add 'V'.
            x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
            a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

            x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
            h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

            x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
            x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
            x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
            x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

            // Store
            x0 = _mm_shuffle_ps(x0,x0, _MM_SHUFFLE(0,3,2,1));
            x1 = _mm_shuffle_ps(x1,x1, _MM_SHUFFLE(0,3,2,1));
            x2 = _mm_shuffle_ps(x2,x2, _MM_SHUFFLE(0,3,2,1));
            x3 = _mm_shuffle_ps(x3,x3, _MM_SHUFFLE(0,3,2,1));

            // _MM_TRANSPOSE4_PS (x0, x1, x2, x3);

            x0 = _mm_mul_ps(x0, pFactor);
            x1 = _mm_mul_ps(x1, pFactor);
            x2 = _mm_mul_ps(x2, pFactor);
            x3 = _mm_mul_ps(x3, pFactor);

            px0 = _mm_cvtps_epi32(x0);
            px1 = _mm_cvtps_epi32(x1);
            px2 = _mm_cvtps_epi32(x2);
            px3 = _mm_cvtps_epi32(x3);

            px0 = _mm_packs_epi32(px0, px0);
            px0 = _mm_packus_epi16(px0, px0);
            *((int*)arrayPx0) = _mm_cvtsi128_si32(px0);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayPx1) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayPx2) = _mm_cvtsi128_si32(px2);

            px3 = _mm_packs_epi32(px3, px3);
            px3 = _mm_packus_epi16(px3, px3);
            *((int*)arrayPx3) = _mm_cvtsi128_si32(px3);

            memcpy(dstPtrTempPx0, arrayPx0, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx1, arrayPx1, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx2, arrayPx2, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempPx3, arrayPx3, 4 * sizeof(Rpp8u));

            srcPtrTempPx0 += 12;
            srcPtrTempPx1 += 12;
            srcPtrTempPx2 += 12;
            srcPtrTempPx3 += 12;
            dstPtrTempPx0 += 12;
            dstPtrTempPx1 += 12;
            dstPtrTempPx2 += 12;
            dstPtrTempPx3 += 12;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
            T *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

            srcPtrTempR = srcPtrTempPx0;
            srcPtrTempG = srcPtrTempPx0 + 1;
            srcPtrTempB = srcPtrTempPx0 + 2;
            dstPtrTempR = dstPtrTempPx0;
            dstPtrTempG = dstPtrTempPx0 + 1;
            dstPtrTempB = dstPtrTempPx0 + 2;

            // RGB to HSV

            Rpp32f hue, sat, val;
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                hue = 0;
            }
            else if (cmax == rf)
            {
                hue = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                hue = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                hue = round(60 * (((rf - gf) / delta) + 4));
            }

            while (hue >= 360)
            {
                hue = hue - 360;
            }
            while (hue < 0)
            {
                hue = 360 + hue;
            }

            if (cmax == 0)
            {
                sat = 0;
            }
            else
            {
                sat = delta / cmax;
            }

            val = cmax;

            // Modify Saturation

            sat *= saturationFactor;
            sat = (sat < (Rpp32f) 1) ? sat : ((Rpp32f) 1);
            sat = (sat > (Rpp32f) 0) ? sat : ((Rpp32f) 0);

            // HSV to RGB

            Rpp32f c, x, m;
            c = val * sat;
            x = c * (1 - abs(int(fmod((hue / 60), 2)) - 1));
            m = val - c;

            if ((0 <= hue) && (hue < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= hue) && (hue < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= hue) && (hue < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= hue) && (hue < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= hue) && (hue < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= hue) && (hue < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempPx0 += 3;
            dstPtrTempPx0 += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus saturationRGB_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                         Rpp32f *batch_saturationFactor,
                         RppiROI *roiPoints, Rpp32u nbatchSize,
                         RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32f saturationFactor = batch_saturationFactor[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                    dstPtrTemp += batch_srcSizeMax[batchCount].width;
                    srcPtrTemp += batch_srcSizeMax[batchCount].width;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                    srcPtrTemp += x1;
                    dstPtrTemp += x1;

                    Rpp32u bufferLength = roiPoints[batchCount].roiWidth;

                    saturationRGB_processBuffer_host(srcPtrTemp, batch_srcSizeMax[batchCount], dstPtrTemp, saturationFactor, bufferLength, chnFormat, channel);

                    srcPtrTemp += bufferLength;
                    dstPtrTemp += bufferLength;

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0)
            {
                x2 = batch_srcSize[batchCount].width;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == 0)
            {
                y2 = batch_srcSize[batchCount].height;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            Rpp32f saturationFactor = batch_saturationFactor[batchCount];

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                    srcPtrTemp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    Rpp32u bufferLength = channel * roiPoints[batchCount].roiWidth;

                    saturationRGB_processBuffer_host(srcPtrTemp, batch_srcSizeMax[batchCount], dstPtrTemp, saturationFactor, bufferLength, chnFormat, channel);

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus saturationRGB_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f saturationFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp64u totalImageDim = channel * srcSize.height * srcSize.width;

    Rpp64u bufferLength;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        bufferLength = srcSize.height * srcSize.width;
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        bufferLength = channel * srcSize.height * srcSize.width;
    }

    saturationRGB_processBuffer_host(srcPtr, srcSize, dstPtr, saturationFactor, bufferLength, chnFormat, channel);

    return RPP_SUCCESS;
}

/**************** Tensor Look Up Table ***************/

template <typename T>
RppStatus tensor_look_up_table_host(T* srcPtr, T* dstPtr, T* lutPtr,
                          Rpp32u tensorDimension, Rpp32u *tensorDimensionValues)
{
    Rpp32u *tensorDimensionValuesTemp;
    tensorDimensionValuesTemp = tensorDimensionValues;

    Rpp32u tensorSize = 1;
    for(int i = 0; i < tensorDimension; i++)
    {
        tensorSize *= *tensorDimensionValuesTemp;
        tensorDimensionValuesTemp++;
    }

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < tensorSize; i++)
    {
        *dstPtrTemp = *(lutPtr + (Rpp32u)(*srcPtrTemp));
        srcPtrTemp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

// /**************** color_convert ***************/

template <typename T, typename U>
RppStatus color_convert_rgb_to_hsv_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr,
    RppiColorConvertMode convertMode, Rpp32u nbatchSize,
    RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtrImage = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        U *dstPtrImage = (U*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(U));

        compute_unpadded_from_padded_host(srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtrImage,
                                            chnFormat, channel);

        compute_rgb_to_hsv_host(srcPtrImage, batch_srcSize[batchCount], dstPtrImage, chnFormat, channel);

        compute_padded_from_unpadded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtr + loc,
                                          chnFormat, channel);

        free(srcPtrImage);
        free(dstPtrImage);
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus color_convert_hsv_to_rgb_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, U* dstPtr,
    RppiColorConvertMode convertMode, Rpp32u nbatchSize,
    RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
    {
        Rpp32u loc = 0;
        compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);

        T *srcPtrImage = (T*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(T));
        U *dstPtrImage = (U*) calloc(channel * batch_srcSize[batchCount].height * batch_srcSize[batchCount].width, sizeof(U));

        compute_unpadded_from_padded_host(srcPtr + loc, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], srcPtrImage,
                                            chnFormat, channel);

        compute_hsv_to_rgb_host(srcPtrImage, batch_srcSize[batchCount], dstPtrImage, chnFormat, channel);

        compute_padded_from_unpadded_host(dstPtrImage, batch_srcSize[batchCount], batch_srcSizeMax[batchCount], dstPtr + loc,
                                          chnFormat, channel);

        free(srcPtrImage);
        free(dstPtrImage);
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus color_convert_rgb_to_hsv_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiColorConvertMode convert_mode, RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_rgb_to_hsv_host(srcPtr, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus color_convert_hsv_to_rgb_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiColorConvertMode convert_mode, RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_hsv_to_rgb_host(srcPtr, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;
}

#endif