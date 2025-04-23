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

#ifndef TESTSUITE_HELPER
#define TESTSUITE_HELPER

#include "rppi.h"

RppStatus compute_image_location_host(RppiSize batch_srcSizeMax, int batchCount, Rpp32u *loc, Rpp32u channel)
{
    for (int m = 0; m < batchCount; m++)
    {
        *loc += (batch_srcSizeMax.height * batch_srcSizeMax.width);
    }
    *loc *= channel;

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_unpadded_from_padded_host(T* srcPtrPadded, RppiSize srcSize, RppiSize srcSizeMax, T* dstPtrUnpadded,
                                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrPaddedChannel, *srcPtrPaddedRow, *dstPtrUnpaddedRow;
    Rpp32u imageDimMax = srcSizeMax.height * srcSizeMax.width;
    dstPtrUnpaddedRow = dstPtrUnpadded;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrPaddedChannel = srcPtrPadded + (c * imageDimMax);
            for (int i = 0; i < srcSize.height; i++)
            {
                srcPtrPaddedRow = srcPtrPaddedChannel + (i * srcSizeMax.width);
                memcpy(dstPtrUnpaddedRow, srcPtrPaddedRow, srcSize.width * sizeof(T));
                dstPtrUnpaddedRow += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;
        for (int i = 0; i < srcSize.height; i++)
        {
            srcPtrPaddedRow = srcPtrPadded + (i * elementsInRowMax);
            memcpy(dstPtrUnpaddedRow, srcPtrPaddedRow, elementsInRow * sizeof(T));
            dstPtrUnpaddedRow += elementsInRow;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus compute_padded_from_unpadded_host(T* srcPtrUnpadded, RppiSize srcSize, RppiSize dstSizeMax, T* dstPtrPadded,
                                                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *dstPtrPaddedChannel, *dstPtrPaddedRow, *srcPtrUnpaddedRow;
    Rpp32u imageDimMax = dstSizeMax.height * dstSizeMax.width;
    srcPtrUnpaddedRow = srcPtrUnpadded;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            dstPtrPaddedChannel = dstPtrPadded + (c * imageDimMax);
            for (int i = 0; i < srcSize.height; i++)
            {
                dstPtrPaddedRow = dstPtrPaddedChannel + (i * dstSizeMax.width);
                memcpy(dstPtrPaddedRow, srcPtrUnpaddedRow, srcSize.width * sizeof(T));
                srcPtrUnpaddedRow += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRowMax = channel * dstSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;
        for (int i = 0; i < srcSize.height; i++)
        {
            dstPtrPaddedRow = dstPtrPadded + (i * elementsInRowMax);
            memcpy(dstPtrPaddedRow, srcPtrUnpaddedRow, elementsInRow * sizeof(T));
            srcPtrUnpaddedRow += elementsInRow;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
inline RppStatus generate_bressenham_line_host(T *dstPtr, RppiSize dstSize, Rpp32u *endpoints, Rpp32u *rasterCoordinates)
{
    Rpp32u *rasterCoordinatesTemp;
    rasterCoordinatesTemp = rasterCoordinates;

    Rpp32s x0 = *endpoints;
    Rpp32s y0 = *(endpoints + 1);
    Rpp32s x1 = *(endpoints + 2);
    Rpp32s y1 = *(endpoints + 3);

    Rpp32s dx, dy;
    Rpp32s stepX, stepY;

    dx = x1 - x0;
    dy = y1 - y0;

    if (dy < 0)
    {
        dy = -dy;
        stepY = -1;
    }
    else
    {
        stepY = 1;
    }

    if (dx < 0)
    {
        dx = -dx;
        stepX = -1;
    }
    else
    {
        stepX = 1;
    }

    dy <<= 1;
    dx <<= 1;

    if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
    {
        *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
        *rasterCoordinatesTemp = y0;
        rasterCoordinatesTemp++;
        *rasterCoordinatesTemp = x0;
        rasterCoordinatesTemp++;
    }

    if (dx > dy)
    {
        Rpp32s fraction = dy - (dx >> 1);
        while (x0 != x1)
        {
            x0 += stepX;
            if (fraction >= 0)
            {
                y0 += stepY;
                fraction -= dx;
            }
            fraction += dy;
            if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
            {
                *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
                *rasterCoordinatesTemp = y0;
                rasterCoordinatesTemp++;
                *rasterCoordinatesTemp = x0;
                rasterCoordinatesTemp++;
            }
        }
    }
    else
    {
        int fraction = dx - (dy >> 1);
        while (y0 != y1)
        {
            if (fraction >= 0)
            {
                x0 += stepX;
                fraction -= dy;
            }
            y0 += stepY;
            fraction += dx;
            if ((0 <= x0) && (x0 < dstSize.width) && (0 <= y0) && (y0 < dstSize.height))
            {
                *(dstPtr + (y0 * dstSize.width) + x0) = (T) 255;
                *rasterCoordinatesTemp = y0;
                rasterCoordinatesTemp++;
                *rasterCoordinatesTemp = x0;
                rasterCoordinatesTemp++;
            }
        }
    }

    return RPP_SUCCESS;
}

#endif