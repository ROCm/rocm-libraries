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

#ifndef RPP_CPU_BATCHPD_HPP
#define RPP_CPU_BATCHPD_HPP

#include <math.h>
#include <algorithm>
#include <typeinfo>
#include <cstring>
#include <omp.h>
#include <half/half.hpp>
using halfhpp = half_float::half;
typedef halfhpp Rpp16f;
#include "rppdefs.h"
#include "rpp_cpu_simd_load_store.hpp"
#include "rpp_cpu_simd_math.hpp"

#define PI                              3.14159265
#define RAD(deg)                        (deg * PI / 180)
#define RPPABS(a)                       ((a < 0) ? (-a) : (a))
#define RPPCEIL(a)                      ((int) (a + 1.0))
#define RPPMIN2(a,b)                    ((a < b) ? a : b)
#define RPPMIN3(a,b,c)                  ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define RPPMAX2(a,b)                    ((a > b) ? a : b)
#define RPPMAX3(a,b,c)                  ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define RPPINRANGE(a, x, y)             ((a >= x) && (a <= y) ? 1 : 0)
#define RPPPRANGECHECK(value, a, b)     (value < (Rpp32f) a) ? ((Rpp32f) a) : ((value < (Rpp32f) b) ? value : ((Rpp32f) b))
#define RPPFLOOR(a)                     ((int) a)
#define RPPISEVEN(a)                    ((a % 2 == 0) ? 1 : 0)
#define RPPPIXELCHECK(pixel)            (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255))
#define RPPPIXELCHECKF32(pixel)         (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 1) ? pixel : ((Rpp32f) 1))
#define RPPPIXELCHECKI8(pixel)          (pixel < (Rpp32f) -128) ? ((Rpp32f) -128) : ((pixel < (Rpp32f) 127) ? pixel : ((Rpp32f) 127))
#define RPPISGREATER(pixel, value)      ((pixel > value) ? 1 : 0)
#define RPPISLESSER(pixel, value)       ((pixel < value) ? 1 : 0)

/*Constants used for Gaussian interpolation*/
// Here sigma is considered as 0.5f
#define GAUSSCONSTANT1                 -2.0f          // 1 / (sigma * sigma * -1 * 2);
#define GAUSSCONSTANT2                  0.7978845608028654f // 1 / ((2 * PI)*(1/2) * sigma)
static uint16_t wyhash16_x;

inline uint32_t hash16(uint32_t input, uint32_t key) {
  uint32_t hash = input * key;
  return ((hash >> 16) ^ hash) & 0xFFFF;
}

inline uint16_t wyhash16() {
  wyhash16_x += 0xfc15;
  return hash16(wyhash16_x, 0x2ab);
}

inline uint16_t rand_range16(const uint16_t s) {
    uint16_t x = wyhash16();
    uint32_t m = (uint32_t)x * (uint32_t)s;
    uint16_t l = (uint16_t)m;
    if (l < s) {
        uint16_t t = -s % s;
        while (l < t) {
            x = wyhash16();
            m = (uint32_t)x * (uint32_t)s;
            l = (uint16_t)m;
        }
    }
    return m >> 16;
}

static unsigned int g_seed;

inline void fast_srand( int seed )
{
    g_seed = seed;
}

inline int fastrand()
{
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}

#if !GPU_SUPPORT
enum class RPPTensorDataType
{
    U8 = 0,
    FP32,
    FP16,
    I8,
};

struct RPPTensorFunctionMetaData
{
    RPPTensorDataType _in_type = RPPTensorDataType::U8;
    RPPTensorDataType _out_type = RPPTensorDataType::U8;
    RppiChnFormat _in_format = RppiChnFormat::RPPI_CHN_PACKED;
    RppiChnFormat _out_format = RppiChnFormat::RPPI_CHN_PLANAR;
    Rpp32u _in_channels = 3;

    RPPTensorFunctionMetaData(RppiChnFormat in_chn_format, RPPTensorDataType in_tensor_type,
                              RPPTensorDataType out_tensor_type, Rpp32u in_channels,
                              bool out_format_change) : _in_format(in_chn_format), _in_type(in_tensor_type),
                                                        _out_type(out_tensor_type), _in_channels(in_channels)
    {
        if (out_format_change)
        {
            if (_in_format == RPPI_CHN_PLANAR)
                _out_format = RppiChnFormat::RPPI_CHN_PACKED;
            else
                _out_format = RppiChnFormat::RPPI_CHN_PLANAR;
        }
        else
            _out_format = _in_format;
    }
};
#endif // GPU_SUPPORT

inline int power_function(int a, int b)
{
    int product = 1;
    for(int i = 0; i < b; i++)
        product *= product * a;
    return product;
}

template <typename T>
RppStatus subtract_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus add_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus multiply_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus min_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus max_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);

template <typename T>
RppStatus bitwise_AND_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus inclusive_OR_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);
template <typename T>
RppStatus exclusive_OR_host_batch(T* srcPtr1, T* srcPtr2, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
                              RppiChnFormat chnFormat, Rpp32u channel);

// Specific Helper Functions

inline Rpp32f gaussian_2d_relative(Rpp32s locI, Rpp32s locJ, Rpp32f std_dev)
{
    Rpp32f relativeGaussian;
    Rpp32f exp1, exp2;
    exp1 = -(locJ * locJ) / (2 * std_dev * std_dev);
    exp2 = -(locI * locI) / (2 * std_dev * std_dev);
    relativeGaussian = exp(exp1 + exp2);

    return relativeGaussian;
}

// Generate Functions

inline void generate_gaussian_kernel_host(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f s, sum = 0.0, multiplier;
    int bound = ((kernelSize - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }
}

inline RppStatus generate_gaussian_kernel_asymmetric_host(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSizeX, Rpp32u kernelSizeY)
{
    Rpp32f s, sum = 0.0, multiplier;
    if (kernelSizeX % 2 == 0)
    {
        return RPP_ERROR;
    }
    if (kernelSizeY % 2 == 0)
    {
        return RPP_ERROR;
    }
    int boundX = ((kernelSizeX - 1) / 2);
    int boundY = ((kernelSizeY - 1) / 2);
    Rpp32u c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -boundY; i <= boundY; i++)
    {
        for (int j = -boundX; j <= boundX; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSizeX * kernelSizeY); i++)
    {
        kernel[i] /= sum;
    }
    return RPP_SUCCESS;
}

template <typename T>
inline void generate_bilateral_kernel_host(Rpp32f multiplierI, Rpp32f multiplierS, Rpp32f multiplier, Rpp32f* kernel, Rpp32u kernelSize, int bound,
                                         T* srcPtrWindow, RppiSize srcSizeMod, Rpp32u remainingElementsInRow, Rpp32u incrementToWindowCenter,
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f sum = 0.0;
    Rpp32f* kernelTemp;
    kernelTemp = kernel;

    T *srcPtrWindowTemp, *srcPtrWindowCenter;
    srcPtrWindowTemp = srcPtrWindow;
    srcPtrWindowCenter = srcPtrWindow + incrementToWindowCenter;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                T pixel = *srcPtrWindowCenter - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                pixel = pixel * pixel;
                *kernelTemp = multiplier * exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum = sum + *kernelTemp;
                kernelTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                T pixel = *srcPtrWindowCenter - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                pixel = pixel * pixel;
                *kernelTemp = multiplier * exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum = sum + *kernelTemp;
                kernelTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }

    kernelTemp = kernel;
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        *kernelTemp = *kernelTemp / sum;
        kernelTemp++;
    }
}

template <typename T>
inline RppStatus generate_evenly_padded_image_host(T* srcPtr, RppiSize srcSize, T* srcPtrMod, RppiSize srcSizeMod,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    if (RPPISEVEN(srcSize.height) != RPPISEVEN(srcSizeMod.height)
        || RPPISEVEN(srcSize.width) != RPPISEVEN(srcSizeMod.width)
        || srcSizeMod.height < srcSize.height
        || srcSizeMod.width < srcSize.width)
    {
        return RPP_ERROR;
    }
    T *srcPtrTemp, *srcPtrModTemp;
    srcPtrTemp = srcPtr;
    srcPtrModTemp = srcPtrMod;
    int bound = (srcSizeMod.height - srcSize.height) / 2;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            memset (srcPtrModTemp,(T) 0,bound * srcSizeMod.width * sizeof(T));
            srcPtrModTemp += (bound * srcSizeMod.width);
            for (int i = 0; i < srcSize.height; i++)
            {
                memset (srcPtrModTemp,(T) 0,bound * sizeof(T));
                srcPtrModTemp += bound;

                memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                srcPtrModTemp += srcSize.width;
                srcPtrTemp += srcSize.width;

                memset (srcPtrModTemp,(T) 0,bound * sizeof(T));
                srcPtrModTemp += bound;
            }
            memset (srcPtrModTemp,(T) 0,bound * srcSizeMod.width * sizeof(T));
            srcPtrModTemp += (bound * srcSizeMod.width);
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u numOfPixelsVtBorder = bound * channel;
        Rpp32u numOfPixelsHrBorder = numOfPixelsVtBorder * srcSizeMod.width;

        memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
        srcPtrModTemp += (numOfPixelsHrBorder);

        for (int i = 0; i < srcSize.height; i++)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsVtBorder);

            memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
            srcPtrModTemp += elementsInRow;
            srcPtrTemp += elementsInRow;

            memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsVtBorder);
        }

        memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
        srcPtrModTemp += (numOfPixelsHrBorder);
    }

    return RPP_SUCCESS;
}

template <typename T>
inline void generate_corner_padded_image_host(T* srcPtr, RppiSize srcSize, T* srcPtrMod, RppiSize srcSizeMod, Rpp32u padType,
                                     RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *srcPtrModTemp;
    srcPtrTemp = srcPtr;
    srcPtrModTemp = srcPtrMod;
    Rpp32u boundY = srcSizeMod.height - srcSize.height;
    Rpp32u boundX = srcSizeMod.width - srcSize.width;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            if (padType == 1 || padType == 2)
            {
                memset (srcPtrModTemp,(T) 0,boundY * srcSizeMod.width * sizeof(T));
                srcPtrModTemp += (boundY * srcSizeMod.width);
            }

            if (padType == 1 || padType == 3)
            {
                for (int i = 0; i < srcSize.height; i++)
                {
                    memset (srcPtrModTemp,(T) 0,boundX * sizeof(T));
                    srcPtrModTemp += boundX;

                    memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    srcPtrModTemp += srcSize.width;
                    srcPtrTemp += srcSize.width;
                }
            }

            if (padType == 2 || padType == 4)
            {
                for (int i = 0; i < srcSize.height; i++)
                {
                    memcpy(srcPtrModTemp, srcPtrTemp, srcSize.width * sizeof(T));
                    srcPtrModTemp += srcSize.width;
                    srcPtrTemp += srcSize.width;

                    memset (srcPtrModTemp,(T) 0,boundX * sizeof(T));
                    srcPtrModTemp += boundX;
                }
            }

            if (padType == 3 || padType == 4)
            {
                memset (srcPtrModTemp,(T) 0,boundY * srcSizeMod.width * sizeof(T));
                srcPtrModTemp += (boundY * srcSizeMod.width);
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u numOfPixelsVtBorder = boundX * channel;
        Rpp32u numOfPixelsHrBorder = boundY * channel * srcSizeMod.width;

        if (padType == 1 || padType == 2)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsHrBorder);
        }

        if (padType == 1 || padType == 3)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
                srcPtrModTemp += (numOfPixelsVtBorder);

                memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
                srcPtrModTemp += elementsInRow;
                srcPtrTemp += elementsInRow;
            }
        }

        if (padType == 2 || padType == 4)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                memcpy(srcPtrModTemp, srcPtrTemp, elementsInRow * sizeof(T));
                srcPtrModTemp += elementsInRow;
                srcPtrTemp += elementsInRow;

                memset (srcPtrModTemp,(T) 0,numOfPixelsVtBorder * sizeof(T));
                srcPtrModTemp += (numOfPixelsVtBorder);
            }
        }

        if (padType == 3 || padType == 4)
        {
            memset (srcPtrModTemp,(T) 0,numOfPixelsHrBorder * sizeof(T));
            srcPtrModTemp += (numOfPixelsHrBorder);
        }
    }
}

inline void generate_box_kernel_host(Rpp32f* kernel, Rpp32u kernelSize)
{
    Rpp32f* kernelTemp;
    kernelTemp = kernel;
    Rpp32f kernelValue = 1.0 / (Rpp32f) (kernelSize * kernelSize);
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        *kernelTemp = kernelValue;
        kernelTemp++;
    }
}

template <typename T>
inline void generate_crop_host(T* srcPtr, RppiSize srcSize, T* srcPtrSubImage, RppiSize srcSizeSubImage, T* dstPtr,
                             RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrSubImageTemp, *dstPtrTemp;
    srcPtrSubImageTemp = srcPtrSubImage;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = (srcSize.width - srcSizeSubImage.width);
        for (int c = 0; c < channel; c++)
        {
            srcPtrSubImageTemp = srcPtrSubImage + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
                Rpp32u bufferLength = srcSizeSubImage.width;
                Rpp32u alignedLength = bufferLength & ~15;

                __m128i px0;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrSubImageTemp);
                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                    srcPtrSubImageTemp +=16;
                    dstPtrTemp +=16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = *srcPtrSubImageTemp++;
                }
                srcPtrSubImageTemp += remainingElementsInRow;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = channel * (srcSize.width - srcSizeSubImage.width);
        Rpp32u elementsInRowCrop = channel * srcSizeSubImage.width;
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
            Rpp32u bufferLength = elementsInRowCrop;
            Rpp32u alignedLength = bufferLength & ~15;

            __m128i px0;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrSubImageTemp);
                _mm_storeu_si128((__m128i *)dstPtrTemp, px0);
                srcPtrSubImageTemp +=16;
                dstPtrTemp +=16;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                *dstPtrTemp++ = *srcPtrSubImageTemp++;
            }
            srcPtrSubImageTemp += remainingElementsInRow;
        }
    }
}

inline RppStatus generate_sobel_kernel_host(Rpp32f* kernel, Rpp32u type)
{
    Rpp32f* kernelTemp;
    kernelTemp = kernel;

    if (type == 1)
    {
        Rpp32f kernelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        Rpp32f* kernelXTemp;
        kernelXTemp = kernelX;

        for (int i = 0; i < 9; i++)
        {
            *kernelTemp = *kernelXTemp;
            kernelTemp++;
            kernelXTemp++;
        }
    }
    else if (type == 2)
    {
        Rpp32f kernelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        Rpp32f* kernelYTemp;
        kernelYTemp = kernelY;

        for (int i = 0; i < 9; i++)
        {
            *kernelTemp = *kernelYTemp;
            kernelTemp++;
            kernelYTemp++;
        }
    }
    else
    {
        return RPP_ERROR;
    }

    return RPP_SUCCESS;
}

template <typename T>
inline void generate_bressenham_line_host(T *dstPtr, RppiSize dstSize, Rpp32u *endpoints, Rpp32u *rasterCoordinates)
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
}

// Kernels for functions

template<typename T, typename U>
inline void convolution_kernel_host(T* srcPtrWindow, U* dstPtrPixel, RppiSize srcSize,
                                       Rpp32f* kernel, RppiSize kernelSize, Rpp32u remainingElementsInRow, U maxVal, U minVal,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f pixel = 0.0;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    Rpp32f* kernelPtrTemp;
    kernelPtrTemp = kernel;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize.height; m++)
        {
            for (int n = 0; n < kernelSize.width; n++)
            {
                pixel += ((*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize.height; m++)
        {
            for (int n = 0; n < kernelSize.width; n++)
            {
                pixel += ((*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    (pixel < (Rpp32f) minVal) ? pixel = (Rpp32f) minVal : ((pixel < (Rpp32f) maxVal) ? pixel : pixel = (Rpp32f) maxVal);
    *dstPtrPixel = (U) round(pixel);
}

template<typename T>
inline void histogram_kernel_host(T* srcPtr, RppiSize srcSize, Rpp32u* histogram,
                                Rpp8u bins,
                                Rpp32u channel)
{
    if (bins == 0)
    {
        *histogram = channel * srcSize.height * srcSize.width;
    }
    else
    {
        Rpp8u rangeInBin = 256 / (bins + 1);
        T *srcPtrTemp;
        srcPtrTemp = srcPtr;
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            *(histogram + (*srcPtrTemp / rangeInBin)) += 1;
            srcPtrTemp++;
        }
    }
}

template <typename T, typename U>
inline void accumulate_kernel_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
                                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *srcPtr1Temp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
    }
}

template <typename U>
inline void normalize_kernel_host(U* dstPtrROI, RppiSize dstSize, Rpp32u channel)
{
    U* dstPtrROITemp;
    dstPtrROITemp = dstPtrROI;

    U multiplier = (U) (1.0 / 255.0);

    Rpp32u imageDim = dstSize.height * dstSize.width * channel;

    for (int i = 0; i < imageDim; i++)
    {
        *dstPtrROITemp = *dstPtrROITemp * multiplier;
        dstPtrROITemp++;
    }
}

template <typename T, typename U>
inline RppStatus resize_kernel_host(T* srcPtr, RppiSize srcSize, U* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (dstSize.height < 0 || dstSize.width < 0)
        {
            return RPP_ERROR;
        }

        Rpp32f hRatio = (((Rpp32f) (dstSize.height - 1)) / ((Rpp32f) (srcSize.height - 1)));
        Rpp32f wRatio = (((Rpp32f) (dstSize.width - 1)) / ((Rpp32f) (srcSize.width - 1)));
        Rpp32f srcLocationRow, srcLocationColumn, pixel;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        T *srcPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
        U *dstPtrTemp;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;

        if ((typeid(Rpp16f) == typeid(T)) || (typeid(Rpp16f) == typeid(U)))
        {
            for (int c = 0; c < channel; c++)
            {
                for (int i = 0; i < dstSize.height; i++)
                {
                    srcLocationRow = ((Rpp32f) i) / hRatio;
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    if (srcLocationRowFloor > (srcSize.height - 2))
                    {
                        srcLocationRowFloor = srcSize.height - 2;
                    }

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                    srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                    for (int j = 0; j < dstSize.width; j++)
                    {
                        srcLocationColumn = ((Rpp32f) j) / wRatio;
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        if (srcLocationColumnFloor > (srcSize.width - 2))
                        {
                            srcLocationColumnFloor = srcSize.width - 2;
                        }
                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (U) pixel;
                        dstPtrTemp ++;
                    }
                }
                srcPtrTemp += srcSize.height * srcSize.width;
            }
        }
        else
        {
            for (int c = 0; c < channel; c++)
            {
                for (int i = 0; i < dstSize.height; i++)
                {
                    srcLocationRow = ((Rpp32f) i) / hRatio;
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    if (srcLocationRowFloor > (srcSize.height - 2))
                    {
                        srcLocationRowFloor = srcSize.height - 2;
                    }

                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                    srcPtrBottomRow  = srcPtrTopRow + srcSize.width;
#pragma omp simd
                    for (int j = 0; j < dstSize.width; j++)
                    {
                        srcLocationColumn = ((Rpp32f) j) / wRatio;
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                        if (srcLocationColumnFloor > (srcSize.width - 2))
                        {
                            srcLocationColumnFloor = srcSize.width - 2;
                        }
                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth))
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (U) pixel;
                        dstPtrTemp ++;
                    }
                }
                srcPtrTemp += srcSize.height * srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (dstSize.height < 0 || dstSize.width < 0)
        {
            return RPP_ERROR;
        }

        Rpp32f hRatio = (((Rpp32f) (dstSize.height - 1)) / ((Rpp32f) (srcSize.height - 1)));
        Rpp32f wRatio = (((Rpp32f) (dstSize.width - 1)) / ((Rpp32f) (srcSize.width - 1)));
        Rpp32f srcLocationRow, srcLocationColumn, pixel;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        T *srcPtrTemp;
        U *dstPtrTemp;
        srcPtrTemp = srcPtr;
        dstPtrTemp = dstPtr;

        Rpp32u heightLimit = srcSize.height - 2;
        Rpp32u widthLimit = srcSize.width - 2;

        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / hRatio;
            srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

            if (srcLocationRowFloor > heightLimit)
            {
                srcLocationRowFloor = heightLimit;
            }

            T *srcPtrTopRow, *srcPtrBottomRow;
            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            Rpp32u bufferLength = dstSize.width;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32u srcLocCF[4] = {0};
            Rpp32f param1[4] = {0};
            Rpp32f param2[4] = {0};
            Rpp32f param3[4] = {0};
            Rpp32f param4[4] = {0};

            __m128 pWRatio = _mm_set1_ps(1.0 / wRatio);
            __m128 p0, p2, p4, p5, p6, p7, pColFloor;
            __m128 p1 = _mm_set1_ps(weightedHeight);
            __m128 p3 = _mm_set1_ps(1 - weightedHeight);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128i pxColFloor;

            Rpp64u vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
            {
                p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                p0 = _mm_mul_ps(p0, pWRatio);
                pColFloor = _mm_floor_ps(p0);
                pxColFloor = _mm_cvtps_epi32(pColFloor);
                p0 = _mm_sub_ps(p0, pColFloor);
                p2  = _mm_sub_ps(pOne, p0);

                p4 = _mm_mul_ps(p3, p2);
                p5 = _mm_mul_ps(p3, p0);
                p6 = _mm_mul_ps(p1, p2);
                p7 = _mm_mul_ps(p1, p0);

                _mm_storeu_si128((__m128i*) srcLocCF, pxColFloor);
                _mm_storeu_ps(param1, p4);
                _mm_storeu_ps(param2, p5);
                _mm_storeu_ps(param3, p6);
                _mm_storeu_ps(param4, p7);

                for (int pos = 0; pos < 4; pos++)
                {
                    if (srcLocCF[pos] > widthLimit)
                    {
                        srcLocCF[pos] = widthLimit;
                    }
                    srcLocCF[pos] *= channel;

                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp++ = (U) ((*(srcPtrTopRow + c + srcLocCF[pos])) * param1[pos])
                                            + ((*(srcPtrTopRow + c + srcLocCF[pos] + channel)) * param2[pos])
                                            + ((*(srcPtrBottomRow + c + srcLocCF[pos])) * param3[pos])
                                            + ((*(srcPtrBottomRow + c + srcLocCF[pos] + channel)) * param4[pos]);
                    }
                }
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount++)
            {
                srcLocationColumn = ((Rpp32f) vectorLoopCount) / wRatio;
                srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                if (srcLocationColumnFloor > (srcSize.width - 2))
                {
                    srcLocationColumnFloor = srcSize.width - 2;
                }

                Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                for (int c = 0; c < channel; c++)
                {
                    pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth))
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));

                    *dstPtrTemp = (U) pixel;
                    dstPtrTemp ++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
inline void resize_crop_kernel_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                           RppiChnFormat chnFormat, Rpp32u channel)
{
    RppiSize srcSizeSubImage;
    T *srcPtrSubImage;

    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    T *srcPtrResize = (T*) calloc(channel * srcSizeSubImage.height * srcSizeSubImage.width, sizeof(T));

    generate_crop_host(srcPtr, srcSize, srcPtrSubImage, srcSizeSubImage, srcPtrResize, chnFormat, channel);

    resize_kernel_host(srcPtrResize, srcSizeSubImage, dstPtr, dstSize, chnFormat, channel);

    free(srcPtrResize);
}

template<typename T>
inline void erode_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp < pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp < pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    *dstPtrPixel = pixel;
}

template<typename T>
inline void dilate_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    *dstPtrPixel = pixel;
}

template<typename T>
inline void median_filter_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T *kernel = (T*)calloc(kernelSize * kernelSize, sizeof(T));
    T *kernelTemp;
    kernelTemp = kernel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *kernelTemp = *srcPtrWindowTemp;
                srcPtrWindowTemp++;
                kernelTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *kernelTemp = *srcPtrWindowTemp;
                srcPtrWindowTemp += channel;
                kernelTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }

    std::sort(kernel, kernel + (kernelSize * kernelSize));

    *dstPtrPixel = *(kernel + (((kernelSize * kernelSize) - 1) / 2));

    free(kernel);
}

template<typename T>
inline void local_binary_pattern_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u remainingElementsInRow, T* centerPixelPtr,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    int pixel = (int) 0;
    T *srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 0);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 1);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 2);
        }
        srcPtrWindowTemp++;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 7);
        }
        srcPtrWindowTemp += 2;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 3);
        }
        srcPtrWindowTemp++;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 6);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 5);
        }
        srcPtrWindowTemp++;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 4);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 0);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 1);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 2);
        }
        srcPtrWindowTemp += channel;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 7);
        }
        srcPtrWindowTemp += (2 * channel);

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 3);
        }
        srcPtrWindowTemp += channel;
        srcPtrWindowTemp += remainingElementsInRow;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 6);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 5);
        }
        srcPtrWindowTemp += channel;

        if (*srcPtrWindowTemp - *centerPixelPtr >= 0)
        {
            pixel += pow(2, 4);
        }
    }

    *dstPtrPixel = (T) RPPPIXELCHECK(pixel);
}

template<typename T>
inline void non_max_suppression_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, T windowCenter,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T pixel;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    pixel = *srcPtrWindowTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                if (*srcPtrWindowTemp > pixel)
                {
                    pixel = *srcPtrWindowTemp;
                }
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    if (windowCenter >= pixel)
    {
        *dstPtrPixel = windowCenter;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }
}

template<typename T>
inline void canny_non_max_suppression_kernel_host(T* dstPtrPixel, T windowCenter, T *position1Ptr, T *position2Ptr)
{
    if ((windowCenter >= *position1Ptr) && (windowCenter >= *position2Ptr))
    {
        *dstPtrPixel = windowCenter;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }
}

template<typename T>
inline void canny_hysterisis_edge_tracing_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                       Rpp32u kernelSize, Rpp32u remainingElementsInRow, T windowCenter, Rpp32u bound,
                                       RppiChnFormat chnFormat, Rpp32u channel)
{
    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;

    for (int m = 0; m < kernelSize; m++)
    {
        for (int n = 0; n < kernelSize; n++)
        {
            if (*srcPtrWindowTemp == (T) 255)
            {
                *dstPtrPixel = (T) 255;
            }
            srcPtrWindowTemp++;
        }
        srcPtrWindowTemp += remainingElementsInRow;
    }
    *dstPtrPixel = (T) 0;
}

template<typename T, typename U>
inline void harris_corner_detector_kernel_host(T* srcPtrWindowX, T* srcPtrWindowY, U* dstPtrPixel, RppiSize srcSize,
                                             Rpp32u kernelSize, Rpp32u remainingElementsInRow, Rpp32f kValue, Rpp32f threshold,
                                             RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f pixel;

    T *srcPtrWindowTempX, *srcPtrWindowTempY;
    srcPtrWindowTempX = srcPtrWindowX;
    srcPtrWindowTempY = srcPtrWindowY;

    Rpp32f sumXX = 0, sumYY = 0, sumXY = 0;

    for (int m = 0; m < kernelSize; m++)
    {
        for (int n = 0; n < kernelSize; n++)
        {
            Rpp32f valX = (Rpp32f) *srcPtrWindowTempX;
            Rpp32f valY = (Rpp32f) *srcPtrWindowTempY;
            sumXX += (valX * valX);
            sumYY += (valY * valY);
            sumXY += (valX * valY);

            srcPtrWindowTempX++;
            srcPtrWindowTempY++;
        }
        srcPtrWindowTempX += remainingElementsInRow;
        srcPtrWindowTempY += remainingElementsInRow;
    }
    Rpp32f det = (sumXX * sumYY) - (sumXY * sumXY);
    Rpp32f trace = sumXX + sumYY;
    pixel = (det) - (kValue * trace * trace);

    if (pixel > threshold)
    {
        *dstPtrPixel = (U) pixel;
    }
    else
    {
        *dstPtrPixel = (U) 0;
    }
}

template<typename T>
inline void harris_corner_set_maximum_kernel_host(T* dstPtrWindow, Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    T* dstPtrWindowTemp;
    dstPtrWindowTemp = dstPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 255;
                dstPtrWindowTemp++;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 255;
                dstPtrWindowTemp += channel;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
}

template<typename T>
inline void harris_corner_set_minimum_kernel_host(T* dstPtrWindow, Rpp32u kernelSize, Rpp32u remainingElementsInRow,
                                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    T* dstPtrWindowTemp;
    dstPtrWindowTemp = dstPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 0;
                dstPtrWindowTemp++;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                *dstPtrWindowTemp = (T) 0;
                dstPtrWindowTemp += channel;
            }
            dstPtrWindowTemp += remainingElementsInRow;
        }
    }
}

inline void tensor_index_exchange_kernel_host(Rpp32u *loopCount, Rpp32u *loopCountTransposed, Rpp32u tensorDimension, Rpp32u dimension1, Rpp32u dimension2)
{
    memcpy(loopCountTransposed, loopCount, tensorDimension * sizeof(Rpp32u));

    loopCountTransposed[dimension2] = loopCount[dimension1];
    loopCountTransposed[dimension1] = loopCount[dimension2];
}

template<typename T>
inline void tensor_transpose_iterate_kernel_host(T* srcPtr, T* dstPtr,
                                               Rpp32u tensorDimensionTemp, Rpp32u tensorDimension,
                                               Rpp32u *tensorDimensionValues, Rpp32u *tensorDimensionValuesProduct,
                                               Rpp32u *loopCount, Rpp32u *loopCountTransposed,
                                               Rpp32u dimension1, Rpp32u dimension2)
{
    if (tensorDimensionTemp >= tensorDimension)
    {
        Rpp32u dstPtrLoc = 0;
        for (int i = tensorDimension - 1; i > 0 ; i--)
        {
            dstPtrLoc = dstPtrLoc + (loopCount[i] * tensorDimensionValuesProduct[i - 1]);
        }
        dstPtrLoc += loopCount[0];

        tensor_index_exchange_kernel_host(loopCount, loopCountTransposed, tensorDimension, dimension1, dimension2);

        Rpp32u srcPtrLoc = 0;
        for (int i = tensorDimension - 1; i > 0 ; i--)
        {
            srcPtrLoc = srcPtrLoc + (loopCountTransposed[i] * tensorDimensionValuesProduct[i - 1]);
        }
        srcPtrLoc += loopCountTransposed[0];

        *(dstPtr + dstPtrLoc) = *(srcPtr + srcPtrLoc);
    }
    for (int i = 0; i < *(tensorDimensionValues + tensorDimensionTemp); i++)
    {
        *(loopCount + tensorDimensionTemp) = i;
        tensor_transpose_iterate_kernel_host(srcPtr, dstPtr,
                                             tensorDimensionTemp + 1, tensorDimension,
                                             tensorDimensionValues, tensorDimensionValuesProduct,
                                             loopCount, loopCountTransposed,
                                             dimension1, dimension2);
    }
}

template<typename T>
inline void fast_corner_detector_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize,
                                           Rpp32u* bresenhamCirclePositions, T threshold, Rpp32u numOfPixels)
{
    T centerPixel = *(srcPtrWindow + (3 * srcSize.width) + 3);
    T max = (T) (RPPPIXELCHECK((Rpp32s) centerPixel + (Rpp32s) threshold));
    T min = (T) (RPPPIXELCHECK((Rpp32s) centerPixel - (Rpp32s) threshold));

    // Find Bresenham Circle for the pixel

    Rpp32u *bresenhamCirclePositionsTemp;
    bresenhamCirclePositionsTemp = bresenhamCirclePositions;

    T *bresenhamCircle = (T*) calloc(16, sizeof(T));
    T *bresenhamCircleTemp;
    bresenhamCircleTemp = bresenhamCircle;

    T* bresenhamCircleOutput = (T*) calloc(16, sizeof(T));

    for (int i = 0; i < 16; i++)
    {
        *bresenhamCircleTemp = *(srcPtrWindow + *bresenhamCirclePositionsTemp);
        bresenhamCircleTemp++;
        bresenhamCirclePositionsTemp++;
    }

    Rpp32u flag = 0;

    *bresenhamCircleOutput = (T) RPPISLESSER(*bresenhamCircle, min);
    *(bresenhamCircleOutput + 8) = (T) RPPISLESSER(*(bresenhamCircle + 8), min);

    if (*bresenhamCircleOutput == 1)
    {
        *(bresenhamCircleOutput + 4) = (T) RPPISLESSER(*(bresenhamCircle + 4), min);
        *(bresenhamCircleOutput + 12) = (T) RPPISLESSER(*(bresenhamCircle + 12), min);
        if (*(bresenhamCircleOutput + 8) == 1)
        {
            if (*(bresenhamCircleOutput + 4) == 1 || *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 1;
            }
        }
        else if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
        {
            flag = 1;
        }
    }
    else if (*(bresenhamCircleOutput + 8) == 1)
    {
        *(bresenhamCircleOutput + 4) = (T) RPPISLESSER(*(bresenhamCircle + 4), min);
        *(bresenhamCircleOutput + 12) = (T) RPPISLESSER(*(bresenhamCircle + 12), min);
        if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
        {
            flag = 1;
        }
    }
    if (flag == 0)
    {
        *bresenhamCircleOutput = (T) RPPISGREATER(*bresenhamCircle, max);
        *(bresenhamCircleOutput + 8) = (T) RPPISGREATER(*(bresenhamCircle + 8), max);

        if (*bresenhamCircleOutput == 1)
        {
            *(bresenhamCircleOutput + 4) = (T) RPPISGREATER(*(bresenhamCircle + 4), max);
            *(bresenhamCircleOutput + 12) = (T) RPPISGREATER(*(bresenhamCircle + 12), max);
            if (*(bresenhamCircleOutput + 8) == 1)
            {
                if (*(bresenhamCircleOutput + 4) == 1 || *(bresenhamCircleOutput + 12) == 1)
                {
                    flag = 2;
                }
            }
            else if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 2;
            }
        }
        else if (*(bresenhamCircleOutput + 8) == 1)
        {
            *(bresenhamCircleOutput + 4) = (T) RPPISGREATER(*(bresenhamCircle + 4), max);
            *(bresenhamCircleOutput + 12) = (T) RPPISGREATER(*(bresenhamCircle + 12), max);
            if (*(bresenhamCircleOutput + 4) == 1 && *(bresenhamCircleOutput + 12) == 1)
            {
                flag = 2;
            }
        }
    }
    if (flag == 0)
    {
        *dstPtrPixel = (T) 0;
    }
    else if (flag == 1)
    {
        *(bresenhamCircleOutput + 1) = (T) RPPISLESSER(*(bresenhamCircle + 1), min);
        *(bresenhamCircleOutput + 2) = (T) RPPISLESSER(*(bresenhamCircle + 2), min);
        *(bresenhamCircleOutput + 3) = (T) RPPISLESSER(*(bresenhamCircle + 3), min);
        *(bresenhamCircleOutput + 5) = (T) RPPISLESSER(*(bresenhamCircle + 5), min);
        *(bresenhamCircleOutput + 6) = (T) RPPISLESSER(*(bresenhamCircle + 6), min);
        *(bresenhamCircleOutput + 7) = (T) RPPISLESSER(*(bresenhamCircle + 7), min);
        *(bresenhamCircleOutput + 9) = (T) RPPISLESSER(*(bresenhamCircle + 9), min);
        *(bresenhamCircleOutput + 10) = (T) RPPISLESSER(*(bresenhamCircle + 10), min);
        *(bresenhamCircleOutput + 11) = (T) RPPISLESSER(*(bresenhamCircle + 11), min);
        *(bresenhamCircleOutput + 13) = (T) RPPISLESSER(*(bresenhamCircle + 13), min);
        *(bresenhamCircleOutput + 14) = (T) RPPISLESSER(*(bresenhamCircle + 14), min);
        *(bresenhamCircleOutput + 15) = (T) RPPISLESSER(*(bresenhamCircle + 15), min);
    }
    else if (flag == 2)
    {
        *(bresenhamCircleOutput + 1) = (T) RPPISGREATER(*(bresenhamCircle + 1), max);
        *(bresenhamCircleOutput + 2) = (T) RPPISGREATER(*(bresenhamCircle + 2), max);
        *(bresenhamCircleOutput + 3) = (T) RPPISGREATER(*(bresenhamCircle + 3), max);
        *(bresenhamCircleOutput + 5) = (T) RPPISGREATER(*(bresenhamCircle + 5), max);
        *(bresenhamCircleOutput + 6) = (T) RPPISGREATER(*(bresenhamCircle + 6), max);
        *(bresenhamCircleOutput + 7) = (T) RPPISGREATER(*(bresenhamCircle + 7), max);
        *(bresenhamCircleOutput + 9) = (T) RPPISGREATER(*(bresenhamCircle + 9), max);
        *(bresenhamCircleOutput + 10) = (T) RPPISGREATER(*(bresenhamCircle + 10), max);
        *(bresenhamCircleOutput + 11) = (T) RPPISGREATER(*(bresenhamCircle + 11), max);
        *(bresenhamCircleOutput + 13) = (T) RPPISGREATER(*(bresenhamCircle + 13), max);
        *(bresenhamCircleOutput + 14) = (T) RPPISGREATER(*(bresenhamCircle + 14), max);
        *(bresenhamCircleOutput + 15) = (T) RPPISGREATER(*(bresenhamCircle + 15), max);
    }

    // Find maximum contiguous pixels in bresenhamCircleOutput with value 1

    Rpp32u count = 0;
    Rpp32u maxLength = 0;

    for (int i = 0; i < 32; i++)
    {
        if (*(bresenhamCircleOutput + (i % 16)) == 0)
        {
            count = 0;
            if (i >= 16)
            {
                break;
            }
        }
        else
        {
            count++;
            maxLength = RPPMAX2(maxLength, count);
        }
    }

    // Corner Classification

    if (maxLength >= numOfPixels)
    {
        *dstPtrPixel = (T) 255;
    }
    else
    {
        *dstPtrPixel = (T) 0;
    }

    free(bresenhamCircle);
    free(bresenhamCircleOutput);
}

template<typename T, typename U>
inline void fast_corner_detector_score_function_kernel_host(T* srcPtrWindow, U* dstPtrPixel, RppiSize srcSize,
                                                          Rpp32u* bresenhamCirclePositions, U centerPixel)
{
    U* bresenhamCircle = (U*) calloc(16, sizeof(U));
    U *bresenhamCircleTemp;
    bresenhamCircleTemp = bresenhamCircle;
    Rpp32u *bresenhamCirclePositionsTemp;
    bresenhamCirclePositionsTemp = bresenhamCirclePositions;

    for (int i = 0; i < 16; i++)
    {
        *bresenhamCircleTemp = (U) *(srcPtrWindow + *bresenhamCirclePositionsTemp);
        bresenhamCircleTemp++;
        bresenhamCirclePositionsTemp++;
    }

    U score = 0;
    bresenhamCircleTemp = bresenhamCircle;
    for (int i = 0; i < 16; i++)
    {
        score += RPPABS(centerPixel - *bresenhamCircleTemp);
        bresenhamCircleTemp++;
    }

    *dstPtrPixel = score;

    free(bresenhamCircle);
}

template<typename T, typename U, typename V>
inline void hog_single_channel_gradient_computations_kernel_host(T* srcPtr, RppiSize srcSize, U* gradientX, U* gradientY, U* gradientMagnitude, V* gradientDirection,
                                                               Rpp32f* gradientKernel, RppiSize rppiGradientKernelSizeX, RppiSize rppiGradientKernelSizeY)
{
    custom_convolve_image_host(srcPtr, srcSize, gradientX, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtr, srcSize, gradientY, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);
    compute_magnitude_host(gradientX, gradientY, srcSize, gradientMagnitude, RPPI_CHN_PLANAR, 1);
    compute_gradient_direction_host(gradientX, gradientY, srcSize, gradientDirection, RPPI_CHN_PLANAR, 1);
}

template<typename T, typename U, typename V>
inline void hog_three_channel_gradient_computations_kernel_host(T* srcPtr, T* srcPtrSingleChannel, RppiSize srcSize,
                                                              U* gradientX0, U* gradientY0, U* gradientX1, U* gradientY1, U* gradientX2, U* gradientY2,
                                                              U* gradientX, U* gradientY,
                                                              U* gradientMagnitude, V* gradientDirection,
                                                              Rpp32f* gradientKernel, RppiSize rppiGradientKernelSizeX, RppiSize rppiGradientKernelSizeY,
                                                              RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDim = srcSize.height * srcSize.width;

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 0, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX0, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY0, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 1, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX1, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY1, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_channel_extract_host(srcPtr, srcSize, srcPtrSingleChannel, 2, chnFormat, channel);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientX2, gradientKernel, rppiGradientKernelSizeX, RPPI_CHN_PLANAR, 1);
    custom_convolve_image_host(srcPtrSingleChannel, srcSize, gradientY2, gradientKernel, rppiGradientKernelSizeY, RPPI_CHN_PLANAR, 1);

    compute_max_host(gradientX0, gradientX1, srcSize, gradientX, channel);
    memcpy(gradientX0, gradientX, imageDim * sizeof(Rpp32s));
    compute_max_host(gradientX0, gradientX2, srcSize, gradientX, channel);

    compute_max_host(gradientY0, gradientY1, srcSize, gradientY, channel);
    memcpy(gradientY0, gradientY, imageDim * sizeof(Rpp32s));
    compute_max_host(gradientY0, gradientY2, srcSize, gradientY, channel);

    compute_magnitude_host(gradientX, gradientY, srcSize, gradientMagnitude, RPPI_CHN_PLANAR, 1);
    compute_gradient_direction_host(gradientX, gradientY, srcSize, gradientDirection, RPPI_CHN_PLANAR, 1);
}

// Convolution Functions
template<typename T>
inline void convolve_image_host_batch(T* srcPtrImage, RppiSize srcSize, RppiSize srcSizeMax, T* dstPtrImage,
                                           T* srcPtrBoundedROI, RppiSize srcSizeBoundedROI,
                                           Rpp32f* kernel, RppiSize kernelSize,
                                           Rpp32f x1, Rpp32f y1, Rpp32f x2, Rpp32f y2,
                                           RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDimMax = srcSizeMax.height * srcSizeMax.width;
    Rpp32u imageDimROI = srcSizeBoundedROI.height * srcSizeBoundedROI.width;

    T maxVal = (T)(std::numeric_limits<T>::max());
    T minVal = (T)(std::numeric_limits<T>::min());

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSizeBoundedROI.width - kernelSize.width;

        for(int c = 0; c < channel; c++)
        {
            T *srcPtrBoundedROIChannel, *srcPtrChannel, *dstPtrChannel;
            srcPtrBoundedROIChannel = srcPtrBoundedROI + (c * imageDimROI);
            srcPtrChannel = srcPtrImage + (c * imageDimMax);
            dstPtrChannel = dstPtrImage + (c * imageDimMax);

            Rpp32u roiRowCount = 0;

            for(int i = 0; i < srcSize.height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrChannel + (i * srcSizeMax.width);
                dstPtrTemp = dstPtrChannel + (i * srcSizeMax.width);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, srcSize.width * sizeof(T));

                    dstPtrTemp += srcSizeMax.width;
                    srcPtrTemp += srcSizeMax.width;
                }
                else
                {
                    T *srcPtrWindow;
                    srcPtrWindow = srcPtrBoundedROIChannel + (roiRowCount * srcSizeBoundedROI.width);
                    for(int j = 0; j < srcSize.width; j++)
                    {
                        if((x1 <= j) && (j <= x2 ))
                        {
                            convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                    kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                    chnFormat, channel);

                            srcPtrWindow++;
                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        else
                        {
                            *dstPtrTemp = *srcPtrTemp;

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                    #pragma omp critical
                    roiRowCount++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSizeBoundedROI.width - kernelSize.width) * channel;
        Rpp32u elementsInRowBoundedROI = channel * srcSizeBoundedROI.width;
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;
        Rpp32u elementsInRow = channel * srcSize.width;

        Rpp32u roiRowCount = 0;

        for(int i = 0; i < srcSize.height; i++)
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
                T *srcPtrWindow;
                srcPtrWindow = srcPtrBoundedROI + (roiRowCount * elementsInRowBoundedROI);
                for(int j = 0; j < srcSize.width; j++)
                {
                    if (!((x1 <= j) && (j <= x2 )))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, channel * sizeof(T));

                        dstPtrTemp += channel;
                        srcPtrTemp += channel;
                    }
                    else
                    {
                        for(int c = 0; c < channel; c++)
                        {

                            convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                    kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                    chnFormat, channel);

                            srcPtrWindow++;
                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                    }
                }
                #pragma omp critical
                roiRowCount++;
            }
        }
    }
}

template<typename T, typename U>
inline void convolve_image_host(T* srcPtrMod, RppiSize srcSizeMod, U* dstPtr, RppiSize srcSize,
                        Rpp32f* kernel, RppiSize kernelSize,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrWindow;
    U *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;

    U maxVal = (U)(std::numeric_limits<U>::max());
    U minVal = (U)(std::numeric_limits<U>::min());

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSizeMod.width - kernelSize.width;

        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize.width - 1);
            }
            srcPtrWindow += ((kernelSize.height - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSizeMod.width - kernelSize.width) * channel;

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize.width - 1) * channel);
        }
    }
}

template<typename T>
inline void convolve_subimage_host(T* srcPtrMod, RppiSize srcSizeMod, T* dstPtr, RppiSize srcSizeSubImage, RppiSize srcSize,
                        Rpp32f* kernel, RppiSize kernelSize,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    int widthDiffPlanar = srcSize.width - srcSizeSubImage.width;
    int widthDiffPacked = (srcSize.width - srcSizeSubImage.width) * channel;

    T *srcPtrWindow, *dstPtrTemp;

    T maxVal = (T)(std::numeric_limits<T>::max());
    T minVal = (T)(std::numeric_limits<T>::min());

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSize.width - kernelSize.width;

        for (int c = 0; c < channel; c++)
        {
            srcPtrWindow = srcPtrMod + (c * srcSize.height * srcSize.width);
            dstPtrTemp = dstPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
                for (int j = 0; j < srcSizeSubImage.width; j++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += widthDiffPlanar;
                dstPtrTemp += widthDiffPlanar;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = (srcSize.width - kernelSize.width) * channel;

        srcPtrWindow = srcPtrMod;
        dstPtrTemp = dstPtr;
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
            for (int j = 0; j < srcSizeSubImage.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize,
                                                 kernel, kernelSize, remainingElementsInRow, maxVal, minVal,
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += widthDiffPacked;
            dstPtrTemp += widthDiffPacked;
        }
    }
}

template <typename T, typename U>
inline RppStatus custom_convolve_image_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                                  Rpp32f *kernel, RppiSize rppiKernelSize,
                                  RppiChnFormat chnFormat, Rpp32u channel)
{
    if (rppiKernelSize.height % 2 == 0 || rppiKernelSize.width % 2 == 0)
    {
        return RPP_ERROR;
    }

    int boundY = ((rppiKernelSize.height - 1) / 2);
    int boundX = ((rppiKernelSize.width - 1) / 2);

    RppiSize srcSizeMod1, srcSizeMod2;

    srcSizeMod1.height = srcSize.height + boundY;
    srcSizeMod1.width = srcSize.width + boundX;
    T *srcPtrMod1 = (T *)calloc(srcSizeMod1.height * srcSizeMod1.width * channel, sizeof(T));
    generate_corner_padded_image_host(srcPtr, srcSize, srcPtrMod1, srcSizeMod1, 1, chnFormat, channel);

    srcSizeMod2.height = srcSizeMod1.height + boundY;
    srcSizeMod2.width = srcSizeMod1.width + boundX;
    T *srcPtrMod2 = (T *)calloc(srcSizeMod2.height * srcSizeMod2.width * channel, sizeof(T));
    generate_corner_padded_image_host(srcPtrMod1, srcSizeMod1, srcPtrMod2, srcSizeMod2, 4, chnFormat, channel);

    convolve_image_host(srcPtrMod2, srcSizeMod2, dstPtr, srcSize, kernel, rppiKernelSize, chnFormat, channel);

    free(srcPtrMod1);
    free(srcPtrMod2);

    return RPP_SUCCESS;
}

// Compute Functions for RPP Image API
template<typename T>
inline RppStatus compute_subimage_location_host(T* ptr, T** ptrSubImage,
                                         RppiSize size, RppiSize *sizeSubImage,
                                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    if ((RPPINRANGE(x1, 0, size.width - 1) == 0)
        || (RPPINRANGE(x2, 0, size.width - 1) == 0)
        || (RPPINRANGE(y1, 0, size.height - 1) == 0)
        || (RPPINRANGE(y2, 0, size.height - 1) == 0))
    {
        return RPP_ERROR;
    }

    int yDiff = (int) y2 - (int) y1;
    int xDiff = (int) x2 - (int) x1;

    sizeSubImage->height = (Rpp32u) RPPABS(yDiff) + 1;
    sizeSubImage->width = (Rpp32u) RPPABS(xDiff) + 1;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        *ptrSubImage = ptr + (RPPMIN2(y1, y2) * size.width) + RPPMIN2(x1, x2);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        *ptrSubImage = ptr + (RPPMIN2(y1, y2) * size.width * channel) + (RPPMIN2(x1, x2) * channel);
    }

    return RPP_SUCCESS;
}

template<typename T>
inline void compute_transpose_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < dstSize.height; i++)
            {
                for (int j = 0; j < dstSize.width; j++)
                {
                    *dstPtrTemp = *(srcPtrTemp + (j * srcSize.width) + i);
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < dstSize.height; i++)
        {
            for (int j = 0; j < dstSize.width; j++)
            {
                srcPtrTemp = srcPtr + (channel * ((j * srcSize.width) + i));
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *(srcPtrTemp + c);
                    dstPtrTemp++;
                }
            }
        }
    }
}

template <typename T, typename U>
inline void compute_add_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) + ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_subtract_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                        Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32s) (*srcPtr1Temp)) - ((Rpp32s) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_multiply_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtr1Temp)) * ((Rpp32f) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp = (T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_bitwise_AND_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                           Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) & ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_inclusive_OR_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                            Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) | ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_exclusive_OR_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                            Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp8u pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp8u) (*srcPtr1Temp)) ^ ((Rpp8u) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_min_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = RPPMIN2(*srcPtr1Temp, ((T)*srcPtr2Temp));
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_max_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = RPPMAX2(*srcPtr1Temp, ((T)*srcPtr2Temp));
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T, typename U>
inline void compute_rgb_to_hsv_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
    U *dstPtrTempH, *dstPtrTempS, *dstPtrTempV;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + (imageDim);
        srcPtrTempB = srcPtr + (2 * imageDim);
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + (imageDim);
        dstPtrTempV = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128i const zero = _mm_setzero_si128();
        __m128 pDiv = _mm_set1_ps(255.0);
        __m128 pMul = _mm_set1_ps(360.0);
        __m128i px0, px1, px2;
        __m128 xR, xG, xB;
        __m128 xH, xS, xV, xC;
        __m128 xX, xY, xZ;

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

            xR = _mm_div_ps(xR, pDiv);
            xG = _mm_div_ps(xG, pDiv);
            xB = _mm_div_ps(xB, pDiv);

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

            // Normalize H to fraction. If H >= 1 then H - 1
            xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

            xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
            xS = _mm_and_ps(xS, xC);
            xG = _mm_and_ps(xG, xC);
            xG = _mm_sub_ps(xG, xH);

            // Multiply by 360
            xG = _mm_mul_ps(xG, pMul);

            _mm_storeu_ps(dstPtrTempH, xG);
            _mm_storeu_ps(dstPtrTempS, xS);
            _mm_storeu_ps(dstPtrTempV, xV);

            srcPtrTempR += 4;
            srcPtrTempG += 4;
            srcPtrTempB += 4;
            dstPtrTempH += 4;
            dstPtrTempS += 4;
            dstPtrTempV += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                *dstPtrTempH = 0;
            }
            else if (cmax == rf)
            {
                *dstPtrTempH = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                *dstPtrTempH = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                *dstPtrTempH = round(60 * (((rf - gf) / delta) + 4));
            }

            while (*dstPtrTempH > 360)
            {
                *dstPtrTempH = *dstPtrTempH - 360;
            }
            while (*dstPtrTempH < 0)
            {
                *dstPtrTempH = 360 + *dstPtrTempH;
            }

            if (cmax == 0)
            {
                *dstPtrTempS = 0;
            }
            else
            {
                *dstPtrTempS = delta / cmax;
            }

            *dstPtrTempV = cmax;

            srcPtrTempR++;
            srcPtrTempG++;
            srcPtrTempB++;
            dstPtrTempH++;
            dstPtrTempS++;
            dstPtrTempV++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTempR = srcPtr;
        srcPtrTempG = srcPtr + 1;
        srcPtrTempB = srcPtr + 2;
        dstPtrTempH = dstPtr;
        dstPtrTempS = dstPtr + 1;
        dstPtrTempV = dstPtr + 2;

        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f rf, gf, bf, cmax, cmin, delta;
            rf = ((Rpp32f) *srcPtrTempR) / 255;
            gf = ((Rpp32f) *srcPtrTempG) / 255;
            bf = ((Rpp32f) *srcPtrTempB) / 255;
            cmax = RPPMAX3(rf, gf, bf);
            cmin = RPPMIN3(rf, gf, bf);
            delta = cmax - cmin;

            if (delta == 0)
            {
                *dstPtrTempH = 0;
            }
            else if (cmax == rf)
            {
                *dstPtrTempH = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                *dstPtrTempH = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                *dstPtrTempH = round(60 * (((rf - gf) / delta) + 4));
            }

            while (*dstPtrTempH > 360)
            {
                *dstPtrTempH = *dstPtrTempH - 360;
            }
            while (*dstPtrTempH < 0)
            {
                *dstPtrTempH = 360 + *dstPtrTempH;
            }

            if (cmax == 0)
            {
                *dstPtrTempS = 0;
            }
            else
            {
                *dstPtrTempS = delta / cmax;
            }

            *dstPtrTempV = cmax;

            srcPtrTempR += 3;
            srcPtrTempG += 3;
            srcPtrTempB += 3;
            dstPtrTempH += 3;
            dstPtrTempS += 3;
            dstPtrTempV += 3;
        }
    }
}

template <typename T, typename U>
inline void compute_hsv_to_rgb_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTempH, *srcPtrTempS, *srcPtrTempV;
    U *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp64u imageDim = srcSize.height * srcSize.width;
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + (imageDim);
        srcPtrTempV = srcPtr + (2 * imageDim);
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + (imageDim);
        dstPtrTempB = dstPtr + (2 * imageDim);

        Rpp64u bufferLength = srcSize.height * srcSize.width;
        Rpp64u alignedLength = bufferLength & ~3;

        __m128 pDiv = _mm_set1_ps(360.0);
        __m128 pMul = _mm_set1_ps(255.0);

        __m128 h0, h1, h2, h3;
        h0 = _mm_set1_ps(1.0);
        __m128 x0, x1, x2, x3;
        __m128 a0, a1;
        __m128i px1, px2, px3;

        Rpp8u arrayR[4];
        Rpp8u arrayG[4];
        Rpp8u arrayB[4];

        Rpp64u vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
        {
            // Load
            h1 =  _mm_loadu_ps((float*)srcPtrTempH);
            h2 =  _mm_loadu_ps((float*)srcPtrTempS);
            h3 =  _mm_loadu_ps((float*)srcPtrTempV);

            h1 = _mm_div_ps(h1, pDiv);

            _MM_TRANSPOSE4_PS (h0, h1, h2, h3);

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

            x1 = _mm_mul_ps(x1, pMul);
            x2 = _mm_mul_ps(x2, pMul);
            x3 = _mm_mul_ps(x3, pMul);

            px1 = _mm_cvtps_epi32(x1);
            px2 = _mm_cvtps_epi32(x2);
            px3 = _mm_cvtps_epi32(x3);

            px1 = _mm_packs_epi32(px1, px1);
            px1 = _mm_packus_epi16(px1, px1);
            *((int*)arrayR) = _mm_cvtsi128_si32(px1);

            px2 = _mm_packs_epi32(px2, px2);
            px2 = _mm_packus_epi16(px2, px2);
            *((int*)arrayG) = _mm_cvtsi128_si32(px2);

            px3 = _mm_packs_epi32(px3, px3);
            px3 = _mm_packus_epi16(px3, px3);
            *((int*)arrayB) = _mm_cvtsi128_si32(px3);

            memcpy(dstPtrTempR, arrayR, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempG, arrayG, 4 * sizeof(Rpp8u));
            memcpy(dstPtrTempB, arrayB, 4 * sizeof(Rpp8u));

            srcPtrTempH += 4;
            srcPtrTempS += 4;
            srcPtrTempV += 4;
            dstPtrTempR += 4;
            dstPtrTempG += 4;
            dstPtrTempB += 4;
        }
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = *srcPtrTempV * *srcPtrTempS;
            x = c * (1 - abs(int(fmod((*srcPtrTempH / 60), 2)) - 1));
            m = *srcPtrTempV - c;

            if ((0 <= *srcPtrTempH) && (*srcPtrTempH < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= *srcPtrTempH) && (*srcPtrTempH < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= *srcPtrTempH) && (*srcPtrTempH < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= *srcPtrTempH) && (*srcPtrTempH < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= *srcPtrTempH) && (*srcPtrTempH < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= *srcPtrTempH) && (*srcPtrTempH < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempH++;
            srcPtrTempS++;
            srcPtrTempV++;
            dstPtrTempR++;
            dstPtrTempG++;
            dstPtrTempB++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTempH = srcPtr;
        srcPtrTempS = srcPtr + 1;
        srcPtrTempV = srcPtr + 2;
        dstPtrTempR = dstPtr;
        dstPtrTempG = dstPtr + 1;
        dstPtrTempB = dstPtr + 2;

        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            Rpp32f c, x, m, rf, gf, bf;
            c = *srcPtrTempV * *srcPtrTempS;
            x = c * (1 - abs(int(fmod((*srcPtrTempH / 60), 2)) - 1));
            m = *srcPtrTempV - c;

            if ((0 <= *srcPtrTempH) && (*srcPtrTempH < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= *srcPtrTempH) && (*srcPtrTempH < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= *srcPtrTempH) && (*srcPtrTempH < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= *srcPtrTempH) && (*srcPtrTempH < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= *srcPtrTempH) && (*srcPtrTempH < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= *srcPtrTempH) && (*srcPtrTempH < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            *dstPtrTempR = (Rpp8u) round((rf + m) * 255);
            *dstPtrTempG = (Rpp8u) round((gf + m) * 255);
            *dstPtrTempB = (Rpp8u) round((bf + m) * 255);

            srcPtrTempH += 3;
            srcPtrTempS += 3;
            srcPtrTempV += 3;
            dstPtrTempR += 3;
            dstPtrTempG += 3;
            dstPtrTempB += 3;
        }
    }
}

template <typename T, typename U>
inline void compute_magnitude_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, U* dstPtr,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp, *srcPtr2Temp;
    U *dstPtrTemp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;
    Rpp32s srcPtr1Value, srcPtr2Value;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        srcPtr1Value = (Rpp32s) *srcPtr1Temp;
        srcPtr2Value = (Rpp32s) *srcPtr2Temp;
        pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(U) round(pixel);
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }
}

template <typename T>
inline void compute_magnitude_ROI_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                                            Rpp32f x1, Rpp32f y1, Rpp32f x2, Rpp32f y2,
                                            RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32u imageDim = srcSize.height * srcSize.width;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            T *srcPtr1Channel, *srcPtr2Channel, *dstPtrChannel;
            srcPtr1Channel = srcPtr1 + (c * imageDim);
            srcPtr2Channel = srcPtr2 + (c * imageDim);
            dstPtrChannel = dstPtr + (c * imageDim);

            for(int i = 0; i < srcSize.height; i++)
            {
                Rpp32f pixel;

                T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Channel + (i * srcSize.width);
                srcPtr2Temp = srcPtr2Channel + (i * srcSize.width);
                dstPtrTemp = dstPtrChannel + (i * srcSize.width);

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtr1Temp, srcSize.width * sizeof(T));

                    srcPtr1Temp += srcSize.width;
                    srcPtr2Temp += srcSize.width;
                    dstPtrTemp += srcSize.width;
                }
                else
                {
                    for(int j = 0; j < srcSize.width; j++)
                    {
                        if((x1 <= j) && (j <= x2 ))
                        {
                            Rpp32s srcPtr1Value = (Rpp32s) *srcPtr1Temp;
                            Rpp32s srcPtr2Value = (Rpp32s) *srcPtr2Temp;
                            pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
                            pixel = RPPPIXELCHECK(pixel);
                            *dstPtrTemp =(T) round(pixel);

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                        else
                        {
                            *dstPtrTemp = *srcPtr1Temp;

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;

        for(int i = 0; i < srcSize.height; i++)
        {
            Rpp32f pixel;

            T *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
            srcPtr1Temp = srcPtr1 + (i * elementsInRow);
            srcPtr2Temp = srcPtr2 + (i * elementsInRow);
            dstPtrTemp = dstPtr + (i * elementsInRow);

            if (!((y1 <= i) && (i <= y2)))
            {
                memcpy(dstPtrTemp, srcPtr1Temp, elementsInRow * sizeof(T));

                srcPtr1Temp += elementsInRow;
                srcPtr2Temp += elementsInRow;
                dstPtrTemp += elementsInRow;
            }
            else
            {
                for(int j = 0; j < srcSize.width; j++)
                {
                    if (!((x1 <= j) && (j <= x2 )))
                    {
                        memcpy(dstPtrTemp, srcPtr1Temp, channel * sizeof(T));

                        srcPtr1Temp += channel;
                        srcPtr2Temp += channel;
                        dstPtrTemp += channel;
                    }
                    else
                    {
                        for(int c = 0; c < channel; c++)
                        {
                            Rpp32s srcPtr1Value = (Rpp32s) *srcPtr1Temp;
                            Rpp32s srcPtr2Value = (Rpp32s) *srcPtr2Temp;
                            pixel = sqrt((srcPtr1Value * srcPtr1Value) + (srcPtr2Value * srcPtr2Value));
                            pixel = RPPPIXELCHECK(pixel);
                            *dstPtrTemp =(T) round(pixel);

                            srcPtr1Temp++;
                            srcPtr2Temp++;
                            dstPtrTemp++;
                        }
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
inline void compute_threshold_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                                 U min, U max, Rpp32u type,
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    U *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (type == 1)
    {
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            if (*srcPtrTemp < min)
            {
                *dstPtrTemp = (U) 0;
            }
            else if (*srcPtrTemp <= max)
            {
                *dstPtrTemp = (U) 255;
            }
            else
            {
                *dstPtrTemp = (U) 0;
            }

            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else if (type == 2)
    {
        for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
        {
            if (RPPABS(*srcPtrTemp) < min)
            {
                *dstPtrTemp = (U) 0;
            }
            else if (RPPABS(*srcPtrTemp) <= max)
            {
                *dstPtrTemp = (U) 255;
            }
            else
            {
                *dstPtrTemp = (U) 0;
            }

            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
}

template <typename T>
inline void compute_data_object_copy_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    memcpy(dstPtr, srcPtr, srcSize.height * srcSize.width * channel * sizeof(T));

}

template <typename T>
inline void compute_downsampled_image_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                                         RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp8u checkEven;
    checkEven = (Rpp8u) RPPISEVEN(srcSize.width);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);

            for (int i = 0; i < dstSize.height; i++)
            {
                for (int j = 0; j < dstSize.width; j++)
                {
                    *dstPtrTemp = *srcPtrTemp;
                    srcPtrTemp += 2;
                    dstPtrTemp++;
                }
                if (checkEven == 0)
                {
                    srcPtrTemp--;
                }
                srcPtrTemp += srcSize.width;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = srcSize.width * channel;

        for (int i = 0; i < dstSize.height; i++)
        {
            for (int j = 0; j < dstSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *srcPtrTemp;
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtrTemp += channel;
            }
            if (checkEven == 0)
            {
                srcPtrTemp -= channel;
            }
            srcPtrTemp += elementsInRow;
        }
    }
}

template <typename T>
inline RppStatus compute_channel_extract_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
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

template <typename T, typename U>
inline void compute_gradient_direction_host(T* gradientX, T* gradientY, RppiSize srcSize, U* gradientDirection,
                                          RppiChnFormat chnFormat, Rpp32u channel)
{
    T *gradientXTemp, *gradientYTemp;
    U *gradientDirectionTemp;
    gradientXTemp = gradientX;
    gradientYTemp = gradientY;
    gradientDirectionTemp = gradientDirection;

    Rpp32f pixel;

    for (int i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        if (*gradientXTemp != 0)
        {
            *gradientDirectionTemp = atan((Rpp32f) *gradientYTemp / (Rpp32f) *gradientXTemp);
        }
        else
        {
            if (*gradientYTemp > 0)
            {
                *gradientDirectionTemp = ((Rpp32f) PI) / 2.0;
            }
            else if (*gradientYTemp < 0)
            {
                *gradientDirectionTemp = ((Rpp32f) PI) / 2.0 * -1.0;
            }
            else if (*gradientYTemp == 0)
            {
                *gradientDirectionTemp = 0.0;
            }
        }
        gradientDirectionTemp++;
        gradientXTemp++;
        gradientYTemp++;
    }
}

inline Rpp32u fogGenerator(Rpp32u srcPtr, Rpp32f fogValue, int colour, int check)
{
    int fog = 0;
    int range = 3;
    if(check >= (240) && fogValue!=0);
    else if(check>=(170))
        range = 1;
    else if(check<=(85))
        range = 2;
    else
        range = 3;
    switch(range)
    {
        case 1:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + fogValue) - (fogValue*4) + (7*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + fogValue) + (7*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + fogValue) + (fogValue*4) + (7*fogValue);
            }
            break;
        case 2:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) - (fogValue*4) + (130*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) + (130*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + pow(fogValue,2)) + (fogValue*4) + 130*fogValue;
            }
            break;
        case 3:
            if(colour==1)
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) - (fogValue*4) + 20 + (100*fogValue);
            }
            else if(colour==2)
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) + 20 + (100*fogValue);
            }
            else
            {
                fog = srcPtr * (1.5 + pow(fogValue,1.5)) + (fogValue*4) + (100*fogValue);
            }
            break;
    }
    fog = RPPPIXELCHECK(fog);
    return fog;
}

inline void compute_image_location_host(RppiSize *batch_srcSizeMax, int batchCount, Rpp32u *loc, Rpp32u channel)
{
    for (int m = 0; m < batchCount; m++)
    {
        *loc += (batch_srcSizeMax[m].height * batch_srcSizeMax[m].width);
    }
    *loc *= channel;
}

template <typename T>
inline void compute_1_channel_minmax_host(T *srcPtr, RppiSize srcSize, RppiSize srcSizeMax,
                                                    T *min, T *max,
                                                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    srcPtrTemp = srcPtr;

    __m128i pMin, pMax;

    for (int i = 0; i < srcSize.height; i++)
    {
        pMin = _mm_set1_epi8(*min);
        pMax = _mm_set1_epi8(*max);

        int bufferLength = srcSize.width;
        int alignedLength = bufferLength & ~15;

        __m128i px0;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
        {
            px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
            pMin = _mm_min_epu8(px0, pMin);
            pMax = _mm_max_epu8(px0, pMax);
            srcPtrTemp +=16;
        }
        *min = (T) HorMin(pMin);
        *max = (T) HorMax(pMax);
        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
        {
            if (*srcPtrTemp < *min)
            {
                *min = *srcPtrTemp;
            }
            if (*srcPtrTemp > *max)
            {
                *max = *srcPtrTemp;
            }
            srcPtrTemp++;
        }
        srcPtrTemp += (srcSizeMax.width - srcSize.width);
    }
}

template <typename T>
inline void compute_3_channel_minmax_host(T *srcPtr, RppiSize srcSize, RppiSize srcSizeMax,
                                               T *min, T *max,
                                               RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp;
    srcPtrTemp = srcPtr;

    __m128i pMin, pMax;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            T minTemp, maxTemp;
            minTemp = *(min + c);
            maxTemp = *(max + c);
            for (int i = 0; i < srcSize.height; i++)
            {
                pMin = _mm_set1_epi8(minTemp);
                pMax = _mm_set1_epi8(maxTemp);

                int bufferLength = srcSize.width;
                int alignedLength = bufferLength & ~15;

                __m128i px0;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                    pMin = _mm_min_epu8(px0, pMin);
                    pMax = _mm_max_epu8(px0, pMax);
                    srcPtrTemp +=16;
                }
                minTemp = (T) HorMin(pMin);
                maxTemp = (T) HorMax(pMax);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if (*srcPtrTemp < minTemp)
                    {
                        minTemp = *srcPtrTemp;
                    }
                    if (*srcPtrTemp > maxTemp)
                    {
                        maxTemp = *srcPtrTemp;
                    }
                    srcPtrTemp++;
                }
                srcPtrTemp += (srcSizeMax.width - srcSize.width);
            }
            *(min + c) = minTemp;
            *(max + c) = maxTemp;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u elementsInRow = channel * srcSize.width;
        Rpp32u elementsInRowMax = channel * srcSizeMax.width;

        T minRTemp, maxRTemp, minGTemp, maxGTemp, minBTemp, maxBTemp;
        minRTemp = *min;
        maxRTemp = *max;
        minGTemp = *(min + 1);
        maxGTemp = *(max + 1);
        minBTemp = *(min + 2);
        maxBTemp = *(max + 2);

        pMin = _mm_set1_epi8(minRTemp);
        pMax = _mm_set1_epi8(maxRTemp);

        for (int i = 0; i < srcSize.height; i++)
        {
            int bufferLength = elementsInRow;
            int alignedLength = bufferLength & ~14;

            __m128i px0;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount+=15)
            {
                px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                pMin = _mm_min_epu8(px0, pMin);
                pMax = _mm_max_epu8(px0, pMax);
                srcPtrTemp +=15;
            }
            for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
            {
                if (*srcPtrTemp < minRTemp)
                {
                    minRTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxRTemp)
                {
                    maxRTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
                if (*srcPtrTemp < minGTemp)
                {
                    minGTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxGTemp)
                {
                    maxGTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
                if (*srcPtrTemp < minBTemp)
                {
                    minBTemp = *srcPtrTemp;
                }
                if (*srcPtrTemp > maxBTemp)
                {
                    maxBTemp = *srcPtrTemp;
                }
                srcPtrTemp++;
            }
            srcPtrTemp += (elementsInRowMax - elementsInRow);
        }

        T minVector[16], maxVector[16];
        _mm_storeu_si128((__m128i *)minVector, pMin);
        _mm_storeu_si128((__m128i *)maxVector, pMax);

        minRTemp = RPPMIN2(RPPMIN3(minVector[0], minVector[3], minVector[6]), RPPMIN3(minVector[9], minVector[12], minRTemp));
        minGTemp = RPPMIN2(RPPMIN3(minVector[1], minVector[4], minVector[7]), RPPMIN3(minVector[10], minVector[13], minGTemp));
        minBTemp = RPPMIN2(RPPMIN3(minVector[2], minVector[5], minVector[8]), RPPMIN3(minVector[11], minVector[14], minBTemp));

        maxRTemp = RPPMAX2(RPPMAX3(maxVector[0], maxVector[3], maxVector[6]), RPPMAX3(maxVector[9], maxVector[12], maxRTemp));
        maxGTemp = RPPMAX2(RPPMAX3(maxVector[1], maxVector[4], maxVector[7]), RPPMAX3(maxVector[10], maxVector[13], maxGTemp));
        maxBTemp = RPPMAX2(RPPMAX3(maxVector[2], maxVector[5], maxVector[8]), RPPMAX3(maxVector[11], maxVector[14], maxBTemp));

        *min = minRTemp;
        *max = maxRTemp;
        *(min + 1) = minGTemp;
        *(max + 1) = maxGTemp;
        *(min + 2) = minBTemp;
        *(max + 2) = maxBTemp;
    }
}

inline void compute_histogram_location_host(Rpp32u *batch_bins, int batchCount, Rpp32u *locHist)
{
    for (int m = 0; m < batchCount; m++)
    {
        *locHist += batch_bins[m];
    }
}

template <typename T>
inline void compute_unpadded_from_padded_host(T* srcPtrPadded, RppiSize srcSize, RppiSize srcSizeMax, T* dstPtrUnpadded,
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
}

template <typename T>
inline void compute_padded_from_unpadded_host(T* srcPtrUnpadded, RppiSize srcSize, RppiSize dstSizeMax, T* dstPtrPadded,
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
}

template <typename T>
inline void compute_planar_to_packed_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                        Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int c = 0; c < channel; c++)
    {
        dstPtrTemp += c;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                *dstPtrTemp = *srcPtrTemp;
                srcPtrTemp++;
                dstPtrTemp += 3;
            }
        }
        dstPtrTemp = dstPtr;
    }
}

template <typename T>
inline void compute_packed_to_planar_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                        Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int c = 0; c < channel; c++)
    {
        srcPtrTemp += c;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                *dstPtrTemp = *srcPtrTemp;
                dstPtrTemp++;
                srcPtrTemp += 3;
            }
        }
        srcPtrTemp = srcPtr;
    }
}

#endif // RPP_CPU_BATCHPD_HPP
