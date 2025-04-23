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

#ifndef RPPI_VALIDATE_OPERATIONS_FUNCTIONS
#define RPPI_VALIDATE_OPERATIONS_FUNCTIONS

#include <iostream>
#include <stdlib.h>

#include "rpp.h"
#include "rppdefs.h"
#include "handle.hpp"

#ifdef OCL_COMPILE
#include <CL/cl.h>
#elif defined (HIP_COMPILE)
#include <hip/hip_runtime_api.h>
#endif

inline RppLayoutParams get_layout_params(RpptLayout layout, Rpp32u channels)
{
    RppLayoutParams layoutParams;
    if(layout == RpptLayout::NCHW || layout == RpptLayout::NCDHW)
    {
        if (channels == 1) // PLN1
        {
            layoutParams.channelParam = 1;
            layoutParams.bufferMultiplier = 1;
        }
        else if (channels == 3) // PLN3
        {
            layoutParams.channelParam = 3;
            layoutParams.bufferMultiplier = 1;
        }
    }
    else if(layout == RpptLayout::NHWC || layout == RpptLayout::NDHWC)
    {
        //PKD
        layoutParams.channelParam = 1;
        layoutParams.bufferMultiplier = channels;
    }
    return layoutParams;
}

inline void copy_host_maxSrcSize(RppiSize maxSrcSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.maxSrcSize[i].height = maxSrcSize.height;
        handle.GetInitHandle()->mem.mcpu.maxSrcSize[i].width = maxSrcSize.width;
    }
}

inline void copy_host_maxDstSize(RppiSize maxDstSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.maxDstSize[i].height = maxDstSize.height;
        handle.GetInitHandle()->mem.mcpu.maxDstSize[i].width = maxDstSize.width;
    }
}

inline void copy_host_roi(RppiROI roiPoints, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].roiHeight = roiPoints.roiHeight;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].roiWidth = roiPoints.roiWidth;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].x = roiPoints.x;
        handle.GetInitHandle()->mem.mcpu.roiPoints[i].y = roiPoints.y;
    }
}

#ifdef GPU_SUPPORT

inline void copy_srcSize(RppiSize *srcSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
           handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] = srcSize[i].height;
           handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] = srcSize[i].width;
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.srcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.srcSize.width, handle.GetInitHandle()->mem.mgpu.csrcSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.srcSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.csrcSize.height, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.srcSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.csrcSize.width, 0, NULL, NULL);
#endif // backend
}

inline void copy_dstSize(RppiSize *dstSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
           handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] = dstSize[i].height;
           handle.GetInitHandle()->mem.mgpu.cdstSize.width[i] = dstSize[i].width;
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstSize.width, handle.GetInitHandle()->mem.mgpu.cdstSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cdstSize.height, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cdstSize.width, 0, NULL, NULL);
#endif // backend
}

inline void copy_roi(RppiROI roiPoints, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        if(roiPoints.roiHeight == 0 && roiPoints.roiWidth == 0)
        {
            handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
            handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        }
        else
        {
            handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight[i] = roiPoints.roiHeight + roiPoints.y;
            handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth[i] = roiPoints.roiWidth + roiPoints.x;
        }
        handle.GetInitHandle()->mem.mgpu.croiPoints.x[i] = roiPoints.x;
        handle.GetInitHandle()->mem.mgpu.croiPoints.y[i] = roiPoints.y;
    }

#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight, handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth, handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.x, handle.GetInitHandle()->mem.mgpu.croiPoints.x, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.roiPoints.y, handle.GetInitHandle()->mem.mgpu.croiPoints.y, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.roiHeight, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.roiWidth, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.x, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.x, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.roiPoints.y, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.croiPoints.y, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_float(float *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.floatArr[paramIndex].floatmem, handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem, sizeof(Rpp32f) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.floatArr[paramIndex].floatmem, CL_FALSE, 0, sizeof(Rpp32f) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.floatArr[paramIndex].floatmem, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_float3(float *param, rpp::Handle& handle, Rpp32u paramIndex)
{
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.float3Arr[paramIndex].floatmem, param, sizeof(Rpp32f) * handle.GetBatchSize() * 3, hipMemcpyHostToDevice);
#endif
}

inline void copy_param_uint(uint *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.uintArr[paramIndex].uintmem, handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.uintArr[paramIndex].uintmem, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.uintArr[paramIndex].uintmem, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_int(int *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.intArr[paramIndex].intmem, handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem, sizeof(Rpp32s) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.intArr[paramIndex].intmem, CL_FALSE, 0, sizeof(Rpp32s) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.intArr[paramIndex].intmem, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_uchar(Rpp8u *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.ucharArr[paramIndex].ucharmem, handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem, sizeof(Rpp8u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.ucharArr[paramIndex].ucharmem, CL_FALSE, 0, sizeof(Rpp8u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.ucharArr[paramIndex].ucharmem, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_char(char *param, rpp::Handle& handle, Rpp32u paramIndex)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.charArr[paramIndex].charmem, handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem, sizeof(Rpp8s) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.charArr[paramIndex].charmem, CL_FALSE, 0, sizeof(Rpp8s) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.charArr[paramIndex].charmem, 0, NULL, NULL);
#endif // backend
}

inline void copy_param_RpptRGB(RpptRGB *param, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize() ; i++)
    {
        handle.GetInitHandle()->mem.mcpu.rgbArr.rgbmem[i] = param[i];
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.rgbArr.rgbmem, handle.GetInitHandle()->mem.mcpu.rgbArr.rgbmem, sizeof(RpptRGB) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#endif // backend
}

inline void copy_srcMaxSize(RppiSize maxSrcSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] = maxSrcSize.height;
        handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i] = maxSrcSize.width;
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxSrcSize.height, handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxSrcSize.width, handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxSrcSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxSrcSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width, 0, NULL, NULL);
#endif // backend
}

inline void copy_dstMaxSize(RppiSize maxDstSize, rpp::Handle& handle)
{
    for(int i = 0; i < handle.GetBatchSize(); i++)
    {
        handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height[i] = maxDstSize.height;
        handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width[i] = maxDstSize.width;
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxDstSize.height, handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.maxDstSize.width, handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxDstSize.height, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.maxDstSize.width, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width, 0, NULL, NULL);
#endif // backend
}

inline void get_srcBatchIndex(rpp::Handle& handle, unsigned int channel, RppiChnFormat chnFormat, bool is_padded = true)
{
    int i;
    handle.GetInitHandle()->mem.mcpu.srcBatchIndex[0] = 0;
    for(i = 0; i < handle.GetBatchSize() - 1; i++)
    {
        handle.GetInitHandle()->mem.mcpu.srcBatchIndex[i+1] = handle.GetInitHandle()->mem.mcpu.srcBatchIndex[i] + handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i] * channel;
    }
    for(i = 0; i < handle.GetBatchSize(); i++)
    {
        if(chnFormat != RPPI_CHN_PLANAR)
        {
            handle.GetInitHandle()->mem.mcpu.inc[i] = 1;
        }
        else
        {
            if(!is_padded)
                handle.GetInitHandle()->mem.mcpu.inc[i] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
            else
                handle.GetInitHandle()->mem.mcpu.inc[i] = handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxSrcSize.width[i];
        }
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.srcBatchIndex, handle.GetInitHandle()->mem.mcpu.srcBatchIndex, sizeof(Rpp64u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.inc, handle.GetInitHandle()->mem.mcpu.inc, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.srcBatchIndex, CL_FALSE, 0, sizeof(Rpp64u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.srcBatchIndex, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.inc, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.inc, 0, NULL, NULL);
#endif // backend
}

inline void get_dstBatchIndex(rpp::Handle& handle, unsigned int channel, RppiChnFormat chnFormat, bool is_padded = true)
{
    int i;
    handle.GetInitHandle()->mem.mcpu.dstBatchIndex[0] = 0;
    for(i = 0; i < handle.GetBatchSize() - 1; i++)
    {
       handle.GetInitHandle()->mem.mcpu.dstBatchIndex[i+1] = handle.GetInitHandle()->mem.mcpu.dstBatchIndex[i] + handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width[i] * channel;
    }
    for(i = 0; i < handle.GetBatchSize(); i++)
    {
        if(chnFormat != RPPI_CHN_PLANAR)
        {
            handle.GetInitHandle()->mem.mcpu.dstInc[i] = 1;
        }
        else
        {
            if(!is_padded)
                handle.GetInitHandle()->mem.mcpu.dstInc[i] = handle.GetInitHandle()->mem.mgpu.cdstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cdstSize.width[i];
            else
                handle.GetInitHandle()->mem.mcpu.dstInc[i] = handle.GetInitHandle()->mem.mgpu.cmaxDstSize.height[i] * handle.GetInitHandle()->mem.mgpu.cmaxDstSize.width[i];
        }
    }
#ifdef HIP_COMPILE
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstBatchIndex, handle.GetInitHandle()->mem.mcpu.dstBatchIndex, sizeof(Rpp64u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
    hipMemcpy(handle.GetInitHandle()->mem.mgpu.dstInc, handle.GetInitHandle()->mem.mcpu.dstInc, sizeof(Rpp32u) * handle.GetBatchSize(), hipMemcpyHostToDevice);
#elif defined(OCL_COMPILE)
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstBatchIndex, CL_FALSE, 0, sizeof(Rpp64u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.dstBatchIndex, 0, NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), handle.GetInitHandle()->mem.mgpu.dstInc, CL_FALSE, 0, sizeof(Rpp32u) * handle.GetBatchSize(), handle.GetInitHandle()->mem.mcpu.dstInc, 0, NULL, NULL);
#endif // backend
}

#endif // GPU_SUPPORT

inline int check_roi_out_of_bounds(RpptROIPtr roiPtrImage, RpptDescPtr srcDescPtr, RpptRoiType type)
{
    int x, y, w, h;
    if (type == RpptRoiType::XYWH)
    {
        x = ((0 <= roiPtrImage->xywhROI.xy.x) && (roiPtrImage->xywhROI.xy.x < srcDescPtr->w)) ? roiPtrImage->xywhROI.xy.x : -1;
        y = ((0 <= roiPtrImage->xywhROI.xy.y) && (roiPtrImage->xywhROI.xy.y < srcDescPtr->h)) ? roiPtrImage->xywhROI.xy.y : -1;
        w = ((roiPtrImage->xywhROI.roiWidth) <= srcDescPtr->w) ? roiPtrImage->xywhROI.roiWidth : -1;
        h = ((roiPtrImage->xywhROI.roiHeight) <= srcDescPtr->h) ? roiPtrImage->xywhROI.roiHeight : -1;
    }
    else if (type == RpptRoiType::LTRB)
    {
        x = ((0 <= roiPtrImage->ltrbROI.lt.x) && (roiPtrImage->ltrbROI.lt.x < srcDescPtr->w)) ? roiPtrImage->ltrbROI.lt.x : -1;
        y = ((0 <= roiPtrImage->ltrbROI.lt.y) && (roiPtrImage->ltrbROI.lt.y < srcDescPtr->h)) ? roiPtrImage->ltrbROI.lt.y : -1;
        w = ((0 <= roiPtrImage->ltrbROI.rb.x) && (roiPtrImage->ltrbROI.rb.x < srcDescPtr->w)) ? roiPtrImage->ltrbROI.rb.x - roiPtrImage->ltrbROI.lt.x + 1 : -1;
        h = ((0 <= roiPtrImage->ltrbROI.rb.y) && (roiPtrImage->ltrbROI.rb.y < srcDescPtr->h)) ? roiPtrImage->ltrbROI.rb.y - roiPtrImage->ltrbROI.lt.y + 1 : -1;
    }
    if ((x < 0) || (y < 0) || (w < 0) || (h < 0))
        return -1;
    return 0;
}

#endif // RPPI_VALIDATE_OPERATIONS_FUNCTIONS
