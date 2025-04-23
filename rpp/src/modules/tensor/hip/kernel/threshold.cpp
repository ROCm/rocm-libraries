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

#include "hip_tensor_executors.hpp"

__device__ void threshold_hip_rgb_compute(d_float24 *pix_f24, float3 *minRGB_f3, float3 *maxRGB_f3, float2 *rangeMinMax)
{
    bool channelCheck[3];
    for (int i = 0; i < 8; i++)
    {
        float pixelR, pixelG, pixelB;
        pixelR = pix_f24->f8[0].f1[i];
        pixelG = pix_f24->f8[1].f1[i];
        pixelB = pix_f24->f8[2].f1[i];
        channelCheck[0] = ((pixelR >= minRGB_f3->x) &&  (pixelR <= maxRGB_f3->x));
        channelCheck[1] = ((pixelG >= minRGB_f3->y) &&  (pixelG <= maxRGB_f3->y));
        channelCheck[2] = ((pixelB >= minRGB_f3->z) &&  (pixelB <= maxRGB_f3->z));
        float outVal = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? rangeMinMax->y : rangeMinMax->x;
        pix_f24->f8[0].f1[i] = outVal;
        pix_f24->f8[1].f1[i] = outVal;
        pix_f24->f8[2].f1[i] = outVal;
    }
}

__device__ void threshold_hip_greyscale_compute(d_float8 *pix_f8, float &minValue, float &maxValue, float2 *rangeMinMax)
{
    for (int i = 0; i < 8; i++)
    {
        float pixel = pix_f8->f1[i];
        pix_f8->f1[i] = (pixel >= minValue) &&  (pixel <= maxValue) ? rangeMinMax->y : rangeMinMax->x;
    }
}

template <typename T>
__global__ void threshold_pkd_tensor(T *srcPtr,
                                     uint2 srcStridesNH,
                                     T *dstPtr,
                                     uint2 dstStridesNH,
                                     float3 *minTensor,
                                     float3 *maxTensor,
                                     float2 rangeMinMax,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    float3 minRGB_f3 = minTensor[id_z];
    float3 maxRGB_f3 = maxTensor[id_z];

    d_float24 pix_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    threshold_hip_rgb_compute(&pix_f24, &minRGB_f3, &maxRGB_f3, &rangeMinMax);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void threshold_pln3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      float3 *minTensor,
                                      float3 *maxTensor,
                                      float2 rangeMinMax,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    float3 minRGB_f3 = minTensor[id_z];
    float3 maxRGB_f3 = maxTensor[id_z];

    d_float24 pix_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    threshold_hip_rgb_compute(&pix_f24, &minRGB_f3, &maxRGB_f3, &rangeMinMax);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void threshold_pln1_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      float *minTensor,
                                      float *maxTensor,
                                      float2 rangeMinMax,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    float minRGB = minTensor[id_z];
    float maxRGB = maxTensor[id_z];

    d_float8 pix_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    threshold_hip_greyscale_compute(&pix_f8, minRGB, maxRGB, &rangeMinMax);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
}

template <typename T>
__global__ void threshold_pkd3_pln3_tensor(T *srcPtr,
                                           uint2 srcStridesNH,
                                           T *dstPtr,
                                           uint3 dstStridesNCH,
                                           float3 *minTensor,
                                           float3 *maxTensor,
                                           float2 rangeMinMax,
                                           RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    float3 minRGB_f3 = minTensor[id_z];
    float3 maxRGB_f3 = maxTensor[id_z];

    d_float24 pix_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    threshold_hip_rgb_compute(&pix_f24, &minRGB_f3, &maxRGB_f3, &rangeMinMax);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void threshold_pln3_pkd3_tensor(T *srcPtr,
                                           uint3 srcStridesNCH,
                                           T *dstPtr,
                                           uint2 dstStridesNH,
                                           float3 *minTensor,
                                           float3 *maxTensor,
                                           float2 rangeMinMax,
                                           RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    float3 minRGB_f3 = minTensor[id_z];
    float3 maxRGB_f3 = maxTensor[id_z];

    d_float24 pix_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    threshold_hip_rgb_compute(&pix_f24, &minRGB_f3, &maxRGB_f3, &rangeMinMax);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_threshold_tensor(T *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    T *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *minTensor,
                                    Rpp32f *maxTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    Rpp32s globalThreads_x = (dstDescPtr->w + 7) >> 3;
    Rpp32s globalThreads_y = dstDescPtr->h;
    Rpp32s globalThreads_z = dstDescPtr->n;

    // set the rangeMin, rangeMax values based on the datatype
    Rpp32f rangeMin, rangeMax;
    if constexpr (std::is_same<T, Rpp8u>::value)
    {
        rangeMin = 0;
        rangeMax = 255;
    }
    else if constexpr (std::is_same<T, Rpp8s>::value)
    {
        rangeMin = -128;
        rangeMax = 127;
    }
    else if constexpr ((std::is_same<T, Rpp32f>::value) || (std::is_same<T, half>::value))
    {
        rangeMin = 0;
        rangeMax = 1;
    }

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(threshold_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           reinterpret_cast<float3 *>(minTensor),
                           reinterpret_cast<float3 *>(maxTensor),
                           make_float2(rangeMin, rangeMax),
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        if (srcDescPtr->c == 3)
        {
            hipLaunchKernelGGL(threshold_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               reinterpret_cast<float3 *>(minTensor),
                               reinterpret_cast<float3 *>(maxTensor),
                               make_float2(rangeMin, rangeMax),
                               roiTensorPtrSrc);
        }
        else
        {
            hipLaunchKernelGGL(threshold_pln1_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               minTensor,
                               maxTensor,
                               make_float2(rangeMin, rangeMax),
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(threshold_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               reinterpret_cast<float3 *>(minTensor),
                               reinterpret_cast<float3 *>(maxTensor),
                               make_float2(rangeMin, rangeMax),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(threshold_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               reinterpret_cast<float3 *>(minTensor),
                               reinterpret_cast<float3 *>(maxTensor),
                               make_float2(rangeMin, rangeMax),
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_threshold_tensor<Rpp8u>(Rpp8u*,
                                                    RpptDescPtr,
                                                    Rpp8u*,
                                                    RpptDescPtr,
                                                    Rpp32f*,
                                                    Rpp32f*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);

template RppStatus hip_exec_threshold_tensor<half>(half*,
                                                   RpptDescPtr,
                                                   half*,
                                                   RpptDescPtr,
                                                   Rpp32f*,
                                                   Rpp32f*,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus hip_exec_threshold_tensor<Rpp32f>(Rpp32f*,
                                                     RpptDescPtr,
                                                     Rpp32f*,
                                                     RpptDescPtr,
                                                     Rpp32f*,
                                                     Rpp32f*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_threshold_tensor<Rpp8s>(Rpp8s*,
                                                    RpptDescPtr,
                                                    Rpp8s*,
                                                    RpptDescPtr,
                                                    Rpp32f*,
                                                    Rpp32f*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);
