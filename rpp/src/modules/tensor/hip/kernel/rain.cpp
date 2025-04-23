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
#include <random>

// Constants to represent the rain intensity for different data types
#define RAIN_INTENSITY_8U 200   // Intensity value for Rpp8u
#define RAIN_INTENSITY_8S 72    // Intensity value for Rpp8s
#define RAIN_INTENSITY_FLOAT 200 * ONE_OVER_255 // Intensity value for Rpp32f and Rpp16f

__device__ __forceinline__ void rain_hip_compute(uchar *srcPtr, d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, float4 *alpha_f4)
{
    dst_f8->f4[0] = rpp_hip_pixel_check_0to255((src2_f8->f4[0] - src1_f8->f4[0]) * *alpha_f4 + src1_f8->f4[0]);
    dst_f8->f4[1] = rpp_hip_pixel_check_0to255((src2_f8->f4[1] - src1_f8->f4[1]) * *alpha_f4 + src1_f8->f4[1]);
}

__device__ __forceinline__ void rain_hip_compute(float *srcPtr, d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, float4 *alpha_f4)
{
    dst_f8->f4[0] = rpp_hip_pixel_check_0to1((src2_f8->f4[0] - src1_f8->f4[0]) * *alpha_f4 + src1_f8->f4[0]);
    dst_f8->f4[1] = rpp_hip_pixel_check_0to1((src2_f8->f4[1] - src1_f8->f4[1]) * *alpha_f4 + src1_f8->f4[1]);
}

__device__ __forceinline__ void rain_hip_compute(schar *srcPtr, d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, float4 *alpha_f4)
{
    dst_f8->f4[0] = rpp_hip_pixel_check_0to255((src2_f8->f4[0] - src1_f8->f4[0]) * *alpha_f4 + src1_f8->f4[0] + (float4)128) - (float4)128;
    dst_f8->f4[1] = rpp_hip_pixel_check_0to255((src2_f8->f4[1] - src1_f8->f4[1]) * *alpha_f4 + src1_f8->f4[1] + (float4)128) - (float4)128;
}

__device__ __forceinline__ void rain_hip_compute(half *srcPtr, d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, float4 *alpha_f4)
{
    dst_f8->f4[0] = rpp_hip_pixel_check_0to1((src2_f8->f4[0] - src1_f8->f4[0]) * *alpha_f4 + src1_f8->f4[0]);
    dst_f8->f4[1] = rpp_hip_pixel_check_0to1((src2_f8->f4[1] - src1_f8->f4[1]) * *alpha_f4 + src1_f8->f4[1]);
}

template <typename T>
__global__ void rain_pkd_hip_tensor(T *srcPtr1,
                                    T *srcPtr2,
                                    uint3 srcStridesNHW,
                                    T *dstPtr,
                                    uint2 dstStridesNH,
                                    float *alpha,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
    {
        return;
    }

    uint srcIdx1 = (id_z * srcStridesNHW.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNHW.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint srcIdx2 = ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNHW.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float4 alpha_f4 = static_cast<float4>(alpha[id_z]);
    d_float24 src1_f24, dst_f24;
    d_float8 src2_f8;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, &src1_f24);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src2_f8);
    rain_hip_compute(srcPtr1, &src1_f24.f8[0], &src2_f8, &dst_f24.f8[0], &alpha_f4);
    rain_hip_compute(srcPtr1, &src1_f24.f8[1], &src2_f8, &dst_f24.f8[1], &alpha_f4);
    rain_hip_compute(srcPtr1, &src1_f24.f8[2], &src2_f8, &dst_f24.f8[2], &alpha_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void rain_pln_hip_tensor(T *srcPtr1,
                                    T *srcPtr2,
                                    uint3 srcStridesNCH,
                                    T *dstPtr,
                                    uint3 dstStridesNCH,
                                    int channelsDst,
                                    float *alpha,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx1 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint srcIdx2 = ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float4 alpha_f4 = static_cast<float4>(alpha[id_z]);
    d_float8 src1_f8, src2_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src1_f8);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src2_f8);
    rain_hip_compute(srcPtr1, &src1_f8, &src2_f8, &dst_f8, &alpha_f4);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx1 += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src1_f8);
        rain_hip_compute(srcPtr1, &src1_f8, &src2_f8, &dst_f8, &alpha_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
        srcIdx1 += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src1_f8);
        rain_hip_compute(srcPtr1, &src1_f8, &src2_f8, &dst_f8, &alpha_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void rain_pkd3_pln3_hip_tensor(T *srcPtr1,
                                          T *srcPtr2,
                                          uint3 srcStridesNHW,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          float *alpha,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx1 = (id_z * srcStridesNHW.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNHW.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint srcIdx2 = ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNHW.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float4 alpha_f4 = static_cast<float4>(alpha[id_z]);
    d_float24 src1_f24, dst_f24;
    d_float8 src2_f8;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, &src1_f24);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src2_f8);
    rain_hip_compute(srcPtr1, &src1_f24.f8[0], &src2_f8, &dst_f24.f8[0], &alpha_f4);
    rain_hip_compute(srcPtr1, &src1_f24.f8[1], &src2_f8, &dst_f24.f8[1], &alpha_f4);
    rain_hip_compute(srcPtr1, &src1_f24.f8[2], &src2_f8, &dst_f24.f8[2], &alpha_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void rain_pln3_pkd3_hip_tensor(T *srcPtr1,
                                          T *srcPtr2,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float *alpha,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx1 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint srcIdx2 = ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float4 alpha_f4 = static_cast<float4>(alpha[id_z]);
    d_float24 src1_f24, dst_f24;
    d_float8 src2_f8;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx1, srcStridesNCH.y, &src1_f24);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src2_f8);
    rain_hip_compute(srcPtr1, &src1_f24.f8[0], &src2_f8, &dst_f24.f8[0], &alpha_f4);
    rain_hip_compute(srcPtr1, &src1_f24.f8[1], &src2_f8, &dst_f24.f8[1], &alpha_f4);
    rain_hip_compute(srcPtr1, &src1_f24.f8[2], &src2_f8, &dst_f24.f8[2], &alpha_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_rain_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f rainPercentage,
                               Rpp32u rainWidth,
                               Rpp32u rainHeight,
                               Rpp32f slantAngle,
                               Rpp32f *alpha,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    Rpp32f rainPercent = rainPercentage * 0.004f; //Scaling factor to convert percentage to a range suitable for rain effect intensity
    Rpp32u numDrops = static_cast<Rpp32u>(rainPercent * srcDescPtr->h * srcDescPtr->w);
    Rpp32f slant = sin(slantAngle) * rainHeight;

    // Seed the random number generator and set up the uniform distributions
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<Rpp32u> distX(0, srcDescPtr->w - slant - 1);
    std::uniform_int_distribution<Rpp32u> distY(0, srcDescPtr->h - rainHeight - 1);

    T *rainLayer = reinterpret_cast<T *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    T initValue = 0;
    if constexpr (std::is_same<T, Rpp8s>::value)
        initValue = static_cast<T>(0x81);   // 0x81 represents -127 in signed 8-bit integer(Rpp8s).
    std::memset(rainLayer, initValue, srcDescPtr->w * srcDescPtr->h * sizeof(T));
    // Choose the rain intensity value based on the data type
    T rainValue = std::is_same<T, Rpp8u>::value ? static_cast<T>(RAIN_INTENSITY_8U) :
                  std::is_same<T, Rpp8s>::value ? static_cast<T>(RAIN_INTENSITY_8S) :
                  static_cast<T>(RAIN_INTENSITY_FLOAT);
    Rpp32f slantPerDropLength = static_cast<Rpp32f>(slant) / rainHeight;
    for (Rpp32u i = 0; i < numDrops; i++)
    {
        Rpp32u xStart = distX(rng);
        Rpp32u yStart = distX(rng);
        for (Rpp32u j = 0; j < rainHeight; j++)
        {
            Rpp32u x = xStart + j * slantPerDropLength;
            Rpp32u y = yStart + j;

            if (x >= 0 && x < srcDescPtr->w && y < srcDescPtr->h)
            {
                T *rainLayerTemp = rainLayer + y * srcDescPtr->w + x;
                for (Rpp32u k = 0; k < rainWidth; k++)
                    rainLayerTemp[k] = rainValue;
            }
        }
    }

    T *rainLayerHip = reinterpret_cast<T *>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
    CHECK_RETURN_STATUS(hipMemcpyAsync(rainLayerHip, rainLayer, srcDescPtr->w * srcDescPtr->h * sizeof(T), hipMemcpyHostToDevice, handle.GetStream()));

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(rain_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           rainLayerHip,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride, srcDescPtr->w),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           alpha,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(rain_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           rainLayerHip,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           alpha,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(rain_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               rainLayerHip,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride, srcDescPtr->w),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               alpha,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(rain_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               rainLayerHip,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               alpha,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_rain_tensor<Rpp8u>(Rpp8u*,
                                               RpptDescPtr,
                                               Rpp8u*,
                                               RpptDescPtr,
                                               Rpp32f,
                                               Rpp32u,
                                               Rpp32u,
                                               Rpp32f,
                                               Rpp32f*,
                                               RpptROIPtr,
                                               RpptRoiType,
                                               rpp::Handle&);

template RppStatus hip_exec_rain_tensor<half>(half*,
                                              RpptDescPtr,
                                              half*,
                                              RpptDescPtr,
                                              Rpp32f,
                                              Rpp32u,
                                              Rpp32u,
                                              Rpp32f,
                                              Rpp32f*,
                                              RpptROIPtr,
                                              RpptRoiType,
                                              rpp::Handle&);

template RppStatus hip_exec_rain_tensor<Rpp32f>(Rpp32f*,
                                                RpptDescPtr,
                                                Rpp32f*,
                                                RpptDescPtr,
                                                Rpp32f,
                                                Rpp32u,
                                                Rpp32u,
                                                Rpp32f,
                                                Rpp32f*,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_rain_tensor<Rpp8s>(Rpp8s*,
                                               RpptDescPtr,
                                               Rpp8s*,
                                               RpptDescPtr,
                                               Rpp32f,
                                               Rpp32u,
                                               Rpp32u,
                                               Rpp32f,
                                               Rpp32f*,
                                               RpptROIPtr,
                                               RpptRoiType,
                                               rpp::Handle&);
