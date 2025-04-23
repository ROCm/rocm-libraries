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

__device__ __forceinline__ void fog_grey_hip_compute(d_float8 *r_f8, d_float8 *g_f8, d_float8 *b_f8, float4 *greyFactor_f4)
{
    float4 grey_f4[2];
    float4 rMultiplier_f4 = static_cast<float4>(RGB_TO_GREY_WEIGHT_RED);
    float4 gMultiplier_f4 = static_cast<float4>(RGB_TO_GREY_WEIGHT_GREEN);
    float4 bMultiplier_f4 = static_cast<float4>(RGB_TO_GREY_WEIGHT_BLUE);
    grey_f4[0] = r_f8->f4[0] * rMultiplier_f4 + g_f8->f4[0] * gMultiplier_f4 + b_f8->f4[0] * bMultiplier_f4;
    grey_f4[1] = r_f8->f4[1] * rMultiplier_f4 + g_f8->f4[1] * gMultiplier_f4 + b_f8->f4[1] * bMultiplier_f4;
    float4 oneMinusGreyFactor_f4 = static_cast<float4>(1.0f) - *greyFactor_f4;
    grey_f4[0] = grey_f4[0] * *greyFactor_f4;
    grey_f4[1] = grey_f4[1] * *greyFactor_f4;
    r_f8->f4[0] = (r_f8->f4[0] * oneMinusGreyFactor_f4) + grey_f4[0];
    g_f8->f4[0] = (g_f8->f4[0] * oneMinusGreyFactor_f4) + grey_f4[0];
    b_f8->f4[0] = (b_f8->f4[0] * oneMinusGreyFactor_f4) + grey_f4[0];
    r_f8->f4[1] = (r_f8->f4[1] * oneMinusGreyFactor_f4) + grey_f4[1];
    g_f8->f4[1] = (g_f8->f4[1] * oneMinusGreyFactor_f4) + grey_f4[1];
    b_f8->f4[1] = (b_f8->f4[1] * oneMinusGreyFactor_f4) + grey_f4[1];
}

__device__ __forceinline__ void fog_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *maskAlpha_f8, d_float8 *maskIntensity_f8, float4 *intensityFactor_f4)
{
    float4 alphaFactor_f4[2];
    alphaFactor_f4[0] = rpp_hip_pixel_check_0to1(maskAlpha_f8->f4[0] + *intensityFactor_f4);
    alphaFactor_f4[1] = rpp_hip_pixel_check_0to1(maskAlpha_f8->f4[1] + *intensityFactor_f4);
    dst_f8->f4[0] = rpp_hip_pixel_check_0to255((src_f8->f4[0] * (static_cast<float4>(1) - alphaFactor_f4[0])) + (maskIntensity_f8->f4[0] * alphaFactor_f4[0]));
    dst_f8->f4[1] = rpp_hip_pixel_check_0to255((src_f8->f4[1] * (static_cast<float4>(1) - alphaFactor_f4[1])) + (maskIntensity_f8->f4[1] * alphaFactor_f4[1]));
}

__device__ __forceinline__ void fog_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *maskAlpha_f8, d_float8 *maskIntensity_f8, float4 *intensityFactor_f4)
{
    float4 pixNorm_f4[2], alphaFactor_f4[2];
    pixNorm_f4[0] = maskIntensity_f8->f4[0] * static_cast<float4>(ONE_OVER_255);
    pixNorm_f4[1] = maskIntensity_f8->f4[1] * static_cast<float4>(ONE_OVER_255);
    alphaFactor_f4[0] = rpp_hip_pixel_check_0to1(maskAlpha_f8->f4[0] + *intensityFactor_f4);
    alphaFactor_f4[1] = rpp_hip_pixel_check_0to1(maskAlpha_f8->f4[1] + *intensityFactor_f4);
    dst_f8->f4[0] = (src_f8->f4[0] * (static_cast<float4>(1) - alphaFactor_f4[0])) + (pixNorm_f4[0] * alphaFactor_f4[0]);
    dst_f8->f4[1] = (src_f8->f4[1] * (static_cast<float4>(1) - alphaFactor_f4[1])) + (pixNorm_f4[1] * alphaFactor_f4[1]);
}

__device__ __forceinline__ void fog_hip_compute(schar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *maskAlpha_f8, d_float8 *maskIntensity_f8, float4 *intensityFactor_f4)
{
    float4 alphaFactor_f4[2];
    alphaFactor_f4[0] = rpp_hip_pixel_check_0to1(maskAlpha_f8->f4[0] + *intensityFactor_f4);
    alphaFactor_f4[1] = rpp_hip_pixel_check_0to1(maskAlpha_f8->f4[1] + *intensityFactor_f4);
    dst_f8->f4[0] = rpp_hip_pixel_check_0to255((src_f8->f4[0] + static_cast<float4>(128)) * (static_cast<float4>(1) - alphaFactor_f4[0]) + (maskIntensity_f8->f4[0] * alphaFactor_f4[0])) - static_cast<float4>(128);
    dst_f8->f4[1] = rpp_hip_pixel_check_0to255((src_f8->f4[1] + static_cast<float4>(128)) * (static_cast<float4>(1) - alphaFactor_f4[1]) + (maskIntensity_f8->f4[1] * alphaFactor_f4[1])) - static_cast<float4>(128);
}

__device__ __forceinline__ void fog_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *maskAlpha_f8, d_float8 *maskIntensity_f8, float4 *intensityFactor_f4)
{
    float4 pixNorm_f4[2], alphaFactor_f4[2];
    pixNorm_f4[0] = maskIntensity_f8->f4[0] * static_cast<float4>(ONE_OVER_255);
    pixNorm_f4[1] = maskIntensity_f8->f4[1] * static_cast<float4>(ONE_OVER_255);
    alphaFactor_f4[0] = rpp_hip_pixel_check_0to1(maskAlpha_f8->f4[0] + *intensityFactor_f4);
    alphaFactor_f4[1] = rpp_hip_pixel_check_0to1(maskAlpha_f8->f4[1] + *intensityFactor_f4);
    dst_f8->f4[0] = (src_f8->f4[0] * (static_cast<float4>(1) - alphaFactor_f4[0])) + (pixNorm_f4[0] * alphaFactor_f4[0]);
    dst_f8->f4[1] = (src_f8->f4[1] * (static_cast<float4>(1) - alphaFactor_f4[1])) + (pixNorm_f4[1] * alphaFactor_f4[1]);
}

template <typename T>
__global__ void fog_pkd_hip_tensor(T *srcPtr,
                                   uint2 srcStridesNH,
                                   T *dstPtr,
                                   uint2 dstStridesNH,
                                   float *fogAlphaMaskPtr,
                                   float *fogIntensityMaskPtr,
                                   float *intensityFactor,
                                   float *greyFactor,
                                   uint *maskLocOffsetX,
                                   uint *maskLocOffsetY,
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
    uint maskIdx = ((id_y + maskLocOffsetY[id_z]) * (srcStridesNH.y / 3)) + (id_x + maskLocOffsetX[id_z]);

    d_float24 src_f24, dst_f24;
    d_float8 maskAlpha_f8, maskIntensity_f8;
    float4 intensityFactor_f4 = static_cast<float4>(intensityFactor[id_z]);
    float4 greyFactor_f4 = static_cast<float4>(greyFactor[id_z]);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    rpp_hip_load8_and_unpack_to_float8(fogAlphaMaskPtr + maskIdx, &maskAlpha_f8);
    rpp_hip_load8_and_unpack_to_float8(fogIntensityMaskPtr + maskIdx, &maskIntensity_f8);
    fog_grey_hip_compute(&src_f24.f8[0], &src_f24.f8[1], &src_f24.f8[2], &greyFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void fog_pln_hip_tensor(T *srcPtr,
                                   uint3 srcStridesNCH,
                                   T *dstPtr,
                                   uint3 dstStridesNCH,
                                   int channelsDst,
                                   float *fogAlphaMaskPtr,
                                   float *fogIntensityMaskPtr,
                                   float *intensityFactor,
                                   float *greyFactor,
                                   uint *maskLocOffsetX,
                                   uint *maskLocOffsetY,
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
    uint maskIdx = ((id_y + maskLocOffsetY[id_z]) * srcStridesNCH.z) + (id_x + maskLocOffsetX[id_z]);

    d_float8 maskAlpha_f8, maskIntensity_f8;
    float4 intensityFactor_f4 = static_cast<float4>(intensityFactor[id_z]);
    float4 greyFactor_f4 = static_cast<float4>(greyFactor[id_z]);
    rpp_hip_load8_and_unpack_to_float8(fogAlphaMaskPtr + maskIdx, &maskAlpha_f8);
    rpp_hip_load8_and_unpack_to_float8(fogIntensityMaskPtr + maskIdx, &maskIntensity_f8);

    if (channelsDst == 3)
    {
        d_float24 src_f24, dst_f24;
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);

        fog_grey_hip_compute(&src_f24.f8[0], &src_f24.f8[1], &src_f24.f8[2], &greyFactor_f4);
        fog_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
        fog_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
        fog_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);

        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
    }
    else
    {
        d_float8 src_f8, dst_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        fog_hip_compute(srcPtr, &src_f8, &dst_f8, &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void fog_pkd3_pln3_hip_tensor(T *srcPtr,
                                         uint2 srcStridesNH,
                                         T *dstPtr,
                                         uint3 dstStridesNCH,
                                         float *fogAlphaMaskPtr,
                                         float *fogIntensityMaskPtr,
                                         float *intensityFactor,
                                         float *greyFactor,
                                         uint *maskLocOffsetX,
                                         uint *maskLocOffsetY,
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
    uint maskIdx = ((id_y + maskLocOffsetY[id_z]) * (srcStridesNH.y / 3)) + (id_x + maskLocOffsetX[id_z]);

    d_float24 src_f24, dst_f24;
    d_float8 maskAlpha_f8, maskIntensity_f8;
    float4 intensityFactor_f4 = static_cast<float4>(intensityFactor[id_z]);
    float4 greyFactor_f4 = static_cast<float4>(greyFactor[id_z]);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    rpp_hip_load8_and_unpack_to_float8(fogAlphaMaskPtr + maskIdx, &maskAlpha_f8);
    rpp_hip_load8_and_unpack_to_float8(fogIntensityMaskPtr + maskIdx, &maskIntensity_f8);
    fog_grey_hip_compute(&src_f24.f8[0], &src_f24.f8[1], &src_f24.f8[2], &greyFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void fog_pln3_pkd3_hip_tensor(T *srcPtr,
                                         uint3 srcStridesNCH,
                                         T *dstPtr,
                                         uint2 dstStridesNH,
                                         float *fogAlphaMaskPtr,
                                         float *fogIntensityMaskPtr,
                                         float *intensityFactor,
                                         float *greyFactor,
                                         uint *maskLocOffsetX,
                                         uint *maskLocOffsetY,
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
    uint maskIdx = ((id_y + maskLocOffsetY[id_z]) * srcStridesNCH.z) + (id_x + maskLocOffsetX[id_z]);

    d_float24 src_f24, dst_f24;
    d_float8 maskAlpha_f8, maskIntensity_f8;
    float4 intensityFactor_f4 = static_cast<float4>(intensityFactor[id_z]);
    float4 greyFactor_f4 = static_cast<float4>(greyFactor[id_z]);
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    rpp_hip_load8_and_unpack_to_float8(fogAlphaMaskPtr + maskIdx, &maskAlpha_f8);
    rpp_hip_load8_and_unpack_to_float8(fogIntensityMaskPtr + maskIdx, &maskIntensity_f8);
    fog_grey_hip_compute(&src_f24.f8[0], &src_f24.f8[1], &src_f24.f8[2], &greyFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    fog_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &maskAlpha_f8, &maskIntensity_f8, &intensityFactor_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_fog_tensor(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              T *dstPtr,
                              RpptDescPtr dstDescPtr,
                              Rpp32f *d_fogAlphaMaskPtr,
                              Rpp32f *d_fogIntensityMaskPtr,
                              Rpp32f *intensityFactor,
                              Rpp32f *greyFactor,
                              Rpp32u *maskLocOffsetX,
                              Rpp32u *maskLocOffsetY,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    Rpp32s globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    Rpp32s globalThreads_y = dstDescPtr->h;
    Rpp32s globalThreads_z = dstDescPtr->n;

    // fill the random starting point (x, y) in mask for each image in batch
    std::random_device rd;  // Random number engine seed
    std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
    for (Rpp32s i = 0; i < dstDescPtr->n; i++)
    {
        std::uniform_int_distribution<> distribX(0, srcDescPtr->w - roiTensorPtrSrc[i].xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, srcDescPtr->h - roiTensorPtrSrc[i].xywhROI.roiHeight);
        maskLocOffsetX[i] = distribX(gen);
        maskLocOffsetY[i] = distribY(gen);
    }

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(fog_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           d_fogAlphaMaskPtr,
                           d_fogIntensityMaskPtr,
                           intensityFactor,
                           greyFactor,
                           maskLocOffsetX,
                           maskLocOffsetY,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(fog_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           d_fogAlphaMaskPtr,
                           d_fogIntensityMaskPtr,
                           intensityFactor,
                           greyFactor,
                           maskLocOffsetX,
                           maskLocOffsetY,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(fog_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               d_fogAlphaMaskPtr,
                               d_fogIntensityMaskPtr,
                               intensityFactor,
                               greyFactor,
                               maskLocOffsetX,
                               maskLocOffsetY,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(fog_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               d_fogAlphaMaskPtr,
                               d_fogIntensityMaskPtr,
                               intensityFactor,
                               greyFactor,
                               maskLocOffsetX,
                               maskLocOffsetY,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_fog_tensor<Rpp8u>(Rpp8u*,
                                              RpptDescPtr,
                                              Rpp8u*,
                                              RpptDescPtr,
                                              Rpp32f*,
                                              Rpp32f*,
                                              Rpp32f*,
                                              Rpp32f*,
                                              Rpp32u*,
                                              Rpp32u*,
                                              RpptROIPtr,
                                              RpptRoiType,
                                              rpp::Handle&);

template RppStatus hip_exec_fog_tensor<half>(half*,
                                             RpptDescPtr,
                                             half*,
                                             RpptDescPtr,
                                             Rpp32f*,
                                             Rpp32f*,
                                             Rpp32f*,
                                             Rpp32f*,
                                             Rpp32u*,
                                             Rpp32u*,
                                             RpptROIPtr,
                                             RpptRoiType,
                                             rpp::Handle&);

template RppStatus hip_exec_fog_tensor<Rpp32f>(Rpp32f*,
                                               RpptDescPtr,
                                               Rpp32f*,
                                               RpptDescPtr,
                                               Rpp32f*,
                                               Rpp32f*,
                                               Rpp32f*,
                                               Rpp32f*,
                                               Rpp32u*,
                                               Rpp32u*,
                                               RpptROIPtr,
                                               RpptRoiType,
                                               rpp::Handle&);

template RppStatus hip_exec_fog_tensor<Rpp8s>(Rpp8s*,
                                              RpptDescPtr,
                                              Rpp8s*,
                                              RpptDescPtr,
                                              Rpp32f*,
                                              Rpp32f*,
                                              Rpp32f*,
                                              Rpp32f*,
                                              Rpp32u*,
                                              Rpp32u*,
                                              RpptROIPtr,
                                              RpptRoiType,
                                              rpp::Handle&);
