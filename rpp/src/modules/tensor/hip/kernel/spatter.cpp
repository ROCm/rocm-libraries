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
#include "kernel_dims.hpp"
#include "spatter_mask.hpp"

__device__ void spatter_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *mask_f8, d_float8 *maskInv_f8, float4 *pix_f4)
{
    dst_f8->f4[0] = (src_f8->f4[0] * maskInv_f8->f4[0]) + (*pix_f4 * mask_f8->f4[0]);
    dst_f8->f4[1] = (src_f8->f4[1] * maskInv_f8->f4[1]) + (*pix_f4 * mask_f8->f4[1]);
}

__device__ void spatter_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *mask_f8, d_float8 *maskInv_f8, float4 *pix_f4)
{
    float4 pixNorm_f4 = *pix_f4 * (float4) ONE_OVER_255;
    dst_f8->f4[0] = (src_f8->f4[0] * maskInv_f8->f4[0]) + (pixNorm_f4 * mask_f8->f4[0]);
    dst_f8->f4[1] = (src_f8->f4[1] * maskInv_f8->f4[1]) + (pixNorm_f4 * mask_f8->f4[1]);
}

__device__ void spatter_hip_compute(schar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *mask_f8, d_float8 *maskInv_f8, float4 *pix_f4)
{
    dst_f8->f4[0] = ((src_f8->f4[0] + (float4)128) * maskInv_f8->f4[0]) + (*pix_f4 * mask_f8->f4[0]) - (float4)128;
    dst_f8->f4[1] = ((src_f8->f4[1] + (float4)128) * maskInv_f8->f4[1]) + (*pix_f4 * mask_f8->f4[1]) - (float4)128;
}

__device__ void spatter_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *mask_f8, d_float8 *maskInv_f8, float4 *pix_f4)
{
    float4 pixNorm_f4 = *pix_f4 * (float4) ONE_OVER_255;
    dst_f8->f4[0] = (src_f8->f4[0] * maskInv_f8->f4[0]) + (pixNorm_f4 * mask_f8->f4[0]);
    dst_f8->f4[1] = (src_f8->f4[1] * maskInv_f8->f4[1]) + (pixNorm_f4 * mask_f8->f4[1]);
}

template <typename T>
__global__ void spatter_pkd_hip_tensor(T *srcPtr,
                                       uint2 srcStridesNH,
                                       T *dstPtr,
                                       uint2 dstStridesNH,
                                       float *spatterMaskPtr,
                                       float *spatterMaskInvPtr,
                                       uint *maskLocArrX,
                                       uint *maskLocArrY,
                                       float3 spatterColor,
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
    uint maskIdx = (SPATTER_MAX_WIDTH * (maskLocArrY[id_z] + id_y)) + maskLocArrX[id_z] + id_x;

    d_float8 mask_f8, maskInv_f8;
    *(d_float8_s *)&mask_f8 = *(d_float8_s *)&spatterMaskPtr[maskIdx];
    *(d_float8_s *)&maskInv_f8 = *(d_float8_s *)&spatterMaskInvPtr[maskIdx];
    float4 r_f4 = (float4)(spatterColor.x);
    float4 g_f4 = (float4)(spatterColor.y);
    float4 b_f4 = (float4)(spatterColor.z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    spatter_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &mask_f8, &maskInv_f8, &r_f4);
    spatter_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &mask_f8, &maskInv_f8, &g_f4);
    spatter_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &mask_f8, &maskInv_f8, &b_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void spatter_pln_hip_tensor(T *srcPtr,
                                       uint3 srcStridesNCH,
                                       T *dstPtr,
                                       uint3 dstStridesNCH,
                                       int channelsDst,
                                       float *spatterMaskPtr,
                                       float *spatterMaskInvPtr,
                                       uint *maskLocArrX,
                                       uint *maskLocArrY,
                                       float3 spatterColor,
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
    uint maskIdx = (SPATTER_MAX_WIDTH * (maskLocArrY[id_z] + id_y)) + maskLocArrX[id_z] + id_x;

    d_float8 mask_f8, maskInv_f8;
    *(d_float8_s *)&mask_f8 = *(d_float8_s *)&spatterMaskPtr[maskIdx];
    *(d_float8_s *)&maskInv_f8 = *(d_float8_s *)&spatterMaskInvPtr[maskIdx];
    float4 r_f4 = (float4)(spatterColor.x);
    float4 g_f4 = (float4)(spatterColor.y);
    float4 b_f4 = (float4)(spatterColor.z);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    spatter_hip_compute(srcPtr, &src_f8, &dst_f8, &mask_f8, &maskInv_f8, &r_f4);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        spatter_hip_compute(srcPtr, &src_f8, &dst_f8, &mask_f8, &maskInv_f8, &g_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        spatter_hip_compute(srcPtr, &src_f8, &dst_f8, &mask_f8, &maskInv_f8, &b_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void spatter_pkd3_pln3_hip_tensor(T *srcPtr,
                                             uint2 srcStridesNH,
                                             T *dstPtr,
                                             uint3 dstStridesNCH,
                                             float *spatterMaskPtr,
                                             float *spatterMaskInvPtr,
                                             uint *maskLocArrX,
                                             uint *maskLocArrY,
                                             float3 spatterColor,
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
    uint maskIdx = (SPATTER_MAX_WIDTH * (maskLocArrY[id_z] + id_y)) + maskLocArrX[id_z] + id_x;

    d_float8 mask_f8, maskInv_f8;
    *(d_float8_s *)&mask_f8 = *(d_float8_s *)&spatterMaskPtr[maskIdx];
    *(d_float8_s *)&maskInv_f8 = *(d_float8_s *)&spatterMaskInvPtr[maskIdx];
    float4 r_f4 = (float4)(spatterColor.x);
    float4 g_f4 = (float4)(spatterColor.y);
    float4 b_f4 = (float4)(spatterColor.z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    spatter_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &mask_f8, &maskInv_f8, &r_f4);
    spatter_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &mask_f8, &maskInv_f8, &g_f4);
    spatter_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &mask_f8, &maskInv_f8, &b_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void spatter_pln3_pkd3_hip_tensor(T *srcPtr,
                                             uint3 srcStridesNCH,
                                             T *dstPtr,
                                             uint2 dstStridesNH,
                                             float *spatterMaskPtr,
                                             float *spatterMaskInvPtr,
                                             uint *maskLocArrX,
                                             uint *maskLocArrY,
                                             float3 spatterColor,
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
    uint maskIdx = (SPATTER_MAX_WIDTH * (maskLocArrY[id_z] + id_y)) + maskLocArrX[id_z] + id_x;

    d_float8 mask_f8, maskInv_f8;
    *(d_float8_s *)&mask_f8 = *(d_float8_s *)&spatterMaskPtr[maskIdx];
    *(d_float8_s *)&maskInv_f8 = *(d_float8_s *)&spatterMaskInvPtr[maskIdx];
    float4 r_f4 = (float4)(spatterColor.x);
    float4 g_f4 = (float4)(spatterColor.y);
    float4 b_f4 = (float4)(spatterColor.z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    spatter_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &mask_f8, &maskInv_f8, &r_f4);
    spatter_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &mask_f8, &maskInv_f8, &g_f4);
    spatter_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &mask_f8, &maskInv_f8, &b_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_spatter_tensor(T *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  T *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptRGB spatterColor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    float3 spatterColor_f3;
    if (dstDescPtr->c == 3)
    {
        spatterColor_f3 = make_float3((float)spatterColor.B, (float)spatterColor.G, (float)spatterColor.R);
    }
    else if (dstDescPtr->c == 1)
    {
        float meanGreyVal = ((float)spatterColor.B + (float)spatterColor.G + (float)spatterColor.R) * 0.3333;
        spatterColor_f3 = make_float3(meanGreyVal, meanGreyVal, meanGreyVal);
    }

    Rpp32u maskSize = SPATTER_MAX_WIDTH * SPATTER_MAX_HEIGHT;
    Rpp32u maskSizeFloat = maskSize * sizeof(float);
    float *spatterMaskPtr, *spatterMaskInvPtr;
    spatterMaskPtr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    spatterMaskInvPtr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem + maskSize;
    CHECK_RETURN_STATUS(hipMemcpy(spatterMaskPtr, spatterMask, maskSizeFloat, hipMemcpyHostToDevice));
    CHECK_RETURN_STATUS(hipMemcpy(spatterMaskInvPtr, spatterMaskInv, maskSizeFloat, hipMemcpyHostToDevice));

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(spatter_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           spatterMaskPtr,
                           spatterMaskInvPtr,
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           spatterColor_f3,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(spatter_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           spatterMaskPtr,
                           spatterMaskInvPtr,
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           spatterColor_f3,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(spatter_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               spatterMaskPtr,
                               spatterMaskInvPtr,
                               handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                               spatterColor_f3,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(spatter_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               spatterMaskPtr,
                               spatterMaskInvPtr,
                               handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                               spatterColor_f3,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_spatter_tensor<Rpp8u>(Rpp8u*,
                                                  RpptDescPtr,
                                                  Rpp8u*,
                                                  RpptDescPtr,
                                                  RpptRGB,
                                                  RpptROIPtr,
                                                  RpptRoiType,
                                                  rpp::Handle&);

template RppStatus hip_exec_spatter_tensor<half>(half*,
                                                 RpptDescPtr,
                                                 half*,
                                                 RpptDescPtr,
                                                 RpptRGB,
                                                 RpptROIPtr,
                                                 RpptRoiType,
                                                 rpp::Handle&);

template RppStatus hip_exec_spatter_tensor<Rpp32f>(Rpp32f*,
                                                   RpptDescPtr,
                                                   Rpp32f*,
                                                   RpptDescPtr,
                                                   RpptRGB,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus hip_exec_spatter_tensor<Rpp8s>(Rpp8s*,
                                                  RpptDescPtr,
                                                  Rpp8s*,
                                                  RpptDescPtr,
                                                  RpptRGB,
                                                  RpptROIPtr,
                                                  RpptRoiType,
                                                  rpp::Handle&);
