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

__device__ void exposure_hip_compute(uchar *srcPtr, d_float8 *pix_f8, float4 *exposureParam_f4)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255(pix_f8->f4[0] * *exposureParam_f4);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255(pix_f8->f4[1] * *exposureParam_f4);
}

__device__ void exposure_hip_compute(float *srcPtr, d_float8 *pix_f8, float4 *exposureParam_f4)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to1(pix_f8->f4[0] * *exposureParam_f4);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to1(pix_f8->f4[1] * *exposureParam_f4);
}

__device__ void exposure_hip_compute(schar *srcPtr, d_float8 *pix_f8, float4 *exposureParam_f4)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255((pix_f8->f4[0] + (float4)128) * *exposureParam_f4) - (float4)128;
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255((pix_f8->f4[1] + (float4)128) * *exposureParam_f4) - (float4)128;
}

__device__ void exposure_hip_compute(half *srcPtr, d_float8 *pix_f8, float4 *exposureParam_f4)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to1(pix_f8->f4[0] * *exposureParam_f4);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to1(pix_f8->f4[1] * *exposureParam_f4);
}

template <typename T>
__global__ void exposure_pkd_hip_tensor(T *srcPtr,
                                        uint2 srcStridesNH,
                                        T *dstPtr,
                                        uint2 dstStridesNH,
                                        float *exposureFactorTensor,
                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    float multiplyingFactor = powf(2.0f, exposureFactorTensor[id_z]);
    float4 exposureParam_f4 = (float4)multiplyingFactor;

    d_float8 pix_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    exposure_hip_compute(srcPtr, &pix_f8, &exposureParam_f4);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
}

template <typename T>
__global__ void exposure_pln_hip_tensor(T *srcPtr,
                                        uint3 srcStridesNCH,
                                        T *dstPtr,
                                        uint3 dstStridesNCH,
                                        int channelsDst,
                                        float *exposureFactorTensor,
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

    float multiplyingFactor = powf(2.0f, exposureFactorTensor[id_z]);
    float4 exposureParam_f4 = (float4)multiplyingFactor;

    d_float8 pix_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    exposure_hip_compute(srcPtr, &pix_f8, &exposureParam_f4);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        exposure_hip_compute(srcPtr, &pix_f8, &exposureParam_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        exposure_hip_compute(srcPtr, &pix_f8, &exposureParam_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
    }
}

template <typename T>
__global__ void exposure_pkd3_pln3_hip_tensor(T *srcPtr,
                                              uint2 srcStridesNH,
                                              T *dstPtr,
                                              uint3 dstStridesNCH,
                                              float *exposureFactorTensor,
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

    float multiplyingFactor = powf(2.0f, exposureFactorTensor[id_z]);
    float4 exposureParam_f4 = (float4)multiplyingFactor;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    exposure_hip_compute(srcPtr, &pix_f24.f8[0], &exposureParam_f4);
    exposure_hip_compute(srcPtr, &pix_f24.f8[1], &exposureParam_f4);
    exposure_hip_compute(srcPtr, &pix_f24.f8[2], &exposureParam_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void exposure_pln3_pkd3_hip_tensor(T *srcPtr,
                                              uint3 srcStridesNCH,
                                              T *dstPtr,
                                              uint2 dstStridesNH,
                                              float *exposureFactorTensor,
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

    float multiplyingFactor = powf(2.0f, exposureFactorTensor[id_z]);
    float4 exposureParam_f4 = (float4)multiplyingFactor;

    d_float24 pix_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    exposure_hip_compute(srcPtr, &pix_f24.f8[0], &exposureParam_f4);
    exposure_hip_compute(srcPtr, &pix_f24.f8[1], &exposureParam_f4);
    exposure_hip_compute(srcPtr, &pix_f24.f8[2], &exposureParam_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_exposure_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(exposure_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(exposure_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(exposure_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(exposure_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_exposure_tensor<Rpp8u>(Rpp8u*,
                                                   RpptDescPtr,
                                                   Rpp8u*,
                                                   RpptDescPtr,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus hip_exec_exposure_tensor<half>(half*,
                                                  RpptDescPtr,
                                                  half*,
                                                  RpptDescPtr,
                                                  RpptROIPtr,
                                                  RpptRoiType,
                                                  rpp::Handle&);

template RppStatus hip_exec_exposure_tensor<Rpp32f>(Rpp32f*,
                                                    RpptDescPtr,
                                                    Rpp32f*,
                                                    RpptDescPtr,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);

template RppStatus hip_exec_exposure_tensor<Rpp8s>(Rpp8s*,
                                                   RpptDescPtr,
                                                   Rpp8s*,
                                                   RpptDescPtr,
                                                   RpptROIPtr,
                                                   RpptRoiType,
                                                   rpp::Handle&);
