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

__device__ void lut_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, uchar *lut)
{
    dst_f8->f4[0] = make_float4((float)lut[(int) src_f8->f1[0]], (float)lut[(int) src_f8->f1[1]], (float)lut[(int) src_f8->f1[2]], (float)lut[(int) src_f8->f1[3]]);
    dst_f8->f4[1] = make_float4((float)lut[(int) src_f8->f1[4]], (float)lut[(int) src_f8->f1[5]], (float)lut[(int) src_f8->f1[6]], (float)lut[(int) src_f8->f1[7]]);
}

__device__ void lut_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, signed char *lut)
{
    dst_f8->f4[0] = make_float4((float)lut[(int) src_f8->f1[0]], (float)lut[(int) src_f8->f1[1]], (float)lut[(int) src_f8->f1[2]], (float)lut[(int) src_f8->f1[3]]);
    dst_f8->f4[1] = make_float4((float)lut[(int) src_f8->f1[4]], (float)lut[(int) src_f8->f1[5]], (float)lut[(int) src_f8->f1[6]], (float)lut[(int) src_f8->f1[7]]);
}

__device__ void lut_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *lut)
{
    dst_f8->f4[0] = make_float4(lut[(int) src_f8->f1[0]], lut[(int) src_f8->f1[1]], lut[(int) src_f8->f1[2]], lut[(int) src_f8->f1[3]]);
    dst_f8->f4[1] = make_float4(lut[(int) src_f8->f1[4]], lut[(int) src_f8->f1[5]], lut[(int) src_f8->f1[6]], lut[(int) src_f8->f1[7]]);
}

__device__ void lut_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, half *lut)
{
    dst_f8->f4[0] = make_float4(lut[(int) src_f8->f1[0]], lut[(int) src_f8->f1[1]], lut[(int) src_f8->f1[2]], lut[(int) src_f8->f1[3]]);
    dst_f8->f4[1] = make_float4(lut[(int) src_f8->f1[4]], lut[(int) src_f8->f1[5]], lut[(int) src_f8->f1[6]], lut[(int) src_f8->f1[7]]);
}

__device__ void lut_hip_compute(signed char *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, signed char *lut)
{
    d_float8 srcNorm_f8;
    srcNorm_f8.f4[0] = src_f8->f4[0] + (float4)128;
    srcNorm_f8.f4[1] = src_f8->f4[1] + (float4)128;
    dst_f8->f4[0] = make_float4((float)lut[(int) srcNorm_f8.f1[0]], (float)lut[(int) srcNorm_f8.f1[1]], (float)lut[(int) srcNorm_f8.f1[2]], (float)lut[(int) srcNorm_f8.f1[3]]);
    dst_f8->f4[1] = make_float4((float)lut[(int) srcNorm_f8.f1[4]], (float)lut[(int) srcNorm_f8.f1[5]], (float)lut[(int) srcNorm_f8.f1[6]], (float)lut[(int) srcNorm_f8.f1[7]]);
}

template <typename T1, typename T2>
__global__ void lut_pkd_hip_tensor(T1 *srcPtr,
                               uint2 srcStridesNH,
                               T2 *dstPtr,
                               uint2 dstStridesNH,
                               T2 *lutPtr,
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

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    lut_hip_compute(srcPtr, &src_f8, &dst_f8, lutPtr);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T1, typename T2>
__global__ void lut_pln_hip_tensor(T1 *srcPtr,
                               uint3 srcStridesNCH,
                               T2 *dstPtr,
                               uint3 dstStridesNCH,
                               T2 *lutPtr,
                               int channelsDst,
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

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    lut_hip_compute(srcPtr, &src_f8, &dst_f8, lutPtr);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        lut_hip_compute(srcPtr, &src_f8, &dst_f8, lutPtr);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        lut_hip_compute(srcPtr, &src_f8, &dst_f8, lutPtr);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T1, typename T2>
__global__ void lut_pkd3_pln3_hip_tensor(T1 *srcPtr,
                                     uint2 srcStridesNH,
                                     T2 *dstPtr,
                                     uint3 dstStridesNCH,
                                     T2 *lutPtr,
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

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    lut_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], lutPtr);
    lut_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], lutPtr);
    lut_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], lutPtr);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T1, typename T2>
__global__ void lut_pln3_pkd3_hip_tensor(T1 *srcPtr,
                                     uint3 srcStridesNCH,
                                     T2 *dstPtr,
                                     uint2 dstStridesNH,
                                     T2 *lutPtr,
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

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    lut_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], lutPtr);
    lut_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], lutPtr);
    lut_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], lutPtr);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T, typename U>
RppStatus hip_exec_lut_tensor(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *dstPtr,
                              RpptDescPtr dstDescPtr,
                              U *lutPtr,
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
        hipLaunchKernelGGL(lut_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           lutPtr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(lut_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           lutPtr,
                           dstDescPtr->c,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(lut_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               lutPtr,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(lut_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               lutPtr,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_lut_tensor<Rpp8u, Rpp8u>(Rpp8u*,
                                                     RpptDescPtr,
                                                     Rpp8u*,
                                                     RpptDescPtr,
                                                     Rpp8u*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_lut_tensor<Rpp8u, half>(Rpp8u*,
                                                    RpptDescPtr,
                                                    half*,
                                                    RpptDescPtr,
                                                    half*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);

template RppStatus hip_exec_lut_tensor<Rpp8u, Rpp32f>(Rpp8u*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      rpp::Handle&);

template RppStatus hip_exec_lut_tensor<Rpp8s, Rpp8s>(Rpp8s*,
                                                     RpptDescPtr,
                                                     Rpp8s*,
                                                     RpptDescPtr,
                                                     Rpp8s*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);
