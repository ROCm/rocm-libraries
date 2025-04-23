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
#include "rpp_hip_math.hpp"

/* BitwiseXOR is logical operation only on U8 types.*/

__device__ void bitwise_xor_hip_compute(d_uchar8 *src1_uc8, d_uchar8 *src2_uc8, d_uchar8 *dst_uc8)
{
    rpp_hip_math_bitwiseXor8(src1_uc8, src2_uc8, dst_uc8);
}

__global__ void bitwise_xor_pkd_hip_tensor(Rpp8u *srcPtr1,
                                           Rpp8u *srcPtr2,
                                           uint2 srcStridesNH,
                                           Rpp8u *dstPtr,
                                           uint2 dstStridesNH,
                                           RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_uchar24 src1_uc24, src2_uc24, dst_uc24;

    rpp_hip_load24_pkd3_and_unpack_to_uchar24_pkd3(srcPtr1 + srcIdx, &src1_uc24);
    rpp_hip_load24_pkd3_and_unpack_to_uchar24_pkd3(srcPtr2 + srcIdx, &src2_uc24);
    bitwise_xor_hip_compute(&src1_uc24.uc8[0], &src2_uc24.uc8[0], &dst_uc24.uc8[0]);
    bitwise_xor_hip_compute(&src1_uc24.uc8[1], &src2_uc24.uc8[1], &dst_uc24.uc8[1]);
    bitwise_xor_hip_compute(&src1_uc24.uc8[2], &src2_uc24.uc8[2], &dst_uc24.uc8[2]);
    rpp_hip_pack_uchar24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_uc24);
}

__global__ void bitwise_xor_pln_hip_tensor(Rpp8u *srcPtr1,
                                           Rpp8u *srcPtr2,
                                           uint3 srcStridesNCH,
                                           Rpp8u *dstPtr,
                                           uint3 dstStridesNCH,
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

    d_uchar8 src1_uc8, src2_uc8, dst_uc8;
    uchar* src1Ptr_uc8 = (uchar*)&src1_uc8;
    uchar* src2Ptr_uc8 = (uchar*)&src2_uc8;

    rpp_hip_load8_to_uchar8(srcPtr1 + srcIdx, src1Ptr_uc8);
    rpp_hip_load8_to_uchar8(srcPtr2 + srcIdx, src2Ptr_uc8);
    bitwise_xor_hip_compute(&src1_uc8, &src2_uc8, &dst_uc8);
    rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_to_uchar8(srcPtr1 + srcIdx, src1Ptr_uc8);
        rpp_hip_load8_to_uchar8(srcPtr2 + srcIdx, src2Ptr_uc8);
        bitwise_xor_hip_compute(&src1_uc8, &src2_uc8, &dst_uc8);
        rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_to_uchar8(srcPtr1 + srcIdx, src1Ptr_uc8);
        rpp_hip_load8_to_uchar8(srcPtr2 + srcIdx, src2Ptr_uc8);
        bitwise_xor_hip_compute(&src1_uc8, &src2_uc8, &dst_uc8);
        rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);
    }
}

__global__ void bitwise_xor_pkd3_pln3_hip_tensor(Rpp8u *srcPtr1,
                                                 Rpp8u *srcPtr2,
                                                 uint2 srcStridesNH,
                                                 Rpp8u *dstPtr,
                                                 uint3 dstStridesNCH,
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

    d_uchar24 src1_uc24, src2_uc24, dst_uc24;

    rpp_hip_load24_pkd3_and_unpack_to_uchar24_pln3(srcPtr1 + srcIdx, &src1_uc24);
    rpp_hip_load24_pkd3_and_unpack_to_uchar24_pln3(srcPtr2 + srcIdx, &src2_uc24);
    bitwise_xor_hip_compute(&src1_uc24.uc8[0], &src2_uc24.uc8[0], &dst_uc24.uc8[0]);
    bitwise_xor_hip_compute(&src1_uc24.uc8[1], &src2_uc24.uc8[1], &dst_uc24.uc8[1]);
    bitwise_xor_hip_compute(&src1_uc24.uc8[2], &src2_uc24.uc8[2], &dst_uc24.uc8[2]);
    rpp_hip_pack_uchar24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_uc24);
}

__global__ void bitwise_xor_pln3_pkd3_hip_tensor(Rpp8u *srcPtr1,
                                                 Rpp8u *srcPtr2,
                                                 uint3 srcStridesNCH,
                                                 Rpp8u *dstPtr,
                                                 uint2 dstStridesNH,
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

    d_uchar24 src1_uc24, src2_uc24, dst_uc24;

    rpp_hip_load24_pln3_and_unpack_to_uchar24_pkd3(srcPtr1 + srcIdx, srcStridesNCH.y, &src1_uc24);
    rpp_hip_load24_pln3_and_unpack_to_uchar24_pkd3(srcPtr2 + srcIdx, srcStridesNCH.y, &src2_uc24);
    bitwise_xor_hip_compute(&src1_uc24.uc8[0], &src2_uc24.uc8[0], &dst_uc24.uc8[0]);
    bitwise_xor_hip_compute(&src1_uc24.uc8[1], &src2_uc24.uc8[1], &dst_uc24.uc8[1]);
    bitwise_xor_hip_compute(&src1_uc24.uc8[2], &src2_uc24.uc8[2], &dst_uc24.uc8[2]);
    rpp_hip_pack_uchar24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_uc24);
}

RppStatus hip_exec_bitwise_xor_tensor(Rpp8u *srcPtr1,
                                       Rpp8u *srcPtr2,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(bitwise_xor_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(bitwise_xor_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(bitwise_xor_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(bitwise_xor_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
