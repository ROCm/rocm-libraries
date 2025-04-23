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
#include "rpp_hip_interpolation.hpp"

__device__ void resize_crop_mirror_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)(srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) / dstDimsWH->x;
    float hRatio = (float)(srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) / dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    locDst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [0-3]
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [4-7]
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [0-3]
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [4-7]
}

__device__ void resize_crop_mirror_roi_and_srclocs_hip_compute_mirror(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)(srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) / dstDimsWH->x;
    float hRatio = (float)(srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) / dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 decrement_f8, locDst_f8x, locDst_f8y;
    decrement_f8.f4[0] = make_float4(dstDimsWH->x - 1, dstDimsWH->x - 2, dstDimsWH->x - 3, dstDimsWH->x - 4);
    decrement_f8.f4[1] = make_float4(dstDimsWH->x - 5, dstDimsWH->x - 6, dstDimsWH->x - 7, dstDimsWH->x - 8);
    locDst_f8x.f4[0] = decrement_f8.f4[0] - (float4)id_x;
    locDst_f8x.f4[1] = decrement_f8.f4[1] - (float4)id_x;
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [width-1 - width-4]
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [width-5 - width-8]
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [0-3]
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [4-7]
}

template <typename T>
__global__ void resize_crop_mirror_bilinear_pkd_hip_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       RpptImagePatchPtr dstImgSize,
                                                       uint *mirrorTensor,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    if(mirrorTensor[id_z] == 1)
        resize_crop_mirror_roi_and_srclocs_hip_compute_mirror(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
    else
        resize_crop_mirror_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void resize_crop_mirror_bilinear_pln_hip_tensor(T *srcPtr,
                                                           uint3 srcStridesNCH,
                                                           T *dstPtr,
                                                           uint3 dstStridesNCH,
                                                           RpptImagePatchPtr dstImgSize,
                                                           int channelsDst,
                                                           uint *mirrorTensor,
                                                           RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];

    d_float16 locSrc_f16;
    if(mirrorTensor[id_z] == 1)
        resize_crop_mirror_roi_and_srclocs_hip_compute_mirror(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
    else
        resize_crop_mirror_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void resize_crop_mirror_bilinear_pkd3_pln3_hip_tensor(T *srcPtr,
                                                                 uint2 srcStridesNH,
                                                                 T *dstPtr,
                                                                 uint3 dstStridesNCH,
                                                                 RpptImagePatchPtr dstImgSize,
                                                                 uint *mirrorTensor,
                                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    if(mirrorTensor[id_z] == 1)
        resize_crop_mirror_roi_and_srclocs_hip_compute_mirror(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
    else
        resize_crop_mirror_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void resize_crop_mirror_bilinear_pln3_pkd3_hip_tensor(T *srcPtr,
                                                                 uint3 srcStridesNCH,
                                                                 T *dstPtr,
                                                                 uint2 dstStridesNH,
                                                                 RpptImagePatchPtr dstImgSize,
                                                                 uint *mirrorTensor,
                                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    if(mirrorTensor[id_z] == 1)
        resize_crop_mirror_roi_and_srclocs_hip_compute_mirror(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);
    else
        resize_crop_mirror_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_resize_crop_mirror_tensor(T *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             T *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptImagePatchPtr dstImgSizes,
                                             RpptInterpolationType interpolationType,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             rpp::Handle& handle)
{
    if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        if (roiType == RpptRoiType::XYWH)
            hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(resize_crop_mirror_bilinear_pkd_hip_tensor,
                            dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                            dstImgSizes,
                            handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(resize_crop_mirror_bilinear_pln_hip_tensor,
                            dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                            dstImgSizes,
                            dstDescPtr->c,
                            handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(resize_crop_mirror_bilinear_pkd3_pln3_hip_tensor,
                                dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                dstImgSizes,
                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                roiTensorPtrSrc);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (dstDescPtr->w + 7) >> 3;
                hipLaunchKernelGGL(resize_crop_mirror_bilinear_pln3_pkd3_hip_tensor,
                                dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                dstImgSizes,
                                handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                roiTensorPtrSrc);
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_resize_crop_mirror_tensor<Rpp8u>(Rpp8u*,
                                                             RpptDescPtr,
                                                             Rpp8u*,
                                                             RpptDescPtr,
                                                             RpptImagePatchPtr,
                                                             RpptInterpolationType,
                                                             RpptROIPtr,
                                                             RpptRoiType,
                                                             rpp::Handle&);

template RppStatus hip_exec_resize_crop_mirror_tensor<half>(half*,
                                                            RpptDescPtr,
                                                            half*,
                                                            RpptDescPtr,
                                                            RpptImagePatchPtr,
                                                            RpptInterpolationType,
                                                            RpptROIPtr,
                                                            RpptRoiType,
                                                            rpp::Handle&);

template RppStatus hip_exec_resize_crop_mirror_tensor<Rpp32f>(Rpp32f*,
                                                              RpptDescPtr,
                                                              Rpp32f*,
                                                              RpptDescPtr,
                                                              RpptImagePatchPtr,
                                                              RpptInterpolationType,
                                                              RpptROIPtr,
                                                              RpptRoiType,
                                                              rpp::Handle&);

template RppStatus hip_exec_resize_crop_mirror_tensor<Rpp8s>(Rpp8s*,
                                                             RpptDescPtr,
                                                             Rpp8s*,
                                                             RpptDescPtr,
                                                             RpptImagePatchPtr,
                                                             RpptInterpolationType,
                                                             RpptROIPtr,
                                                             RpptRoiType,
                                                             rpp::Handle&);
