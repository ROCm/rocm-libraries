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

__device__ void water_roi_and_srclocs_hip_compute(int id_x, int id_y, float4 *amplX_f4, float4 *amplY_f4,
                                                  float freqX, float freqY, float phaseX, float phaseY, d_float16 *locSrc_f16)
{
    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = static_cast<float4>(id_x) + increment_f8.f4[0];
    locDst_f8x.f4[1] = static_cast<float4>(id_x) + increment_f8.f4[1];
    locDst_f8y.f4[0] = static_cast<float4>(id_y);
    locDst_f8y.f4[1] = static_cast<float4>(id_y);

    d_float8 sinFactor_f8, cosFactor_f8;
    sinFactor_f8.f4[0] = static_cast<float4>((sinf(fmaf(freqX, static_cast<float>(id_y), phaseX))));
    sinFactor_f8.f4[1] = sinFactor_f8.f4[0];
    cosFactor_f8.f1[0] = cosf(fmaf(freqY, locDst_f8x.f1[0], phaseY));
    cosFactor_f8.f1[1] = cosf(fmaf(freqY, locDst_f8x.f1[1], phaseY));
    cosFactor_f8.f1[2] = cosf(fmaf(freqY, locDst_f8x.f1[2], phaseY));
    cosFactor_f8.f1[3] = cosf(fmaf(freqY, locDst_f8x.f1[3], phaseY));
    cosFactor_f8.f1[4] = cosf(fmaf(freqY, locDst_f8x.f1[4], phaseY));
    cosFactor_f8.f1[5] = cosf(fmaf(freqY, locDst_f8x.f1[5], phaseY));
    cosFactor_f8.f1[6] = cosf(fmaf(freqY, locDst_f8x.f1[6], phaseY));
    cosFactor_f8.f1[7] = cosf(fmaf(freqY, locDst_f8x.f1[7], phaseY));

    locSrc_f16->f4[0] =  locDst_f8x.f4[0] + (*amplX_f4 * sinFactor_f8.f4[0]);  // Compute src x locations in float for dst x locations [0-3]
    locSrc_f16->f4[1] =  locDst_f8x.f4[1] + (*amplX_f4 * sinFactor_f8.f4[1]);  // Compute src x locations in float for dst x locations [4-7]
    locSrc_f16->f4[2] =  locDst_f8y.f4[0] + (*amplY_f4 * cosFactor_f8.f4[0]);  // Compute src y locations in float for dst y locations [0-3]
    locSrc_f16->f4[3] =  locDst_f8y.f4[1] + (*amplY_f4 * cosFactor_f8.f4[1]);  // Compute src y locations in float for dst y locations [4-7]
}

template <typename T>
__global__ void water_pkd_tensor(T *srcPtr,
                                 uint2 srcStridesNH,
                                 T *dstPtr,
                                 uint2 dstStridesNH,
                                 float *amplXTensor,
                                 float *amplYTensor,
                                 float *freqXTensor,
                                 float *freqYTensor,
                                 float *phaseXTensor,
                                 float *phaseYTensor,
                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y > roiTensorPtrSrc[id_z].ltrbROI.rb.y) || (id_x > roiTensorPtrSrc[id_z].ltrbROI.rb.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float4 amplX_f4 = static_cast<float4>(amplXTensor[id_z]);
    float4 amplY_f4 = static_cast<float4>(amplYTensor[id_z]);
    float freqX = freqXTensor[id_z];
    float freqY = freqYTensor[id_z];
    float phaseX = phaseXTensor[id_z];
    float phaseY = phaseYTensor[id_z];

    int4 srcRoi_i4 = *(reinterpret_cast<int4 *>(&roiTensorPtrSrc[id_z]));
    d_float16 locSrc_f16;
    water_roi_and_srclocs_hip_compute(id_x, id_y, &amplX_f4, &amplY_f4, freqX, freqY, phaseX, phaseY, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void water_pln_tensor(T *srcPtr,
                                 uint3 srcStridesNCH,
                                 T *dstPtr,
                                 uint3 dstStridesNCH,
                                 int channelsDst,
                                 float *amplXTensor,
                                 float *amplYTensor,
                                 float *freqXTensor,
                                 float *freqYTensor,
                                 float *phaseXTensor,
                                 float *phaseYTensor,
                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y > roiTensorPtrSrc[id_z].ltrbROI.rb.y) || (id_x > roiTensorPtrSrc[id_z].ltrbROI.rb.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float4 amplX_f4 = static_cast<float4>(amplXTensor[id_z]);
    float4 amplY_f4 = static_cast<float4>(amplYTensor[id_z]);
    float freqX = freqXTensor[id_z];
    float freqY = freqYTensor[id_z];
    float phaseX = phaseXTensor[id_z];
    float phaseY = phaseYTensor[id_z];

    int4 srcRoi_i4 = *(reinterpret_cast<int4 *>(&roiTensorPtrSrc[id_z]));
    d_float16 locSrc_f16;
    water_roi_and_srclocs_hip_compute(id_x, id_y, &amplX_f4, &amplY_f4, freqX, freqY, phaseX, phaseY, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void water_pkd3_pln3_tensor(T *srcPtr,
                                       uint2 srcStridesNH,
                                       T *dstPtr,
                                       uint3 dstStridesNCH,
                                       float *amplXTensor,
                                       float *amplYTensor,
                                       float *freqXTensor,
                                       float *freqYTensor,
                                       float *phaseXTensor,
                                       float *phaseYTensor,
                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y > roiTensorPtrSrc[id_z].ltrbROI.rb.y) || (id_x > roiTensorPtrSrc[id_z].ltrbROI.rb.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float4 amplX_f4 = static_cast<float4>(amplXTensor[id_z]);
    float4 amplY_f4 = static_cast<float4>(amplYTensor[id_z]);
    float freqX = freqXTensor[id_z];
    float freqY = freqYTensor[id_z];
    float phaseX = phaseXTensor[id_z];
    float phaseY = phaseYTensor[id_z];

    int4 srcRoi_i4 = *(reinterpret_cast<int4 *>(&roiTensorPtrSrc[id_z]));
    d_float16 locSrc_f16;
    water_roi_and_srclocs_hip_compute(id_x, id_y, &amplX_f4, &amplY_f4, freqX, freqY, phaseX, phaseY, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void water_pln3_pkd3_tensor(T *srcPtr,
                                       uint3 srcStridesNCH,
                                       T *dstPtr,
                                       uint2 dstStridesNH,
                                       float *amplXTensor,
                                       float *amplYTensor,
                                       float *freqXTensor,
                                       float *freqYTensor,
                                       float *phaseXTensor,
                                       float *phaseYTensor,
                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y > roiTensorPtrSrc[id_z].ltrbROI.rb.y) || (id_x > roiTensorPtrSrc[id_z].ltrbROI.rb.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float4 amplX_f4 = static_cast<float4>(amplXTensor[id_z]);
    float4 amplY_f4 = static_cast<float4>(amplYTensor[id_z]);
    float freqX = freqXTensor[id_z];
    float freqY = freqYTensor[id_z];
    float phaseX = phaseXTensor[id_z];
    float phaseY = phaseYTensor[id_z];

    int4 srcRoi_i4 = *(reinterpret_cast<int4 *>(&roiTensorPtrSrc[id_z]));
    d_float16 locSrc_f16;
    water_roi_and_srclocs_hip_compute(id_x, id_y, &amplX_f4, &amplY_f4, freqX, freqY, phaseX, phaseY, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_water_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(water_pkd_tensor,
                           dim3(ceil(static_cast<float>(globalThreads_x)/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[4].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[5].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(water_pln_tensor,
                           dim3(ceil(static_cast<float>(globalThreads_x)/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[4].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[5].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(water_pkd3_pln3_tensor,
                               dim3(ceil(static_cast<float>(globalThreads_x)/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[4].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[5].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(water_pln3_pkd3_tensor,
                               dim3(ceil(static_cast<float>(globalThreads_x)/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[4].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[5].floatmem,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_water_tensor<Rpp8u>(Rpp8u*,
                                                RpptDescPtr,
                                                Rpp8u*,
                                                RpptDescPtr,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_water_tensor<half>(half*,
                                                RpptDescPtr,
                                                half*,
                                                RpptDescPtr,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_water_tensor<Rpp32f>(Rpp32f*,
                                                RpptDescPtr,
                                                Rpp32f*,
                                                RpptDescPtr,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_water_tensor<Rpp8s>(Rpp8s*,
                                                RpptDescPtr,
                                                Rpp8s*,
                                                RpptDescPtr,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);
