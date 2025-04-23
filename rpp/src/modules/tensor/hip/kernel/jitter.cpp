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
#include "rpp_hip_random.hpp"
#include "rpp_hip_interpolation.hpp"
#include "rng_seed_stream.hpp"

__device__ __forceinline__ void jitter_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, RpptXorwowStateBoxMuller *xorwowState, uint kernelSize, uint bound, int id_x, int id_y, d_float16 *locSrc_f16)
{
    d_float8 widthIncrement_f8, heightIncrement_f8;
    rpp_hip_rng_8_xorwow_f32(xorwowState, &widthIncrement_f8);
    rpp_hip_math_multiply8_const(&widthIncrement_f8, &widthIncrement_f8, static_cast<float4>(kernelSize));
    rpp_hip_rng_8_xorwow_f32(xorwowState, &heightIncrement_f8);
    rpp_hip_math_multiply8_const(&heightIncrement_f8, &heightIncrement_f8, static_cast<float4>(kernelSize));

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);   // 8 element vectorized kernel needs 8 increments - creating uint4 for increments 0, 1, 2, 3 here, and adding (float4)4 later to get 4, 5, 6, 7 incremented srcLocs
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = static_cast<float4>(id_x) + increment_f8.f4[0];
    locDst_f8x.f4[1] = static_cast<float4>(id_x) + increment_f8.f4[1];
    locDst_f8y.f4[0] = locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = static_cast<float4>(srcRoiPtr_i4->x) + locDst_f8x.f4[0] + widthIncrement_f8.f4[0] - static_cast<float4>(bound);
    locSrc_f16->f8[0].f4[1] = static_cast<float4>(srcRoiPtr_i4->x) + locDst_f8x.f4[1] + widthIncrement_f8.f4[1] - static_cast<float4>(bound);
    locSrc_f16->f8[1].f4[0] = static_cast<float4>(srcRoiPtr_i4->y) + locDst_f8y.f4[0] + heightIncrement_f8.f4[0] - static_cast<float4>(bound);
    locSrc_f16->f8[1].f4[1] = static_cast<float4>(srcRoiPtr_i4->y) + locDst_f8y.f4[1] + heightIncrement_f8.f4[1] - static_cast<float4>(bound);

    // Apply boundary checks and adjustments
    for(int i = 0; i < 8; ++i)
    {
        locSrc_f16->f1[i] = fmaxf(fminf(floorf(locSrc_f16->f1[i]), static_cast<float>(srcRoiPtr_i4->z - 1)), 0.0f);
        locSrc_f16->f1[i + 8] = fmaxf(fminf(floorf(locSrc_f16->f1[i + 8]), static_cast<float>(srcRoiPtr_i4->w - bound)), 0.0f);
    }
}

template <typename T>
__global__ void jitter_pkd_tensor(T *srcPtr,
                                  uint2 srcStridesNH,
                                  T *dstPtr,
                                  uint2 dstStridesNH,
                                  uint *kernelsize,
                                  RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                  uint *xorwowSeedStream,
                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    uint kernelSize = kernelsize[id_z];
    uint bound = (kernelSize - 1) / 2;

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    jitter_roi_and_srclocs_hip_compute(&srcRoi_i4, &xorwowState, kernelSize, bound, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void jitter_pln_tensor(T *srcPtr,
                                  uint3 srcStridesNCH,
                                  T *dstPtr,
                                  uint3 dstStridesNCH,
                                  int channelsDst,
                                  uint *kernelsize,
                                  RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                  uint *xorwowSeedStream,
                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    uint kernelSize = kernelsize[id_z];
    uint bound = (kernelSize - 1) / 2;

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    jitter_roi_and_srclocs_hip_compute(&srcRoi_i4, &xorwowState, kernelSize, bound, id_x, id_y, &locSrc_f16);

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
__global__ void jitter_pkd3_pln3_tensor(T *srcPtr,
                                        uint2 srcStridesNH,
                                        T *dstPtr,
                                        uint3 dstStridesNCH,
                                        uint *kernelsize,
                                        RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                        uint *xorwowSeedStream,
                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    uint kernelSize = kernelsize[id_z];
    uint bound = (kernelSize - 1) / 2;

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    jitter_roi_and_srclocs_hip_compute(&srcRoi_i4, &xorwowState, kernelSize, bound, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void jitter_pln3_pkd3_tensor(T *srcPtr,
                                        uint3 srcStridesNCH,
                                        T *dstPtr,
                                        uint2 dstStridesNH,
                                        uint *kernelsize,
                                        RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                        uint *xorwowSeedStream,
                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    uint kernelSize = kernelsize[id_z];
    uint bound = (kernelSize - 1) / 2;

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    jitter_roi_and_srclocs_hip_compute(&srcRoi_i4, &xorwowState, kernelSize, bound, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_jitter_tensor(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 uint *kernelSizeTensor,
                                 RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    Rpp32u *xorwowSeedStream;
    xorwowSeedStream = (Rpp32u *)&xorwowInitialStatePtr[1];
    CHECK_RETURN_STATUS(hipMemcpyAsync(xorwowSeedStream, rngSeedStream4050, SEED_STREAM_MAX_SIZE * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream()));

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(jitter_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           kernelSizeTensor,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(jitter_pln_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           kernelSizeTensor,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(jitter_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               kernelSizeTensor,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(jitter_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               kernelSizeTensor,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_jitter_tensor<Rpp8u>(Rpp8u*,
                                                 RpptDescPtr,
                                                 Rpp8u*,
                                                 RpptDescPtr,
                                                 uint*,
                                                 RpptXorwowStateBoxMuller*,
                                                 RpptROIPtr,
                                                 RpptRoiType,
                                                 rpp::Handle&);

template RppStatus hip_exec_jitter_tensor<half>(half*,
                                                RpptDescPtr,
                                                half*,
                                                RpptDescPtr,
                                                uint*,
                                                RpptXorwowStateBoxMuller*,
                                                RpptROIPtr,
                                                RpptRoiType,
                                                rpp::Handle&);

template RppStatus hip_exec_jitter_tensor<Rpp32f>(Rpp32f*,
                                                  RpptDescPtr,
                                                  Rpp32f*,
                                                  RpptDescPtr,
                                                  uint*,
                                                  RpptXorwowStateBoxMuller*,
                                                  RpptROIPtr,
                                                  RpptRoiType,
                                                  rpp::Handle&);

template RppStatus hip_exec_jitter_tensor<Rpp8s>(Rpp8s*,
                                                 RpptDescPtr,
                                                 Rpp8s*,
                                                 RpptDescPtr,
                                                 uint*,
                                                 RpptXorwowStateBoxMuller*,
                                                 RpptROIPtr,
                                                 RpptRoiType,
                                                 rpp::Handle&);
