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
#include "rng_seed_stream.hpp"

__device__ void shot_noise_1_hip_compute(float *pix, RpptXorwowStateBoxMuller *xorwowState)
{
    if (*pix < 5)
        *pix = rpp_hip_rng_1_inverse_transform_sampling_f32(*pix, xorwowState);
    else
        *pix = (1 / rpp_hip_math_inverse_sqrt1(*pix)) * rpp_hip_rng_1_gaussian_f32(xorwowState) + *pix;
}

__device__ void shot_noise_8_hip_compute(d_float8 *pix_f8, RpptXorwowStateBoxMuller *xorwowState)
{
    shot_noise_1_hip_compute(&pix_f8->f1[0], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[1], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[2], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[3], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[4], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[5], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[6], xorwowState);
    shot_noise_1_hip_compute(&pix_f8->f1[7], xorwowState);
}

__device__ void shot_noise_24_hip_compute(d_float24 *pix_f24, RpptXorwowStateBoxMuller *xorwowState)
{
    shot_noise_8_hip_compute(&pix_f24->f8[0], xorwowState);
    shot_noise_8_hip_compute(&pix_f24->f8[1], xorwowState);
    shot_noise_8_hip_compute(&pix_f24->f8[2], xorwowState);
}

__device__ void shot_noise_8_adjusted_input_hip_compute(uchar *srcPtr, d_float8 *pix_f8) {}
__device__ void shot_noise_8_adjusted_input_hip_compute(float *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)255.0f); }
__device__ void shot_noise_8_adjusted_input_hip_compute(schar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_add8_const(pix_f8, pix_f8, (float4)128.0f); }
__device__ void shot_noise_8_adjusted_input_hip_compute(half *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)255.0f); }

__device__ void shot_noise_24_adjusted_input_hip_compute(uchar *srcPtr, d_float24 *pix_f24) {}
__device__ void shot_noise_24_adjusted_input_hip_compute(float *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)255.0f); }
__device__ void shot_noise_24_adjusted_input_hip_compute(schar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_add24_const(pix_f24, pix_f24, (float4)128.0f); }
__device__ void shot_noise_24_adjusted_input_hip_compute(half *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)255.0f); }

__device__ void shot_noise_8_adjusted_output_hip_compute(uchar *srcPtr, d_float8 *pix_f8) {}
__device__ void shot_noise_8_adjusted_output_hip_compute(float *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)ONE_OVER_255); }
__device__ void shot_noise_8_adjusted_output_hip_compute(schar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_subtract8_const(pix_f8, pix_f8, (float4)128.0f); }
__device__ void shot_noise_8_adjusted_output_hip_compute(half *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)ONE_OVER_255); }

__device__ void shot_noise_24_adjusted_output_hip_compute(uchar *srcPtr, d_float24 *pix_f24) {}
__device__ void shot_noise_24_adjusted_output_hip_compute(float *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)ONE_OVER_255); }
__device__ void shot_noise_24_adjusted_output_hip_compute(schar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_subtract24_const(pix_f24, pix_f24, (float4)128.0f); }
__device__ void shot_noise_24_adjusted_output_hip_compute(half *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)ONE_OVER_255); }

template <typename T>
__global__ void shot_noise_pkd_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float *shotNoiseFactorTensor,
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

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float shotNoiseFactor = shotNoiseFactorTensor[id_z];

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    if (shotNoiseFactor != 0.0f)
    {
        float shotNoiseFactorInv = 1 / shotNoiseFactor;

        RpptXorwowStateBoxMuller xorwowState;
        uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
        xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

        shot_noise_24_adjusted_input_hip_compute(srcPtr, &pix_f24);
        rpp_hip_math_multiply24_const(&pix_f24, &pix_f24, (float4)shotNoiseFactorInv);
        shot_noise_24_hip_compute(&pix_f24, &xorwowState);
        rpp_hip_math_multiply24_const(&pix_f24, &pix_f24, (float4)shotNoiseFactor);
        rpp_hip_pixel_check_0to255(&pix_f24);
        shot_noise_24_adjusted_output_hip_compute(srcPtr, &pix_f24);
    }
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void shot_noise_pln_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          int channelsDst,
                                          float *shotNoiseFactorTensor,
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

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float shotNoiseFactor = shotNoiseFactorTensor[id_z];

    d_float8 pix_f8;

    if (shotNoiseFactor != 0.0f)
    {
        float shotNoiseFactorInv = 1 / shotNoiseFactor;

        RpptXorwowStateBoxMuller xorwowState;
        uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
        xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        shot_noise_8_adjusted_input_hip_compute(srcPtr, &pix_f8);
        rpp_hip_math_multiply8_const(&pix_f8, &pix_f8, (float4)shotNoiseFactorInv);
        shot_noise_8_hip_compute(&pix_f8, &xorwowState);
        rpp_hip_math_multiply8_const(&pix_f8, &pix_f8, (float4)shotNoiseFactor);
        rpp_hip_pixel_check_0to255(&pix_f8);
        shot_noise_8_adjusted_output_hip_compute(srcPtr, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        if (channelsDst == 3)
        {
            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            shot_noise_8_adjusted_input_hip_compute(srcPtr, &pix_f8);
            rpp_hip_math_multiply8_const(&pix_f8, &pix_f8, (float4)shotNoiseFactorInv);
            shot_noise_8_hip_compute(&pix_f8, &xorwowState);
            rpp_hip_math_multiply8_const(&pix_f8, &pix_f8, (float4)shotNoiseFactor);
            rpp_hip_pixel_check_0to255(&pix_f8);
            shot_noise_8_adjusted_output_hip_compute(srcPtr, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            shot_noise_8_adjusted_input_hip_compute(srcPtr, &pix_f8);
            rpp_hip_math_multiply8_const(&pix_f8, &pix_f8, (float4)shotNoiseFactorInv);
            shot_noise_8_hip_compute(&pix_f8, &xorwowState);
            rpp_hip_math_multiply8_const(&pix_f8, &pix_f8, (float4)shotNoiseFactor);
            rpp_hip_pixel_check_0to255(&pix_f8);
            shot_noise_8_adjusted_output_hip_compute(srcPtr, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
    }
    else
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        if (channelsDst == 3)
        {
            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
    }
}

template <typename T>
__global__ void shot_noise_pkd3_pln3_hip_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
                                                float *shotNoiseFactorTensor,
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

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float shotNoiseFactor = shotNoiseFactorTensor[id_z];

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    if (shotNoiseFactor != 0.0f)
    {
        float shotNoiseFactorInv = 1 / shotNoiseFactor;

        RpptXorwowStateBoxMuller xorwowState;
        uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
        xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

        shot_noise_24_adjusted_input_hip_compute(srcPtr, &pix_f24);
        rpp_hip_math_multiply24_const(&pix_f24, &pix_f24, (float4)shotNoiseFactorInv);
        shot_noise_24_hip_compute(&pix_f24, &xorwowState);
        rpp_hip_math_multiply24_const(&pix_f24, &pix_f24, (float4)shotNoiseFactor);
        rpp_hip_pixel_check_0to255(&pix_f24);
        shot_noise_24_adjusted_output_hip_compute(srcPtr, &pix_f24);
    }
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void shot_noise_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                float *shotNoiseFactorTensor,
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

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float shotNoiseFactor = shotNoiseFactorTensor[id_z];

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    if (shotNoiseFactor != 0.0f)
    {
        float shotNoiseFactorInv = 1 / shotNoiseFactor;

        RpptXorwowStateBoxMuller xorwowState;
        uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
        xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

        shot_noise_24_adjusted_input_hip_compute(srcPtr, &pix_f24);
        rpp_hip_math_multiply24_const(&pix_f24, &pix_f24, (float4)shotNoiseFactorInv);
        shot_noise_24_hip_compute(&pix_f24, &xorwowState);
        rpp_hip_math_multiply24_const(&pix_f24, &pix_f24, (float4)shotNoiseFactor);
        rpp_hip_pixel_check_0to255(&pix_f24);
        shot_noise_24_adjusted_output_hip_compute(srcPtr, &pix_f24);
    }
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_shot_noise_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *xorwowSeedStream;
    xorwowSeedStream = (Rpp32u *)&xorwowInitialStatePtr[1];
    CHECK_RETURN_STATUS(hipMemcpy(xorwowSeedStream, rngSeedStream4050, SEED_STREAM_MAX_SIZE * sizeof(Rpp32u), hipMemcpyHostToDevice));

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(shot_noise_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(shot_noise_pln_hip_tensor,
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
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(shot_noise_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(shot_noise_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_shot_noise_tensor<Rpp8u>(Rpp8u*,
                                                     RpptDescPtr,
                                                     Rpp8u*,
                                                     RpptDescPtr,
                                                     RpptXorwowStateBoxMuller*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_shot_noise_tensor<half>(half*,
                                                    RpptDescPtr,
                                                    half*,
                                                    RpptDescPtr,
                                                    RpptXorwowStateBoxMuller*,
                                                    RpptROIPtr,
                                                    RpptRoiType,
                                                    rpp::Handle&);

template RppStatus hip_exec_shot_noise_tensor<Rpp32f>(Rpp32f*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptDescPtr,
                                                      RpptXorwowStateBoxMuller*,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      rpp::Handle&);

template RppStatus hip_exec_shot_noise_tensor<Rpp8s>(Rpp8s*,
                                                     RpptDescPtr,
                                                     Rpp8s*,
                                                     RpptDescPtr,
                                                     RpptXorwowStateBoxMuller*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);
