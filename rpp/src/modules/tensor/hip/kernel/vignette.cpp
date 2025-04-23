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

__device__ void vignette_gaussian_hip_compute(float &multiplier, int2 &halfDimsWH_i2, int2 &idXY_i2, d_float8 *gaussianValue_f8)
{
    float rowLocComponent;
    rowLocComponent = idXY_i2.y - halfDimsWH_i2.y;
    rowLocComponent *= (rowLocComponent);

    float4 rowLocComponent_f4 = static_cast<float4>(rowLocComponent);
    float4 multiplier_f4 = static_cast<float4>(multiplier);

    d_float8 colLocComponent_f8;
    colLocComponent_f8.f4[0] = make_float4(idXY_i2.x, idXY_i2.x + 1, idXY_i2.x + 2, idXY_i2.x + 3);
    colLocComponent_f8.f4[1] = colLocComponent_f8.f4[0] + static_cast<float4>(4);
    colLocComponent_f8.f4[0] -= static_cast<float4>(halfDimsWH_i2.x);
    colLocComponent_f8.f4[1] -= static_cast<float4>(halfDimsWH_i2.x);
    colLocComponent_f8.f4[0] = (colLocComponent_f8.f4[0] * colLocComponent_f8.f4[0]) + rowLocComponent_f4;
    colLocComponent_f8.f4[1] = (colLocComponent_f8.f4[1] * colLocComponent_f8.f4[1]) + rowLocComponent_f4;
    colLocComponent_f8.f4[0] = colLocComponent_f8.f4[0] * multiplier_f4;
    colLocComponent_f8.f4[1] = colLocComponent_f8.f4[1] * multiplier_f4;

    gaussianValue_f8->f4[0] = make_float4(expf(colLocComponent_f8.f4[0].x), expf(colLocComponent_f8.f4[0].y), expf(colLocComponent_f8.f4[0].z), expf(colLocComponent_f8.f4[0].w));
    gaussianValue_f8->f4[1] = make_float4(expf(colLocComponent_f8.f4[1].x), expf(colLocComponent_f8.f4[1].y), expf(colLocComponent_f8.f4[1].z), expf(colLocComponent_f8.f4[1].w));
}

__device__ void vignette_8_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *gaussianValue_f8)
{
    dst_f8->f4[0] = src_f8->f4[0] * gaussianValue_f8->f4[0];
    dst_f8->f4[1] = src_f8->f4[1] * gaussianValue_f8->f4[1];
    rpp_hip_math_nearbyintf8(dst_f8, dst_f8);
    rpp_hip_pixel_check_0to255(dst_f8);
}

__device__ void vignette_8_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *gaussianValue_f8)
{
    dst_f8->f4[0] = src_f8->f4[0] * gaussianValue_f8->f4[0];
    dst_f8->f4[1] = src_f8->f4[1] * gaussianValue_f8->f4[1];
    rpp_hip_pixel_check_0to1(dst_f8);
}

__device__ void vignette_8_hip_compute(signed char *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *gaussianValue_f8)
{
    float4 i8Offset_f4 = static_cast<float4>(128.0f);
    rpp_hip_math_add8_const(src_f8, src_f8, i8Offset_f4);
    dst_f8->f4[0] = src_f8->f4[0] * gaussianValue_f8->f4[0];
    dst_f8->f4[1] = src_f8->f4[1] * gaussianValue_f8->f4[1];
    rpp_hip_math_nearbyintf8(dst_f8, dst_f8);
    rpp_hip_pixel_check_0to255(dst_f8);
    rpp_hip_math_subtract8_const(dst_f8, dst_f8, i8Offset_f4);
}

__device__ void vignette_8_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, d_float8 *gaussianValue_f8)
{
    dst_f8->f4[0] = src_f8->f4[0] * gaussianValue_f8->f4[0];
    dst_f8->f4[1] = src_f8->f4[1] * gaussianValue_f8->f4[1];
    rpp_hip_pixel_check_0to1(dst_f8);
}

template <typename T>
__device__ void vignette_24_hip_compute(T *srcPtr, d_float24 *src_f24, d_float24 *dst_f24, d_float8 *gaussianValue_f8)
{
    vignette_8_hip_compute(srcPtr, &(src_f24->f8[0]), &(dst_f24->f8[0]), gaussianValue_f8);
    vignette_8_hip_compute(srcPtr, &(src_f24->f8[1]), &(dst_f24->f8[1]), gaussianValue_f8);
    vignette_8_hip_compute(srcPtr, &(src_f24->f8[2]), &(dst_f24->f8[2]), gaussianValue_f8);
}

template <typename T>
__global__ void vignette_pkd_tensor(T *srcPtr,
                                    uint2 srcStridesNH,
                                    T *dstPtr,
                                    uint2 dstStridesNH,
                                    float *vignetteIntensity,
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

    float intensity = vignetteIntensity[id_z];
    int2 halfDimsWH_i2 = make_int2(roiTensorPtrSrc[id_z].xywhROI.roiWidth >> 1, roiTensorPtrSrc[id_z].xywhROI.roiHeight >> 1);
    int2 idXY_i2 = make_int2(id_x, id_y);
    float radius = fmaxf(halfDimsWH_i2.x, halfDimsWH_i2.y);
    float multiplier = -(0.25f * intensity) / (radius * radius);

    d_float24 src_f24, dst_f24;
    d_float8 gaussianValue_f8;
    vignette_gaussian_hip_compute(multiplier, halfDimsWH_i2, idXY_i2, &gaussianValue_f8);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    vignette_24_hip_compute(srcPtr, &src_f24, &dst_f24, &gaussianValue_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void vignette_pln_tensor(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    T *dstPtr,
                                    uint3 dstStridesNCH,
                                    int channelsDst,
                                    float *vignetteIntensity,
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

    float intensity = vignetteIntensity[id_z];
    int2 halfDimsWH_i2 = make_int2(roiTensorPtrSrc[id_z].xywhROI.roiWidth >> 1, roiTensorPtrSrc[id_z].xywhROI.roiHeight >> 1);
    int2 idXY_i2 = make_int2(id_x, id_y);
    float radius = fmaxf(halfDimsWH_i2.x, halfDimsWH_i2.y);
    float multiplier = -(0.25f * intensity) / (radius * radius);

    d_float8 src_f8, dst_f8;
    d_float8 gaussianValue_f8;
    vignette_gaussian_hip_compute(multiplier, halfDimsWH_i2, idXY_i2, &gaussianValue_f8);

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    vignette_8_hip_compute(srcPtr, &src_f8, &dst_f8, &gaussianValue_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        vignette_8_hip_compute(srcPtr, &src_f8, &dst_f8, &gaussianValue_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        vignette_8_hip_compute(srcPtr, &src_f8, &dst_f8, &gaussianValue_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void vignette_pkd3_pln3_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          float *vignetteIntensity,
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

    float intensity = vignetteIntensity[id_z];
    int2 halfDimsWH_i2 = make_int2(roiTensorPtrSrc[id_z].xywhROI.roiWidth >> 1, roiTensorPtrSrc[id_z].xywhROI.roiHeight >> 1);
    int2 idXY_i2 = make_int2(id_x, id_y);
    float radius = fmaxf(halfDimsWH_i2.x, halfDimsWH_i2.y);
    float multiplier = -(0.25f * intensity) / (radius * radius);

    d_float24 src_f24, dst_f24;
    d_float8 gaussianValue_f8;
    vignette_gaussian_hip_compute(multiplier, halfDimsWH_i2, idXY_i2, &gaussianValue_f8);

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    vignette_24_hip_compute(srcPtr, &src_f24, &dst_f24, &gaussianValue_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void vignette_pln3_pkd3_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float *vignetteIntensity,
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

    float intensity = vignetteIntensity[id_z];
    int2 halfDimsWH_i2 = make_int2(roiTensorPtrSrc[id_z].xywhROI.roiWidth >> 1, roiTensorPtrSrc[id_z].xywhROI.roiHeight >> 1);
    int2 idXY_i2 = make_int2(id_x, id_y);
    float radius = fmaxf(halfDimsWH_i2.x, halfDimsWH_i2.y);
    float multiplier = -(0.25f * intensity) / (radius * radius);

    d_float24 src_f24, dst_f24;
    d_float8 gaussianValue_f8;
    vignette_gaussian_hip_compute(multiplier, halfDimsWH_i2, idXY_i2, &gaussianValue_f8);

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    vignette_24_hip_compute(srcPtr, &src_f24, &dst_f24, &gaussianValue_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_vignette_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   Rpp32f *vignetteIntensityTensor,
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
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(vignette_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           vignetteIntensityTensor,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(vignette_pln_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           vignetteIntensityTensor,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(vignette_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               vignetteIntensityTensor,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(vignette_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               vignetteIntensityTensor,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_vignette_tensor<Rpp8u>(Rpp8u*,
                                                   RpptDescPtr,
                                                   Rpp8u*,
                                                   RpptDescPtr,
                                                   RpptROIPtr,
                                                   Rpp32f*,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus hip_exec_vignette_tensor<half>(half*,
                                                   RpptDescPtr,
                                                   half*,
                                                   RpptDescPtr,
                                                   RpptROIPtr,
                                                   Rpp32f*,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus hip_exec_vignette_tensor<Rpp32f>(Rpp32f*,
                                                   RpptDescPtr,
                                                   Rpp32f*,
                                                   RpptDescPtr,
                                                   RpptROIPtr,
                                                   Rpp32f*,
                                                   RpptRoiType,
                                                   rpp::Handle&);

template RppStatus hip_exec_vignette_tensor<Rpp8s>(Rpp8s*,
                                                   RpptDescPtr,
                                                   Rpp8s*,
                                                   RpptDescPtr,
                                                   RpptROIPtr,
                                                   Rpp32f*,
                                                   RpptRoiType,
                                                   rpp::Handle&);
