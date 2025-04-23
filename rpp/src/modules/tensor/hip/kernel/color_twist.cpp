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

__device__ void color_twist_1RGB_hip_compute(float *pixelR, float *pixelG, float *pixelB, float4 *colorTwistParams_f4)
{
    // RGB to HSV

    float hue, sat, v, add;
    float rf, gf, bf, cmax, cmin, delta;
    rf = *pixelR;
    gf = *pixelG;
    bf = *pixelB;
    cmax = fmaxf(fmaxf(rf, gf), bf);
    cmin = fminf(fminf(rf, gf), bf);
    delta = cmax - cmin;
    hue = 0.0f;
    sat = 0.0f;
    add = 0.0f;
    if ((delta != 0) && (cmax != 0))
    {
        sat = delta / cmax;
        if (cmax == rf)
        {
            hue = gf - bf;
            add = 0.0f;
        }
        else if (cmax == gf)
        {
            hue = bf - rf;
            add = 2.0f;
        }
        else
        {
            hue = rf - gf;
            add = 4.0f;
        }
        hue /= delta;
    }
    v = cmax;

    // Modify Hue and Saturation

    hue += colorTwistParams_f4->z + add;
    if (hue >= 6.0f) hue -= 6.0f;
    if (hue < 0) hue += 6.0f;
    sat *= colorTwistParams_f4->w;
    sat = fmaxf(0.0f, fminf(1.0f, sat));

    // HSV to RGB with brightness/contrast adjustment

    int hueIntegerPart = (int) hue;
    float hueFractionPart = hue - hueIntegerPart;
    float vsat = v * sat;
    float vsatf = vsat * hueFractionPart;
    float p = v - vsat;
    float q = v - vsatf;
    float t = v - vsat + vsatf;
    switch (hueIntegerPart)
    {
        case 0: rf = v; gf = t; bf = p; break;
        case 1: rf = q; gf = v; bf = p; break;
        case 2: rf = p; gf = v; bf = t; break;
        case 3: rf = p; gf = q; bf = v; break;
        case 4: rf = t; gf = p; bf = v; break;
        case 5: rf = v; gf = p; bf = q; break;
    }
    *pixelR = fmaf(rf, colorTwistParams_f4->x, colorTwistParams_f4->y);
    *pixelG = fmaf(gf, colorTwistParams_f4->x, colorTwistParams_f4->y);
    *pixelB = fmaf(bf, colorTwistParams_f4->x, colorTwistParams_f4->y);
}

__device__ void color_twist_8RGB_hip_compute(d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    color_twist_1RGB_hip_compute(&(pix_f24->f1[ 0]), &(pix_f24->f1[ 8]), &(pix_f24->f1[16]), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->f1[ 1]), &(pix_f24->f1[ 9]), &(pix_f24->f1[17]), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->f1[ 2]), &(pix_f24->f1[10]), &(pix_f24->f1[18]), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->f1[ 3]), &(pix_f24->f1[11]), &(pix_f24->f1[19]), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->f1[ 4]), &(pix_f24->f1[12]), &(pix_f24->f1[20]), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->f1[ 5]), &(pix_f24->f1[13]), &(pix_f24->f1[21]), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->f1[ 6]), &(pix_f24->f1[14]), &(pix_f24->f1[22]), colorTwistParams_f4);
    color_twist_1RGB_hip_compute(&(pix_f24->f1[ 7]), &(pix_f24->f1[15]), &(pix_f24->f1[23]), colorTwistParams_f4);
}

__device__ void color_twist_hip_compute(uchar *srcPtr, d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    float4 normalizer_f4 = (float4) ONE_OVER_255;
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    colorTwistParams_f4->x = colorTwistParams_f4->x * 255.0f;
    colorTwistParams_f4->z = (((int)colorTwistParams_f4->z) % 360) * SIX_OVER_360;
    color_twist_8RGB_hip_compute(pix_f24, colorTwistParams_f4);
    rpp_hip_pixel_check_0to255(pix_f24);
}
__device__ void color_twist_hip_compute(float *srcPtr, d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    colorTwistParams_f4->y = colorTwistParams_f4->y * ONE_OVER_255;
    colorTwistParams_f4->z = (((int)colorTwistParams_f4->z) % 360) * SIX_OVER_360;
    color_twist_8RGB_hip_compute(pix_f24, colorTwistParams_f4);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ void color_twist_hip_compute(half *srcPtr, d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    colorTwistParams_f4->y = colorTwistParams_f4->y * ONE_OVER_255;
    colorTwistParams_f4->z = (((int)colorTwistParams_f4->z) % 360) * SIX_OVER_360;
    color_twist_8RGB_hip_compute(pix_f24, colorTwistParams_f4);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ void color_twist_hip_compute(schar *srcPtr, d_float24 *pix_f24, float4 *colorTwistParams_f4)
{
    float4 i8Offset_f4 = (float4) 128.0f;
    float4 normalizer_f4 = (float4) ONE_OVER_255;
    rpp_hip_math_add24_const(pix_f24, pix_f24, i8Offset_f4);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    colorTwistParams_f4->x = colorTwistParams_f4->x * 255.0f;
    colorTwistParams_f4->z = (((int)colorTwistParams_f4->z) % 360) * SIX_OVER_360;
    color_twist_8RGB_hip_compute(pix_f24, colorTwistParams_f4);
    rpp_hip_pixel_check_0to255(pix_f24);
    rpp_hip_math_subtract24_const(pix_f24, pix_f24, i8Offset_f4);
}

template <typename T>
__global__ void color_twist_pkd_hip_tensor(T *srcPtr,
                                       uint2 srcStridesNH,
                                       T *dstPtr,
                                       uint2 dstStridesNH,
                                       float *brightnessTensor,
                                       float *contrastTensor,
                                       float *hueTensor,
                                       float *saturationTensor,
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

    float4 colorTwistParams_f4 = make_float4(brightnessTensor[id_z], contrastTensor[id_z], hueTensor[id_z], saturationTensor[id_z]);

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    color_twist_hip_compute(srcPtr, &pix_f24, &colorTwistParams_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void color_twist_pln_hip_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      float *brightnessTensor,
                                      float *contrastTensor,
                                      float *hueTensor,
                                      float *saturationTensor,
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

    float4 colorTwistParams_f4 = make_float4(brightnessTensor[id_z], contrastTensor[id_z], hueTensor[id_z], saturationTensor[id_z]);

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    color_twist_hip_compute(srcPtr, &pix_f24, &colorTwistParams_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void color_twist_pkd3_pln3_hip_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint3 dstStridesNCH,
                                            float *brightnessTensor,
                                            float *contrastTensor,
                                            float *hueTensor,
                                            float *saturationTensor,
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

    float4 colorTwistParams_f4 = make_float4(brightnessTensor[id_z], contrastTensor[id_z], hueTensor[id_z], saturationTensor[id_z]);

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    color_twist_hip_compute(srcPtr, &pix_f24, &colorTwistParams_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void color_twist_pln3_pkd3_hip_tensor(T *srcPtr,
                                            uint3 srcStridesNCH,
                                            T *dstPtr,
                                            uint2 dstStridesNH,
                                            float *brightnessTensor,
                                            float *contrastTensor,
                                            float *hueTensor,
                                            float *saturationTensor,
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

    float4 colorTwistParams_f4 = make_float4(brightnessTensor[id_z], contrastTensor[id_z], hueTensor[id_z], saturationTensor[id_z]);

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    color_twist_hip_compute(srcPtr, &pix_f24, &colorTwistParams_f4);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_color_twist_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
            hipLaunchKernelGGL(color_twist_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
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
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(color_twist_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(color_twist_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
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
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(color_twist_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
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
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_color_twist_tensor<Rpp8u>(Rpp8u*,
                                                      RpptDescPtr,
                                                      Rpp8u*,
                                                      RpptDescPtr,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      rpp::Handle&);

template RppStatus hip_exec_color_twist_tensor<half>(half*,
                                                     RpptDescPtr,
                                                     half*,
                                                     RpptDescPtr,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_color_twist_tensor<Rpp32f>(Rpp32f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptDescPtr,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_color_twist_tensor<Rpp8s>(Rpp8s*,
                                                      RpptDescPtr,
                                                      Rpp8s*,
                                                      RpptDescPtr,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      rpp::Handle&);
