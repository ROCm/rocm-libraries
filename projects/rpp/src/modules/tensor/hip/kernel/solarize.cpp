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

__device__ void solarize_hip_rgb_compute(d_float24* pix_f24, float& thresholdParam, float& maxVal) {
    for (int i = 0; i < 8; i++) {
        pix_f24->f8[0].f1[i] = (pix_f24->f8[0].f1[i] >= thresholdParam)
                                   ? (maxVal - pix_f24->f8[0].f1[i])
                                   : pix_f24->f8[0].f1[i];
        pix_f24->f8[1].f1[i] = (pix_f24->f8[1].f1[i] >= thresholdParam)
                                   ? (maxVal - pix_f24->f8[1].f1[i])
                                   : pix_f24->f8[1].f1[i];
        pix_f24->f8[2].f1[i] = (pix_f24->f8[2].f1[i] >= thresholdParam)
                                   ? (maxVal - pix_f24->f8[2].f1[i])
                                   : pix_f24->f8[2].f1[i];
    }
}

__device__ void solarize_hip_greyscale_compute(d_float8* pix_f8, float& thresholdParam,
                                               float& maxVal) {
    for (int i = 0; i < 8; i++)
        pix_f8->f1[i] =
            (pix_f8->f1[i] >= thresholdParam) ? (maxVal - pix_f8->f1[i]) : pix_f8->f1[i];
}

template <typename T>
__global__ void solarize_pkd_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                    float* thresholdTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                  ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float thresholdParam = thresholdTensor[id_z];
    float maxVal = 1.0f;  // Default for float

    if constexpr (std::is_same<T, Rpp8u>::value) {
        thresholdParam =
            roundf(thresholdParam * 255.0f);  // Scale threshold [0, 1] to full uint8 range [0, 255]
        maxVal = 255.0f;                      // Maximum value for uint8
    } else if constexpr (std::is_same<T, Rpp8s>::value) {
        thresholdParam = roundf(thresholdParam * 255.0f) -
                         128.0f;  // Scale threshold [0, 1] to signed int8 range [-128, 127]
        maxVal = -1.0f;           // Use -1 as "max" for solarize operation on signed int8
    }

    d_float24 pix_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    solarize_hip_rgb_compute(&pix_f24, thresholdParam, maxVal);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void solarize_pln3_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr, uint3 dstStridesNCH,
                                     float* thresholdTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float thresholdParam = thresholdTensor[id_z];
    float maxVal = 1.0f;  // Default for float

    if constexpr (std::is_same<T, Rpp8u>::value) {
        thresholdParam =
            roundf(thresholdParam * 255.0f);  // Scale threshold [0,1] to full uint8 range [0,255]
        maxVal = 255.0f;                      // Maximum value for uint8
    } else if constexpr (std::is_same<T, Rpp8s>::value) {
        thresholdParam = roundf(thresholdParam * 255.0f) -
                         128.0f;  // Scale threshold [0,1] to signed int8 range [-128,127]
        maxVal = -1.0f;           // Use -1 as "max" for solarize operation on signed int8
    }

    d_float24 pix_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    solarize_hip_rgb_compute(&pix_f24, thresholdParam, maxVal);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void solarize_pln1_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                     float* thresholdTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    float thresholdParam = thresholdTensor[id_z];
    float maxVal = 1.0f;  // Default for float

    if constexpr (std::is_same<T, Rpp8u>::value) {
        thresholdParam =
            roundf(thresholdParam * 255.0f);  // Scale threshold [0,1] to full uint8 range [0,255]
        maxVal = 255.0f;                      // Maximum value for uint8
    } else if constexpr (std::is_same<T, Rpp8s>::value) {
        thresholdParam = roundf(thresholdParam * 255.0f) -
                         128.0f;  // Scale threshold [0,1] to signed int8 range [-128,127]
        maxVal = -1.0f;           // Use -1 as "max" for solarize operation on signed int8
    }

    d_float8 pix_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    solarize_hip_greyscale_compute(&pix_f8, thresholdParam, maxVal);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
}

template <typename T>
__global__ void solarize_pkd3_pln3_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                          uint3 dstStridesNCH, float* thresholdTensor,
                                          RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) +
                  ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float thresholdParam = thresholdTensor[id_z];
    float maxVal = 1.0f;  // Default for float

    if constexpr (std::is_same<T, Rpp8u>::value) {
        thresholdParam =
            roundf(thresholdParam * 255.0f);  // Scale threshold [0,1] to full uint8 range [0,255]
        maxVal = 255.0f;                      // Maximum value for uint8
    } else if constexpr (std::is_same<T, Rpp8s>::value) {
        thresholdParam = roundf(thresholdParam * 255.0f) -
                         128.0f;  // Scale threshold [0,1] to signed int8 range [-128,127]
        maxVal = -1.0f;           // Use -1 as "max" for solarize operation on signed int8
    }

    d_float24 pix_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    solarize_hip_rgb_compute(&pix_f24, thresholdParam, maxVal);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void solarize_pln3_pkd3_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                          uint2 dstStridesNH, float* thresholdTensor,
                                          RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float thresholdParam = thresholdTensor[id_z];
    float maxVal = 1.0f;  // Default for float

    if constexpr (std::is_same<T, Rpp8u>::value) {
        thresholdParam =
            roundf(thresholdParam * 255.0f);  // Scale threshold [0,1] to full uint8 range [0,255]
        maxVal = 255.0f;                      // Maximum value for uint8
    } else if constexpr (std::is_same<T, Rpp8s>::value) {
        thresholdParam = roundf(thresholdParam * 255.0f) -
                         128.0f;  // Scale threshold [0,1] to signed int8 range [-128,127]
        maxVal = -1.0f;           // Use -1 as "max" for solarize operation on signed int8
    }

    d_float24 pix_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    solarize_hip_rgb_compute(&pix_f24, thresholdParam, maxVal);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_solarize_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                   RpptDescPtr dstDescPtr, Rpp32f* thresholdTensor,
                                   RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                   rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    Rpp32s globalThreads_x = (dstDescPtr->w + 7) >> 3;
    Rpp32s globalThreads_y = dstDescPtr->h;
    Rpp32s globalThreads_z = dstDescPtr->n;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        hipLaunchKernelGGL(
            solarize_pkd_tensor,
            dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                 ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride), thresholdTensor,
            roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        if (srcDescPtr->c == 3) {
            hipLaunchKernelGGL(solarize_pln3_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               thresholdTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else {
            hipLaunchKernelGGL(
                solarize_pln1_tensor,
                dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                     ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                thresholdTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            hipLaunchKernelGGL(solarize_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               thresholdTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            hipLaunchKernelGGL(solarize_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               thresholdTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_solarize_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                   Rpp32f*, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_solarize_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr, Rpp32f*,
                                                  RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_solarize_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                    Rpp32f*, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_solarize_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                   Rpp32f*, RpptROIPtr, RpptRoiType, rpp::Handle&);
