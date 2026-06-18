/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

#include "api_helpers.hpp"
#include "hip_tensor_executors.hpp"

// -------------------- Set 0 - grid_dropout main kernels --------------------

template <typename T>
__global__ void grid_dropout_pkd_hip_tensor(T* dstPtr, uint3 dstStridesNHW,
                                            RpptRoiLtrb* anchorBoxInfoTensor,
                                            uint boxesInEachImage) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int batch_idx = id_z / boxesInEachImage;

    if ((id_y >= (anchorBoxInfoTensor[id_z].rb.y - anchorBoxInfoTensor[id_z].lt.y + 1)) ||
        (id_x >= (anchorBoxInfoTensor[id_z].rb.x - anchorBoxInfoTensor[id_z].lt.x + 1)))
        return;

    // NHWC: element offset = n * nStride + y * hStride + x * wStride (matches CPU grid_dropout NHWC
    // path)
    uint dstIdx = (batch_idx * dstStridesNHW.x) +
                  (id_y + anchorBoxInfoTensor[id_z].lt.y) * dstStridesNHW.y +
                  (id_x + anchorBoxInfoTensor[id_z].lt.x) * dstStridesNHW.z;

    if constexpr (std::is_same<T, Rpp8s>::value) {
        *reinterpret_cast<char3*>(&dstPtr[dstIdx]) = make_char3(-128, -128, -128);
    } else if constexpr (std::is_same<T, Rpp8u>::value) {
        *reinterpret_cast<uchar3*>(&dstPtr[dstIdx]) = make_uchar3(0, 0, 0);
    } else if constexpr (std::is_same<T, Rpp32f>::value) {
        *reinterpret_cast<float3*>(&dstPtr[dstIdx]) = make_float3(0.0f, 0.0f, 0.0f);
    } else  // half
    {
        *reinterpret_cast<d_half3_s*>(&dstPtr[dstIdx]) = {(half)0, (half)0, (half)0};
    }
}

template <typename T>
__global__ void grid_dropout_pln_hip_tensor(T* dstPtr, uint3 dstStridesNCH,
                                            RpptRoiLtrb* anchorBoxInfoTensor,
                                            uint boxesInEachImage) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int batch_idx = id_z / boxesInEachImage;

    if ((id_y >= (anchorBoxInfoTensor[id_z].rb.y - anchorBoxInfoTensor[id_z].lt.y + 1)) ||
        (id_x >= (anchorBoxInfoTensor[id_z].rb.x - anchorBoxInfoTensor[id_z].lt.x + 1)))
        return;

    uint dstIdx = (batch_idx * dstStridesNCH.x) +
                  (id_y + anchorBoxInfoTensor[id_z].lt.y) * dstStridesNCH.z +
                  (id_x + anchorBoxInfoTensor[id_z].lt.x);

    if constexpr (std::is_same<T, Rpp8s>::value) {
        dstPtr[dstIdx] = -128.0f;
        dstPtr[dstIdx + dstStridesNCH.y] = -128.0f;
        dstPtr[dstIdx + dstStridesNCH.y * 2] = -128.0f;
    } else {
        dstPtr[dstIdx] = 0.0f;
        dstPtr[dstIdx + dstStridesNCH.y] = 0.0f;
        dstPtr[dstIdx + dstStridesNCH.y * 2] = 0.0f;
    }
}

template <typename T>
__global__ void grid_dropout_pln1_hip_tensor(T* dstPtr, uint3 dstStridesNCH,
                                             RpptRoiLtrb* anchorBoxInfoTensor,
                                             uint boxesInEachImage) {
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int batch_idx = id_z / boxesInEachImage;

    if ((id_y >= (anchorBoxInfoTensor[id_z].rb.y - anchorBoxInfoTensor[id_z].lt.y + 1)) ||
        (id_x >= (anchorBoxInfoTensor[id_z].rb.x - anchorBoxInfoTensor[id_z].lt.x + 1)))
        return;

    uint dstIdx = (batch_idx * dstStridesNCH.x) +
                  (id_y + anchorBoxInfoTensor[id_z].lt.y) * dstStridesNCH.z +
                  (id_x + anchorBoxInfoTensor[id_z].lt.x);
    if constexpr (std::is_same<T, Rpp8s>::value)
        dstPtr[dstIdx] = -128.0f;
    else
        dstPtr[dstIdx] = 0.0f;
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_grid_dropout_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                       RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                                       Rpp32u boxesInEachImage, Rpp32u maxHoleW, Rpp32u maxHoleH,
                                       RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                       rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = maxHoleW;
    int globalThreads_y = maxHoleH;
    int globalThreads_z = srcDescPtr->n * boxesInEachImage;
    if (dstDescPtr->layout == RpptLayout::NHWC) {
        if (dstDescPtr->c != 3) return RPP_ERROR_NOT_IMPLEMENTED;

        // if src layout is NHWC, copy src to dst
        if (srcDescPtr->layout == RpptLayout::NHWC)
            RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(
                dstPtr, srcPtr,
                static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)),
                hipMemcpyDeviceToDevice, handle.GetStream()));
        else if (srcDescPtr->layout == RpptLayout::NCHW) {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = handle.GetBatchSize();
            hipLaunchKernelGGL(convert_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            globalThreads_x = maxHoleW;
            globalThreads_y = maxHoleH;
            globalThreads_z = srcDescPtr->n * boxesInEachImage;
        }

        hipLaunchKernelGGL(grid_dropout_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride,
                                      dstDescPtr->strides.wStride),
                           anchorBoxInfoTensor, boxesInEachImage);
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->c == 1)) {
        RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(
            dstPtr, srcPtr,
            static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)),
            hipMemcpyDeviceToDevice, handle.GetStream()));
        hipLaunchKernelGGL(grid_dropout_pln1_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor, boxesInEachImage);
    } else if ((dstDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->c == 3)) {
        // if src layout is NCHW, copy src to dst
        if (srcDescPtr->layout == RpptLayout::NCHW)
            RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(
                dstPtr, srcPtr,
                static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)),
                hipMemcpyDeviceToDevice, handle.GetStream()));
        else if (srcDescPtr->layout == RpptLayout::NHWC) {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = handle.GetBatchSize();
            hipLaunchKernelGGL(convert_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            globalThreads_x = maxHoleW;
            globalThreads_y = maxHoleH;
            globalThreads_z = srcDescPtr->n * boxesInEachImage;
        }

        hipLaunchKernelGGL(grid_dropout_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                           handle.GetStream(), dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                      dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor, boxesInEachImage);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_grid_dropout_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                       RpptRoiLtrb*, Rpp32u, Rpp32u, Rpp32u,
                                                       RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_grid_dropout_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr,
                                                      RpptRoiLtrb*, Rpp32u, Rpp32u, Rpp32u,
                                                      RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_grid_dropout_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                        RpptRoiLtrb*, Rpp32u, Rpp32u, Rpp32u,
                                                        RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_grid_dropout_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                       RpptRoiLtrb*, Rpp32u, Rpp32u, Rpp32u,
                                                       RpptROIPtr, RpptRoiType, rpp::Handle&);
