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
#include "rpp_hip_rgb_hsv_conversion.hpp"

__device__ void hue_1rgb_hip_compute(float* pixelR, float* pixelG, float* pixelB, float* hueParam) {
    // Convert RGB to HSV
    float hue, sat, val, add = 0.0f;
    rgb_to_hsv_hip(pixelR, pixelG, pixelB, hue, sat, val, add);

    // Apply hue adjustment
    hue += *hueParam + add;

    // Normalize hue to 0-6 range
    if (hue < 0)
        hue += 6.0f;
    else if (hue >= 6.0f)
        hue -= 6.0f;

    // Convert HSV to RGB
    hsv_to_rgb_hip(hue, sat, val, pixelR, pixelG, pixelB);
}

__device__ void hue_8rgb_hip_compute(d_float24* pix_f24, float* hueParam) {
    hue_1rgb_hip_compute(&(pix_f24->f1[0]), &(pix_f24->f1[8]), &(pix_f24->f1[16]), hueParam);
    hue_1rgb_hip_compute(&(pix_f24->f1[1]), &(pix_f24->f1[9]), &(pix_f24->f1[17]), hueParam);
    hue_1rgb_hip_compute(&(pix_f24->f1[2]), &(pix_f24->f1[10]), &(pix_f24->f1[18]), hueParam);
    hue_1rgb_hip_compute(&(pix_f24->f1[3]), &(pix_f24->f1[11]), &(pix_f24->f1[19]), hueParam);
    hue_1rgb_hip_compute(&(pix_f24->f1[4]), &(pix_f24->f1[12]), &(pix_f24->f1[20]), hueParam);
    hue_1rgb_hip_compute(&(pix_f24->f1[5]), &(pix_f24->f1[13]), &(pix_f24->f1[21]), hueParam);
    hue_1rgb_hip_compute(&(pix_f24->f1[6]), &(pix_f24->f1[14]), &(pix_f24->f1[22]), hueParam);
    hue_1rgb_hip_compute(&(pix_f24->f1[7]), &(pix_f24->f1[15]), &(pix_f24->f1[23]), hueParam);
}

template <typename T>
__global__ void hue_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                   float* hueTensor, RpptROIPtr roiTensorPtrSrc) {
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

    float hueModulus = fmodf(hueTensor[id_z], 360.0f);
    if (hueModulus < 0.0f) hueModulus += 360.0f;
    float hueParam = hueModulus * 0.01666667f;  // 6 * 1/360

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    hue_8rgb_hip_compute(&pix_f24, &hueParam);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void hue_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr, uint3 dstStridesNCH,
                                   float* hueTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) ||
        (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth)) {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) +
                  ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float hueModulus = fmodf(hueTensor[id_z], 360.0f);
    if (hueModulus < 0.0f) hueModulus += 360.0f;
    float hueParam = hueModulus * 0.01666667f;  // 6 * 1/360

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    hue_8rgb_hip_compute(&pix_f24, &hueParam);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void hue_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                         uint3 dstStridesNCH, float* hueTensor,
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

    float hueModulus = fmodf(hueTensor[id_z], 360.0f);
    if (hueModulus < 0.0f) hueModulus += 360.0f;
    float hueParam = hueModulus * 0.01666667f;  // 6 * 1/360

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    hue_8rgb_hip_compute(&pix_f24, &hueParam);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void hue_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                         uint2 dstStridesNH, float* hueTensor,
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

    float hueModulus = fmodf(hueTensor[id_z], 360.0f);
    if (hueModulus < 0.0f) hueModulus += 360.0f;
    float hueParam = hueModulus * 0.01666667f;  // 6 * 1/360

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    hue_8rgb_hip_compute(&pix_f24, &hueParam);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_hue_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                              Rpp32f* hueTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                              rpp::Handle& handle) {
    if (roiType == RpptRoiType::LTRB) hip_exec_roi_conversion_ltrb_to_xywh(roiTensorPtrSrc, handle);

    if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
            hipLaunchKernelGGL(
                hue_pkd_hip_tensor,
                dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                     ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                     ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(),
                srcPtr, make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                dstPtr, make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                hueTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            hipLaunchKernelGGL(hue_pln_hip_tensor,
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
                               hueTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NHWC) &&
                   (dstDescPtr->layout == RpptLayout::NCHW)) {
            hipLaunchKernelGGL(hue_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               hueTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            hipLaunchKernelGGL(hue_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               hueTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_hue_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr, Rpp32f*,
                                              RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_hue_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr, Rpp32f*,
                                             RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_hue_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr, Rpp32f*,
                                               RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_hue_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr, Rpp32f*,
                                              RpptROIPtr, RpptRoiType, rpp::Handle&);
