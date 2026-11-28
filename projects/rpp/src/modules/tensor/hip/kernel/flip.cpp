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

// The last block of a mirrored row is copied pixel by pixel, since an 8-pixel load would start left
// of the ROI.
template <typename T>
__device__ void flip_mirror_tail_hip_copy(T* srcPtr, uint srcIdx, uint srcPixelStride,
                                          uint srcChannelStride, T* dstPtr, uint dstIdx,
                                          uint dstPixelStride, uint dstChannelStride, int channels,
                                          int pixels) {
    for (int i = 0; i < pixels; i++, srcIdx -= srcPixelStride, dstIdx += dstPixelStride)
        for (int c = 0; c < channels; c++)
            dstPtr[dstIdx + c * dstChannelStride] = srcPtr[srcIdx + c * srcChannelStride];
}

template <typename T>
__global__ void flip_pkd_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr, uint2 dstStridesNH,
                                    uint2 dstDimsWH, unsigned int* horizontalTensor,
                                    unsigned int* verticalTensor, RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x) ||
        (id_y > roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y) ||
        (id_x > roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x)) {
        return;
    }

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    d_float24 pix_f24;
    uint srcIdx = id_z * srcStridesNH.x;
    uint horizontalFlag = horizontalTensor[id_z];
    uint verticalFlag = verticalTensor[id_z];

    RpptRoiLtrb roi = roiTensorPtrSrc[id_z].ltrbROI;
    int roiWidth = roi.rb.x - roi.lt.x + 1;
    if (horizontalFlag && (id_x + 8 > roiWidth)) {
        srcIdx += ((verticalFlag ? roi.rb.y - id_y : roi.lt.y + id_y) * srcStridesNH.y) +
                  (roi.rb.x - id_x) * 3;
        flip_mirror_tail_hip_copy(srcPtr, srcIdx, 3, 1, dstPtr, dstIdx, 3, 1, 3, roiWidth - id_x);
        return;
    }

    if (horizontalFlag == 0 && verticalFlag == 0) {
        srcIdx += ((id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNH.y) +
                  (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    } else if (horizontalFlag == 1 && verticalFlag == 1) {
        srcIdx += ((roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNH.y) +
                  (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    } else if (horizontalFlag == 1) {
        srcIdx += ((id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNH.y) +
                  (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    } else {
        srcIdx += ((roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNH.y) +
                  (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    }

    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void flip_pln_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr, uint3 dstStridesNCH,
                                    uint2 dstDimsWH, int channelsDst,
                                    unsigned int* horizontalTensor, unsigned int* verticalTensor,
                                    RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x) ||
        (id_y > roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y) ||
        (id_x > roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x)) {
        return;
    }

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint srcIdx = id_z * srcStridesNCH.x;
    uint horizontalFlag = horizontalTensor[id_z];
    uint verticalFlag = verticalTensor[id_z];

    RpptRoiLtrb roi = roiTensorPtrSrc[id_z].ltrbROI;
    int roiWidth = roi.rb.x - roi.lt.x + 1;
    if (horizontalFlag && (id_x + 8 > roiWidth)) {
        srcIdx += ((verticalFlag ? roi.rb.y - id_y : roi.lt.y + id_y) * srcStridesNCH.z) +
                  (roi.rb.x - id_x);
        flip_mirror_tail_hip_copy(srcPtr, srcIdx, 1, srcStridesNCH.y, dstPtr, dstIdx, 1,
                                  dstStridesNCH.y, channelsDst, roiWidth - id_x);
        return;
    }

    if (horizontalFlag == 0 && verticalFlag == 0)
        srcIdx += ((id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x);
    else if (horizontalFlag == 1 && verticalFlag == 1)
        srcIdx += ((roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNCH.z) +
                  (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7);
    else if (horizontalFlag == 1)
        srcIdx += ((id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNCH.z) +
                  (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7);
    else
        srcIdx += ((roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x);

    d_float8 pix_f8;

    if (horizontalFlag) {
        rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        if (channelsDst == 3) {
            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
    } else {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        if (channelsDst == 3) {
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
__global__ void flip_pkd3_pln3_hip_tensor(T* srcPtr, uint2 srcStridesNH, T* dstPtr,
                                          uint3 dstStridesNCH, uint2 dstDimsWH,
                                          unsigned int* horizontalTensor,
                                          unsigned int* verticalTensor,
                                          RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x) ||
        (id_y > roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y) ||
        (id_x > roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x)) {
        return;
    }

    d_float24 pix_f24;
    uint srcIdx = id_z * srcStridesNH.x;
    uint horizontalFlag = horizontalTensor[id_z];
    uint verticalFlag = verticalTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    RpptRoiLtrb roi = roiTensorPtrSrc[id_z].ltrbROI;
    int roiWidth = roi.rb.x - roi.lt.x + 1;
    if (horizontalFlag && (id_x + 8 > roiWidth)) {
        srcIdx += ((verticalFlag ? roi.rb.y - id_y : roi.lt.y + id_y) * srcStridesNH.y) +
                  (roi.rb.x - id_x) * 3;
        flip_mirror_tail_hip_copy(srcPtr, srcIdx, 3, 1, dstPtr, dstIdx, 1, dstStridesNCH.y, 3,
                                  roiWidth - id_x);
        return;
    }

    if (horizontalFlag == 0 && verticalFlag == 0) {
        srcIdx += ((id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNH.y) +
                  (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    } else if (horizontalFlag == 1 && verticalFlag == 1) {
        srcIdx += ((roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNH.y) +
                  (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    } else if (horizontalFlag == 1) {
        srcIdx += ((id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNH.y) +
                  (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    } else {
        srcIdx += ((roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNH.y) +
                  (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    }

    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void flip_pln3_pkd3_hip_tensor(T* srcPtr, uint3 srcStridesNCH, T* dstPtr,
                                          uint2 dstStridesNH, uint2 dstDimsWH,
                                          unsigned int* horizontalTensor,
                                          unsigned int* verticalTensor,
                                          RpptROIPtr roiTensorPtrSrc) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x) ||
        (id_y > roiTensorPtrSrc[id_z].ltrbROI.rb.y - roiTensorPtrSrc[id_z].ltrbROI.lt.y) ||
        (id_x > roiTensorPtrSrc[id_z].ltrbROI.rb.x - roiTensorPtrSrc[id_z].ltrbROI.lt.x)) {
        return;
    }

    d_float24 pix_f24;
    uint srcIdx = id_z * srcStridesNCH.x;
    uint horizontalFlag = horizontalTensor[id_z];
    uint verticalFlag = verticalTensor[id_z];
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    RpptRoiLtrb roi = roiTensorPtrSrc[id_z].ltrbROI;
    int roiWidth = roi.rb.x - roi.lt.x + 1;
    if (horizontalFlag && (id_x + 8 > roiWidth)) {
        srcIdx += ((verticalFlag ? roi.rb.y - id_y : roi.lt.y + id_y) * srcStridesNCH.z) +
                  (roi.rb.x - id_x);
        flip_mirror_tail_hip_copy(srcPtr, srcIdx, 1, srcStridesNCH.y, dstPtr, dstIdx, 3, 1, 3,
                                  roiWidth - id_x);
        return;
    }

    if (horizontalFlag == 0 && verticalFlag == 0) {
        srcIdx += ((id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    } else if (horizontalFlag == 1 && verticalFlag == 1) {
        srcIdx += ((roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNCH.z) +
                  (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, srcStridesNCH.y,
                                                              &pix_f24);
    } else if (horizontalFlag == 1) {
        srcIdx += ((id_y + roiTensorPtrSrc[id_z].ltrbROI.lt.y) * srcStridesNCH.z) +
                  (roiTensorPtrSrc[id_z].ltrbROI.rb.x - id_x - 7);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, srcStridesNCH.y,
                                                              &pix_f24);
    } else {
        srcIdx += ((roiTensorPtrSrc[id_z].ltrbROI.rb.y - id_y) * srcStridesNCH.z) +
                  (id_x + roiTensorPtrSrc[id_z].ltrbROI.lt.x);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    }

    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_flip_tensor(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr, RpptDescPtr dstDescPtr,
                               Rpp32u* horizontalTensor, Rpp32u* verticalTensor,
                               RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                               rpp::Handle& handle) {
    if (roiType == RpptRoiType::XYWH)
        RPP_RETURN_IF_ERROR(hip_exec_roi_conversion_xywh_to_ltrb(roiTensorPtrSrc, handle));

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        hipLaunchKernelGGL(
            flip_pkd_hip_tensor,
            dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                 ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
            make_uint2(dstDescPtr->w, dstDescPtr->h), horizontalTensor, verticalTensor,
            roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        hipLaunchKernelGGL(flip_pln_hip_tensor,
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
                           make_uint2(dstDescPtr->w, dstDescPtr->h), dstDescPtr->c,
                           horizontalTensor, verticalTensor, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            hipLaunchKernelGGL(flip_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h), horizontalTensor,
                               verticalTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(flip_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h), horizontalTensor,
                               verticalTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus hip_exec_flip_single_image(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32u* horizontalTensor,
                                     Rpp32u* verticalTensor, RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType, rpp::Handle& handle) {
    if (roiType == RpptRoiType::XYWH)
        RPP_RETURN_IF_ERROR(hip_exec_roi_conversion_xywh_to_ltrb(roiTensorPtrSrc, handle));

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = 1;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC)) {
        hipLaunchKernelGGL(
            flip_pkd_hip_tensor,
            dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                 ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                 ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
            dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0, handle.GetStream(), srcPtr,
            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride), dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
            make_uint2(dstDescPtr->w, dstDescPtr->h), horizontalTensor, verticalTensor,
            roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
               (dstDescPtr->layout == RpptLayout::NCHW)) {
        hipLaunchKernelGGL(flip_pln_hip_tensor,
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
                           make_uint2(dstDescPtr->w, dstDescPtr->h), dstDescPtr->c,
                           horizontalTensor, verticalTensor, roiTensorPtrSrc);
        HIP_CHECK_LAUNCH_RETURN();
    } else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3)) {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW)) {
            hipLaunchKernelGGL(flip_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride,
                                          dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h), horizontalTensor,
                               verticalTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        } else if ((srcDescPtr->layout == RpptLayout::NCHW) &&
                   (dstDescPtr->layout == RpptLayout::NHWC)) {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(flip_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X),
                                    ceil((float)globalThreads_y / LOCAL_THREADS_Y),
                                    ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z), 0,
                               handle.GetStream(), srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride,
                                          srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(dstDescPtr->w, dstDescPtr->h), horizontalTensor,
                               verticalTensor, roiTensorPtrSrc);
            HIP_CHECK_LAUNCH_RETURN();
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_flip_tensor<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr, Rpp32u*,
                                               Rpp32u*, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_flip_tensor<half>(half*, RpptDescPtr, half*, RpptDescPtr, Rpp32u*,
                                              Rpp32u*, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_flip_tensor<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr, Rpp32u*,
                                                Rpp32u*, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_flip_tensor<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr, Rpp32u*,
                                               Rpp32u*, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_flip_single_image<Rpp8u>(Rpp8u*, RpptDescPtr, Rpp8u*, RpptDescPtr,
                                                     Rpp32u*, Rpp32u*, RpptROIPtr, RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_flip_single_image<half>(half*, RpptDescPtr, half*, RpptDescPtr, Rpp32u*,
                                                    Rpp32u*, RpptROIPtr, RpptRoiType, rpp::Handle&);

template RppStatus hip_exec_flip_single_image<Rpp32f>(Rpp32f*, RpptDescPtr, Rpp32f*, RpptDescPtr,
                                                      Rpp32u*, Rpp32u*, RpptROIPtr, RpptRoiType,
                                                      rpp::Handle&);

template RppStatus hip_exec_flip_single_image<Rpp8s>(Rpp8s*, RpptDescPtr, Rpp8s*, RpptDescPtr,
                                                     Rpp32u*, Rpp32u*, RpptROIPtr, RpptRoiType,
                                                     rpp::Handle&);
