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

#include "rppt_tensor_effects_augmentations.h"

#include <random>

#include "api_helpers.hpp"
#include "fog_mask.hpp"
#include "host_tensor_executors.hpp"
#include "kernel_dims.hpp"
#include "rpp_cpu_random.hpp"
#include "rppdefs.h"
#include "rppt_validate.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif  // GPU_SUPPORT

/******************** gridmask ********************/

RppStatus rppt_gridmask(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio,
                        Rpp32f gridAngle, RpptUintVector2D translateVector,
                        RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                        RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            gridmask_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector,
                                       roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            gridmask_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc,
                roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            gridmask_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc,
                roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            gridmask_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector,
                                       roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            RPP_HIP_RETURN_IF_ERROR(
                hipMemset((void*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), 0,
                          dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(Rpp8u)));
            hip_exec_gridmask_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, tileWidth,
                gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            RPP_HIP_RETURN_IF_ERROR(
                hipMemset((void*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), 0,
                          dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(half)));
            hip_exec_gridmask_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            RPP_HIP_RETURN_IF_ERROR(
                hipMemset((void*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), 0,
                          dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(Rpp32f)));
            hip_exec_gridmask_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            RPP_HIP_RETURN_IF_ERROR(
                hipMemset((void*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), -128,
                          dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(Rpp8s)));
            hip_exec_gridmask_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, tileWidth,
                gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** spatter ********************/

RppStatus rppt_spatter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                       RpptDescPtr dstDescPtr, RpptRGB spatterColor, RpptROIPtr roiTensorPtrSrc,
                       RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if (roiType == RpptRoiType::XYWH) {
            for (int i = 0; i < srcDescPtr->n; i++)
                if ((roiTensorPtrSrc[i].xywhROI.roiWidth > SPATTER_MAX_WIDTH) ||
                    (roiTensorPtrSrc[i].xywhROI.roiHeight > SPATTER_MAX_HEIGHT))
                    return RPP_ERROR_HIGH_SRC_DIMENSION;
        } else if (roiType == RpptRoiType::LTRB) {
            for (int i = 0; i < srcDescPtr->n; i++)
                if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x >
                     SPATTER_MAX_XDIM) ||
                    (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y >
                     SPATTER_MAX_YDIM))
                    return RPP_ERROR_HIGH_SRC_DIMENSION;
        }

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            spatter_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, spatterColor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            spatter_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, spatterColor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            spatter_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, spatterColor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            spatter_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, spatterColor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        RpptROI roiTensorPtrSrcHost[dstDescPtr->n];
        RPP_HIP_RETURN_IF_ERROR(hipMemcpy(roiTensorPtrSrcHost, roiTensorPtrSrc,
                                          dstDescPtr->n * sizeof(RpptROI), hipMemcpyDeviceToHost));
        if (roiType == RpptRoiType::XYWH) {
            for (int i = 0; i < dstDescPtr->n; i++)
                if ((roiTensorPtrSrcHost[i].xywhROI.roiWidth > SPATTER_MAX_WIDTH) ||
                    (roiTensorPtrSrcHost[i].xywhROI.roiHeight > SPATTER_MAX_HEIGHT))
                    return RPP_ERROR_HIGH_SRC_DIMENSION;
        } else if (roiType == RpptRoiType::LTRB) {
            for (int i = 0; i < dstDescPtr->n; i++)
                if ((roiTensorPtrSrcHost[i].ltrbROI.rb.x - roiTensorPtrSrcHost[i].ltrbROI.lt.x >
                     SPATTER_MAX_XDIM) ||
                    (roiTensorPtrSrcHost[i].ltrbROI.rb.y - roiTensorPtrSrcHost[i].ltrbROI.lt.y >
                     SPATTER_MAX_YDIM))
                    return RPP_ERROR_HIGH_SRC_DIMENSION;
        }

        std::random_device rd;   // Random number engine seed
        std::mt19937 gen(rd());  // Seeding rd() to fast mersenne twister engine
        Rpp32u *maskLocArrHostX = nullptr, *maskLocArrHostY = nullptr;
        RPP_HIP_RETURN_IF_ERROR(hipHostMalloc(&maskLocArrHostX, dstDescPtr->n * sizeof(Rpp32u)));
        RPP_HIP_RETURN_IF_ERROR(hipHostMalloc(&maskLocArrHostY, dstDescPtr->n * sizeof(Rpp32u)));

        for (int i = 0; i < dstDescPtr->n; i++) {
            std::uniform_int_distribution<> distribX(
                0, SPATTER_MAX_WIDTH - roiTensorPtrSrcHost[i].xywhROI.roiWidth);
            std::uniform_int_distribution<> distribY(
                0, SPATTER_MAX_HEIGHT - roiTensorPtrSrcHost[i].xywhROI.roiHeight);
            maskLocArrHostX[i] = distribX(gen);
            maskLocArrHostY[i] = distribY(gen);
        }
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_spatter_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, spatterColor,
                maskLocArrHostX, maskLocArrHostY, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_spatter_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, spatterColor, maskLocArrHostX, maskLocArrHostY, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_spatter_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, spatterColor, maskLocArrHostX, maskLocArrHostY, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_spatter_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, spatterColor,
                maskLocArrHostX, maskLocArrHostY, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        RPP_HIP_RETURN_IF_ERROR(hipHostFree(maskLocArrHostX));
        RPP_HIP_RETURN_IF_ERROR(hipHostFree(maskLocArrHostY));

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** salt_and_pepper_noise ********************/

RppStatus rppt_salt_and_pepper_noise(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                                     RpptDescPtr dstDescPtr, Rpp32f* noiseProbabilityTensor,
                                     Rpp32f* saltProbabilityTensor, Rpp32f* saltValueTensor,
                                     Rpp32f* pepperValueTensor, Rpp32u seed,
                                     RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                     rppHandle_t rppHandle, RppBackend executionBackend) {
    for (int i = 0; i < srcDescPtr->n; i++)
        if (!RPPINRANGE(noiseProbabilityTensor[i], 0, 1) ||
            !RPPINRANGE(saltProbabilityTensor[i], 0, 1) || !RPPINRANGE(saltValueTensor[i], 0, 1) ||
            !RPPINRANGE(pepperValueTensor[i], 0, 1))
            return RPP_ERROR_INVALID_ARGUMENTS;

    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        RpptXorwowState xorwowInitialState[SIMD_FLOAT_VECTOR_LENGTH];
        rpp_host_rng_xorwow_f32_initialize_multiseed_stream<SIMD_FLOAT_VECTOR_LENGTH>(
            xorwowInitialState, seed);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            salt_and_pepper_noise_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor,
                xorwowInitialState, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            salt_and_pepper_noise_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor,
                pepperValueTensor, xorwowInitialState, roiTensorPtrSrc, roiType, layoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            salt_and_pepper_noise_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor,
                pepperValueTensor, xorwowInitialState, roiTensorPtrSrc, roiType, layoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            salt_and_pepper_noise_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor,
                xorwowInitialState, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        RpptXorwowState xorwowInitialState;
        xorwowInitialState.x[0] = 0x75BCD15 + seed;
        xorwowInitialState.x[1] = 0x159A55E5 + seed;
        xorwowInitialState.x[2] = 0x1F123BB5 + seed;
        xorwowInitialState.x[3] = 0x5491333 + seed;
        xorwowInitialState.x[4] = 0x583F19 + seed;
        xorwowInitialState.counter = 0x64F0C9 + seed;

        RpptXorwowState* d_xorwowInitialStatePtr;
        d_xorwowInitialStatePtr =
            (RpptXorwowState*)handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        RPP_HIP_RETURN_IF_ERROR(hipMemcpy(d_xorwowInitialStatePtr, &xorwowInitialState,
                                          sizeof(RpptXorwowState), hipMemcpyHostToDevice));

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_salt_and_pepper_noise_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor,
                d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_salt_and_pepper_noise_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor,
                pepperValueTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_salt_and_pepper_noise_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor,
                pepperValueTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_salt_and_pepper_noise_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor,
                d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** shot_noise ********************/

RppStatus rppt_shot_noise(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr, Rpp32f* shotNoiseFactorTensor, Rpp32u seed,
                          RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                          RppBackend executionBackend) {
    for (int i = 0; i < srcDescPtr->n; i++)
        if (RPPISLESSER(shotNoiseFactorTensor[i], 0)) return RPP_ERROR_INVALID_ARGUMENTS;

    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        RpptXorwowState xorwowInitialState[SIMD_FLOAT_VECTOR_LENGTH];
        rpp_host_rng_xorwow_f32_initialize_multiseed_stream<SIMD_FLOAT_VECTOR_LENGTH>(
            xorwowInitialState, seed);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            shot_noise_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr, shotNoiseFactorTensor, xorwowInitialState,
                                         roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            shot_noise_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, shotNoiseFactorTensor, xorwowInitialState, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            shot_noise_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, shotNoiseFactorTensor, xorwowInitialState, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            shot_noise_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr, shotNoiseFactorTensor, xorwowInitialState,
                                         roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        RpptXorwowStateBoxMuller xorwowInitialState;
        xorwowInitialState.x[0] = 0x75BCD15 + seed;
        xorwowInitialState.x[1] = 0x159A55E5 + seed;
        xorwowInitialState.x[2] = 0x1F123BB5 + seed;
        xorwowInitialState.x[3] = 0x5491333 + seed;
        xorwowInitialState.x[4] = 0x583F19 + seed;
        xorwowInitialState.counter = 0x64F0C9 + seed;
        xorwowInitialState.boxMullerFlag = 0;
        xorwowInitialState.boxMullerExtra = 0.0f;

        RpptXorwowStateBoxMuller* d_xorwowInitialStatePtr;
        d_xorwowInitialStatePtr =
            (RpptXorwowStateBoxMuller*)handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        RPP_HIP_RETURN_IF_ERROR(hipMemcpy(d_xorwowInitialStatePtr, &xorwowInitialState,
                                          sizeof(RpptXorwowStateBoxMuller), hipMemcpyHostToDevice));

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_shot_noise_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                shotNoiseFactorTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_shot_noise_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, shotNoiseFactorTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_shot_noise_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, shotNoiseFactorTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_shot_noise_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                shotNoiseFactorTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** gaussian_noise ********************/

RppStatus rppt_gaussian_noise(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr, Rpp32f* meanTensor, Rpp32f* stdDevTensor,
                              Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                              rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        RpptXorwowStateBoxMuller xorwowInitialState[SIMD_FLOAT_VECTOR_LENGTH];
        rpp_host_rng_xorwow_f32_initialize_multiseed_stream_boxmuller<SIMD_FLOAT_VECTOR_LENGTH>(
            xorwowInitialState, seed);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            gaussian_noise_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, meanTensor,
                stdDevTensor, xorwowInitialState, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            gaussian_noise_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, meanTensor, stdDevTensor, xorwowInitialState, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            gaussian_noise_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, meanTensor, stdDevTensor, xorwowInitialState, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            gaussian_noise_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, meanTensor,
                stdDevTensor, xorwowInitialState, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        RpptXorwowStateBoxMuller xorwowInitialState;
        xorwowInitialState.x[0] = 0x75BCD15 + seed;
        xorwowInitialState.x[1] = 0x159A55E5 + seed;
        xorwowInitialState.x[2] = 0x1F123BB5 + seed;
        xorwowInitialState.x[3] = 0x5491333 + seed;
        xorwowInitialState.x[4] = 0x583F19 + seed;
        xorwowInitialState.counter = 0x64F0C9 + seed;
        xorwowInitialState.boxMullerFlag = 0;
        xorwowInitialState.boxMullerExtra = 0.0f;

        RpptXorwowStateBoxMuller* d_xorwowInitialStatePtr;
        d_xorwowInitialStatePtr =
            (RpptXorwowStateBoxMuller*)handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        RPP_HIP_RETURN_IF_ERROR(hipMemcpy(d_xorwowInitialStatePtr, &xorwowInitialState,
                                          sizeof(RpptXorwowStateBoxMuller), hipMemcpyHostToDevice));

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_gaussian_noise_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, meanTensor,
                stdDevTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_gaussian_noise_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, meanTensor, stdDevTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_gaussian_noise_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, meanTensor, stdDevTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_gaussian_noise_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, meanTensor,
                stdDevTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

RppStatus rppt_gaussian_noise_voxel(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr,
                                    RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32f* meanTensor, Rpp32f* stdDevTensor, Rpp32u seed,
                                    RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType,
                                    rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (srcGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) &&
        (dstGenericDescPtr->layout != RpptLayout::NDHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;
    if (srcGenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams;
        if ((srcGenericDescPtr->layout == RpptLayout::NCDHW) &&
            (dstGenericDescPtr->layout == RpptLayout::NCDHW))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);
        else if ((srcGenericDescPtr->layout == RpptLayout::NDHWC) &&
                 (dstGenericDescPtr->layout == RpptLayout::NDHWC))
            layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[4]);

        RpptXorwowStateBoxMuller xorwowInitialState[SIMD_FLOAT_VECTOR_LENGTH];
        rpp_host_rng_xorwow_f32_initialize_multiseed_stream_boxmuller<SIMD_FLOAT_VECTOR_LENGTH>(
            xorwowInitialState, seed);

        if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            gaussian_noise_voxel_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                meanTensor, stdDevTensor, xorwowInitialState, roiGenericPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            gaussian_noise_voxel_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, meanTensor, stdDevTensor, xorwowInitialState, roiGenericPtrSrc,
                roiType, layoutParams, handle);
        } else {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        RpptXorwowStateBoxMuller xorwowInitialState;
        xorwowInitialState.x[0] = 0x75BCD15 + seed;
        xorwowInitialState.x[1] = 0x159A55E5 + seed;
        xorwowInitialState.x[2] = 0x1F123BB5 + seed;
        xorwowInitialState.x[3] = 0x5491333 + seed;
        xorwowInitialState.x[4] = 0x583F19 + seed;
        xorwowInitialState.counter = 0x64F0C9 + seed;
        xorwowInitialState.boxMullerFlag = 0;
        xorwowInitialState.boxMullerExtra = 0.0f;

        RpptXorwowStateBoxMuller* d_xorwowInitialStatePtr;
        d_xorwowInitialStatePtr =
            (RpptXorwowStateBoxMuller*)handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        RPP_HIP_RETURN_IF_ERROR(hipMemcpy(d_xorwowInitialStatePtr, &xorwowInitialState,
                                          sizeof(RpptXorwowStateBoxMuller), hipMemcpyHostToDevice));

        if ((srcGenericDescPtr->dataType == RpptDataType::U8) &&
            (dstGenericDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_gaussian_noise_voxel_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes, srcGenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                d_xorwowInitialStatePtr, meanTensor, stdDevTensor, roiGenericPtrSrc, handle);
        } else if ((srcGenericDescPtr->dataType == RpptDataType::F32) &&
                   (dstGenericDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_gaussian_noise_voxel_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) +
                                          srcGenericDescPtr->offsetInBytes),
                srcGenericDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, d_xorwowInitialStatePtr, meanTensor, stdDevTensor,
                roiGenericPtrSrc, handle);
        } else {
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** non_linear_blend ********************/

RppStatus rppt_non_linear_blend(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f* stdDevTensor,
                                RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                rppHandle_t rppHandle, RppBackend executionBackend) {
    for (int i = 0; i < srcDescPtr->n; i++)
        if (stdDevTensor[i] == 0) return RPP_ERROR_ZERO_DIVISION;

    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            non_linear_blend_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, stdDevTensor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            non_linear_blend_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, stdDevTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            non_linear_blend_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, stdDevTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            non_linear_blend_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, stdDevTensor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_non_linear_blend_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, stdDevTensor,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_non_linear_blend_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, stdDevTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_non_linear_blend_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, stdDevTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_non_linear_blend_tensor(
                static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, stdDevTensor,
                roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** water ********************/

RppStatus rppt_water(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr, Rpp32f* amplitudeXTensor, Rpp32f* amplitudeYTensor,
                     Rpp32f* frequencyXTensor, Rpp32f* frequencyYTensor, Rpp32f* phaseXTensor,
                     Rpp32f* phaseYTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                     rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            water_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                amplitudeXTensor, amplitudeYTensor, frequencyXTensor, frequencyYTensor,
                phaseXTensor, phaseYTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            water_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, amplitudeXTensor, amplitudeYTensor, frequencyXTensor, frequencyYTensor,
                phaseXTensor, phaseYTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            water_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, amplitudeXTensor, amplitudeYTensor, frequencyXTensor, frequencyYTensor,
                phaseXTensor, phaseYTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            water_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                amplitudeXTensor, amplitudeYTensor, frequencyXTensor, frequencyYTensor,
                phaseXTensor, phaseYTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_water_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                amplitudeXTensor, amplitudeYTensor, frequencyXTensor, frequencyYTensor,
                phaseXTensor, phaseYTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_water_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, amplitudeXTensor, amplitudeYTensor, frequencyXTensor, frequencyYTensor,
                phaseXTensor, phaseYTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_water_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, amplitudeXTensor, amplitudeYTensor, frequencyXTensor, frequencyYTensor,
                phaseXTensor, phaseYTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_water_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                amplitudeXTensor, amplitudeYTensor, frequencyXTensor, frequencyYTensor,
                phaseXTensor, phaseYTensor, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** vignette ********************/

RppStatus rppt_vignette(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr, Rpp32f* vignetteIntensityTensor,
                        RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                        RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            vignette_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                vignetteIntensityTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            vignette_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, vignetteIntensityTensor, roiTensorPtrSrc, roiType, layoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            vignette_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, vignetteIntensityTensor, roiTensorPtrSrc, roiType, layoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            vignette_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                vignetteIntensityTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_vignette_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrSrc, vignetteIntensityTensor, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_vignette_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, vignetteIntensityTensor, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_vignette_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, roiTensorPtrSrc, vignetteIntensityTensor, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_vignette_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrSrc, vignetteIntensityTensor, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** erase ********************/

RppStatus rppt_erase(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                     RppPtr_t colorsTensor, Rpp32u* numBoxesTensor, RpptROIPtr roiTensorPtrSrc,
                     RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            erase_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                              anchorBoxInfoTensor, static_cast<Rpp8u*>(colorsTensor),
                              numBoxesTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            erase_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + srcDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp16f*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            erase_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp32f*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            erase_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                              anchorBoxInfoTensor, static_cast<Rpp8s*>(colorsTensor),
                              numBoxesTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_erase_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, static_cast<Rpp8u*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_erase_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<half*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_erase_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp32f*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_erase_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, static_cast<Rpp8s*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** ricap ********************/

RppStatus rppt_ricap(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr, Rpp32u* permutationTensor,
                     RpptROIPtr roiPtrInputCropRegion, RpptRoiType roiType, rppHandle_t rppHandle,
                     RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcDescPtr->n == 1) return RPP_ERROR;
    if ((check_roi_out_of_bounds(&roiPtrInputCropRegion[0], srcDescPtr, roiType) == -1) ||
        (check_roi_out_of_bounds(&roiPtrInputCropRegion[1], srcDescPtr, roiType) == -1) ||
        (check_roi_out_of_bounds(&roiPtrInputCropRegion[2], srcDescPtr, roiType) == -1) ||
        (check_roi_out_of_bounds(&roiPtrInputCropRegion[3], srcDescPtr, roiType) == -1))
        return RPP_ERROR_OUT_OF_BOUND_SRC_ROI;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            ricap_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                permutationTensor, roiPtrInputCropRegion, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            ricap_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, permutationTensor, roiPtrInputCropRegion, roiType, layoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            ricap_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, permutationTensor, roiPtrInputCropRegion, roiType, layoutParams,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            ricap_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                permutationTensor, roiPtrInputCropRegion, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        Rpp32u* permutationHipTensor =
            reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        RPP_HIP_RETURN_IF_ERROR(hipMemcpy(permutationHipTensor, permutationTensor,
                                          sizeof(Rpp32u) * 4 * dstDescPtr->n,
                                          hipMemcpyHostToDevice));

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_ricap_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                permutationHipTensor, roiPtrInputCropRegion, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_ricap_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, permutationHipTensor, roiPtrInputCropRegion, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_ricap_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, permutationHipTensor, roiPtrInputCropRegion, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_ricap_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                permutationHipTensor, roiPtrInputCropRegion, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** glitch ********************/

RppStatus rppt_glitch(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                      RpptDescPtr dstDescPtr, RpptChannelOffsets* rgbOffsets,
                      RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                      RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            glitch_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rgbOffsets,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            glitch_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rgbOffsets, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            glitch_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rgbOffsets, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            glitch_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rgbOffsets,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_glitch_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr, rgbOffsets, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_glitch_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rgbOffsets, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_glitch_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rgbOffsets, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_glitch_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr, rgbOffsets, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** jitter ********************/

RppStatus rppt_jitter(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                      RpptDescPtr dstDescPtr, Rpp32u* kernelSizeTensor, Rpp32u seed,
                      RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                      RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        RpptXorwowStateBoxMuller xorwowInitialState[SIMD_FLOAT_VECTOR_LENGTH];
        rpp_host_rng_xorwow_f32_initialize_multiseed_stream_boxmuller<SIMD_FLOAT_VECTOR_LENGTH>(
            xorwowInitialState, seed);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            jitter_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr, kernelSizeTensor, xorwowInitialState,
                                     roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            jitter_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, kernelSizeTensor, xorwowInitialState, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            jitter_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, kernelSizeTensor, xorwowInitialState, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            jitter_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr, kernelSizeTensor, xorwowInitialState,
                                     roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        RpptXorwowStateBoxMuller xorwowInitialState;
        xorwowInitialState.x[0] = 0x75BCD15 + seed;
        xorwowInitialState.x[1] = 0x159A55E5 + seed;
        xorwowInitialState.x[2] = 0x1F123BB5 + seed;
        xorwowInitialState.x[3] = 0x5491333 + seed;
        xorwowInitialState.x[4] = 0x583F19 + seed;
        xorwowInitialState.counter = 0x64F0C9 + seed;
        xorwowInitialState.boxMullerFlag = 0;
        xorwowInitialState.boxMullerExtra = 0.0f;

        RpptXorwowStateBoxMuller* d_xorwowInitialStatePtr;
        d_xorwowInitialStatePtr = reinterpret_cast<RpptXorwowStateBoxMuller*>(
            handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        RPP_HIP_RETURN_IF_ERROR(hipMemcpy(d_xorwowInitialStatePtr, &xorwowInitialState,
                                          sizeof(RpptXorwowStateBoxMuller), hipMemcpyHostToDevice));

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_jitter_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                kernelSizeTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_jitter_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, kernelSizeTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_jitter_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, kernelSizeTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_jitter_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                kernelSizeTensor, d_xorwowInitialStatePtr, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** pixelate ********************/

RppStatus rppt_pixelate(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr, RppPtr_t interDstPtr, Rpp32f pixelationPercentage,
                        RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                        RppBackend executionBackend) {
    // This function performs pixelation through a two-step resizing process:
    // 1. The image is first resized to a smaller intermediate size using bilinear interpolation.
    // 2. The intermediate image is then resized back to the original size using nearest neighbor
    // interpolation.

    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (pixelationPercentage < 0 || pixelationPercentage > 100) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        RpptImagePatchPtr internalDstImgSizes =
            reinterpret_cast<RpptImagePatch*>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
        RpptROI* internalRoiTensorPtrSrc =
            reinterpret_cast<RpptROI*>(internalDstImgSizes + dstDescPtr->n);

        for (int i = 0; i < dstDescPtr->n; i++) {
            internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth =
                (roiTensorPtrSrc[i].xywhROI.roiWidth * (100 - pixelationPercentage)) / 100;
            internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight =
                (roiTensorPtrSrc[i].xywhROI.roiHeight * (100 - pixelationPercentage)) / 100;
            internalRoiTensorPtrSrc[i].xywhROI.xy.x =
                (roiTensorPtrSrc[i].xywhROI.xy.x * (100 - pixelationPercentage)) / 100;
            internalRoiTensorPtrSrc[i].xywhROI.xy.y =
                (roiTensorPtrSrc[i].xywhROI.xy.y * (100 - pixelationPercentage)) / 100;
        }

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            resize_bilinear_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(interDstPtr), srcDescPtr, internalDstImgSizes, roiTensorPtrSrc,
                roiType, srcLayoutParams, handle);
            for (int i = 0; i < dstDescPtr->n; i++) {
                internalDstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
                internalDstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
            }
            resize_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(interDstPtr), srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr, internalDstImgSizes, internalRoiTensorPtrSrc,
                                        roiType, srcLayoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            resize_bilinear_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                reinterpret_cast<Rpp16f*>(interDstPtr), srcDescPtr, internalDstImgSizes,
                roiTensorPtrSrc, roiType, srcLayoutParams, handle);
            for (int i = 0; i < dstDescPtr->n; i++) {
                internalDstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
                internalDstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
            }
            resize_nn_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(interDstPtr), srcDescPtr,
                reinterpret_cast<Rpp16f*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                internalDstImgSizes, internalRoiTensorPtrSrc, roiType, srcLayoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            resize_bilinear_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                reinterpret_cast<Rpp32f*>(interDstPtr), srcDescPtr, internalDstImgSizes,
                roiTensorPtrSrc, roiType, srcLayoutParams, handle);
            for (int i = 0; i < dstDescPtr->n; i++) {
                internalDstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
                internalDstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
            }
            resize_nn_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(interDstPtr), srcDescPtr,
                reinterpret_cast<Rpp32f*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                internalDstImgSizes, internalRoiTensorPtrSrc, roiType, srcLayoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            resize_bilinear_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(interDstPtr), srcDescPtr, internalDstImgSizes, roiTensorPtrSrc,
                roiType, srcLayoutParams, handle);
            for (int i = 0; i < dstDescPtr->n; i++) {
                internalDstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
                internalDstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
            }
            resize_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(interDstPtr), srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr, internalDstImgSizes, internalRoiTensorPtrSrc,
                                        roiType, srcLayoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
        RpptDesc interDesc = *srcDescPtr;
        RpptDescPtr interDescPtr = &interDesc;

        RpptImagePatchPtr internalDstImgSizes = reinterpret_cast<RpptImagePatch*>(
            handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
        RpptROI* internalRoiTensorPtrSrc =
            reinterpret_cast<RpptROI*>(internalDstImgSizes + dstDescPtr->n);

        for (int i = 0; i < srcDescPtr->n; i++) {
            internalRoiTensorPtrSrc[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth;
            internalRoiTensorPtrSrc[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight;
            internalDstImgSizes[i].width =
                (roiTensorPtrSrc[i].xywhROI.roiWidth * (100 - pixelationPercentage)) / 100;
            internalDstImgSizes[i].height =
                (roiTensorPtrSrc[i].xywhROI.roiHeight * (100 - pixelationPercentage)) / 100;
        }

        interDescPtr->h = (interDescPtr->h * (100 - pixelationPercentage)) / 100;
        interDescPtr->w = (((interDescPtr->w * ((100 - pixelationPercentage) / 100)) / 8) * 8) + 8;
        interDescPtr->strides.nStride = interDescPtr->w * interDescPtr->h * interDescPtr->c;
        interDescPtr->strides.cStride =
            (srcDescPtr->layout == RpptLayout::NCHW) ? (interDescPtr->w * interDescPtr->h) : 1;
        interDescPtr->strides.hStride = (srcDescPtr->layout == RpptLayout::NCHW)
                                            ? interDescPtr->w
                                            : (interDescPtr->c * interDescPtr->w);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_resize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr, static_cast<Rpp8u*>(interDstPtr), interDescPtr,
                                   internalDstImgSizes, interpolationType, roiTensorPtrSrc, roiType,
                                   handle);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
            for (int i = 0; i < srcDescPtr->n; i++) {
                internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth;
                internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight;
                internalRoiTensorPtrSrc[i].xywhROI.roiWidth =
                    (internalRoiTensorPtrSrc[i].xywhROI.roiWidth * (100 - pixelationPercentage)) /
                    100;
                internalRoiTensorPtrSrc[i].xywhROI.roiHeight =
                    (internalRoiTensorPtrSrc[i].xywhROI.roiHeight * (100 - pixelationPercentage)) /
                    100;
                internalRoiTensorPtrSrc[i].xywhROI.xy.x =
                    (internalRoiTensorPtrSrc[i].xywhROI.xy.x * (100 - pixelationPercentage)) / 100;
                internalRoiTensorPtrSrc[i].xywhROI.xy.y =
                    (internalRoiTensorPtrSrc[i].xywhROI.xy.y * (100 - pixelationPercentage)) / 100;
            }
            hip_exec_resize_tensor(static_cast<Rpp8u*>(interDstPtr), interDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr, internalDstImgSizes, interpolationType,
                                   internalRoiTensorPtrSrc, roiType, handle);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_resize_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr, reinterpret_cast<half*>(static_cast<Rpp8u*>(interDstPtr)), interDescPtr,
                internalDstImgSizes, interpolationType, roiTensorPtrSrc, roiType, handle);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
            for (int i = 0; i < srcDescPtr->n; i++) {
                internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth;
                internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight;
                internalRoiTensorPtrSrc[i].xywhROI.roiWidth =
                    (internalRoiTensorPtrSrc[i].xywhROI.roiWidth * (100 - pixelationPercentage)) /
                    100;
                internalRoiTensorPtrSrc[i].xywhROI.roiHeight =
                    (internalRoiTensorPtrSrc[i].xywhROI.roiHeight * (100 - pixelationPercentage)) /
                    100;
                internalRoiTensorPtrSrc[i].xywhROI.xy.x =
                    (internalRoiTensorPtrSrc[i].xywhROI.xy.x * (100 - pixelationPercentage)) / 100;
                internalRoiTensorPtrSrc[i].xywhROI.xy.y =
                    (internalRoiTensorPtrSrc[i].xywhROI.xy.y * (100 - pixelationPercentage)) / 100;
            }
            hip_exec_resize_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(interDstPtr)), interDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, internalDstImgSizes, interpolationType, internalRoiTensorPtrSrc,
                roiType, handle);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_resize_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr, reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(interDstPtr)),
                interDescPtr, internalDstImgSizes, interpolationType, roiTensorPtrSrc, roiType,
                handle);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
            for (int i = 0; i < srcDescPtr->n; i++) {
                internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth;
                internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight;
                internalRoiTensorPtrSrc[i].xywhROI.roiWidth =
                    (internalRoiTensorPtrSrc[i].xywhROI.roiWidth * (100 - pixelationPercentage)) /
                    100;
                internalRoiTensorPtrSrc[i].xywhROI.roiHeight =
                    (internalRoiTensorPtrSrc[i].xywhROI.roiHeight * (100 - pixelationPercentage)) /
                    100;
                internalRoiTensorPtrSrc[i].xywhROI.xy.x =
                    (internalRoiTensorPtrSrc[i].xywhROI.xy.x * (100 - pixelationPercentage)) / 100;
                internalRoiTensorPtrSrc[i].xywhROI.xy.y =
                    (internalRoiTensorPtrSrc[i].xywhROI.xy.y * (100 - pixelationPercentage)) / 100;
            }
            hip_exec_resize_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(interDstPtr)), interDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, internalDstImgSizes, interpolationType, internalRoiTensorPtrSrc,
                roiType, handle);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_resize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr, static_cast<Rpp8s*>(interDstPtr), interDescPtr,
                                   internalDstImgSizes, interpolationType, roiTensorPtrSrc, roiType,
                                   handle);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
            for (int i = 0; i < srcDescPtr->n; i++) {
                internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth;
                internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight;
                internalRoiTensorPtrSrc[i].xywhROI.roiWidth =
                    (internalRoiTensorPtrSrc[i].xywhROI.roiWidth * (100 - pixelationPercentage)) /
                    100;
                internalRoiTensorPtrSrc[i].xywhROI.roiHeight =
                    (internalRoiTensorPtrSrc[i].xywhROI.roiHeight * (100 - pixelationPercentage)) /
                    100;
                internalRoiTensorPtrSrc[i].xywhROI.xy.x =
                    (internalRoiTensorPtrSrc[i].xywhROI.xy.x * (100 - pixelationPercentage)) / 100;
                internalRoiTensorPtrSrc[i].xywhROI.xy.y =
                    (internalRoiTensorPtrSrc[i].xywhROI.xy.y * (100 - pixelationPercentage)) / 100;
            }
            hip_exec_resize_tensor(static_cast<Rpp8s*>(interDstPtr), interDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr, internalDstImgSizes, interpolationType,
                                   internalRoiTensorPtrSrc, roiType, handle);
            RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** fog ********************/

RppStatus rppt_fog(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr,
                   Rpp32f* intensityFactor, Rpp32f* grayFactor, RpptROIPtr roiTensorPtrSrc,
                   RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        // Do the setup required for resizing the fog mask based on max size in the batch

        // Initialize and set descriptor for original fog mask
        RpptDesc fogMaskSrcDesc;
        RpptDescPtr fogMaskSrcDescPtr = &fogMaskSrcDesc;
        set_fog_mask_descriptor(fogMaskSrcDescPtr, 2, FOG_MAX_HEIGHT, FOG_MAX_WIDTH, 1);

        // Initialize and set descriptor for resized fog mask
        RpptDesc fogMaskDstDesc;
        RpptDescPtr fogMaskDstDescPtr = &fogMaskDstDesc;
        set_fog_mask_descriptor(fogMaskDstDescPtr, 2, srcDescPtr->h, srcDescPtr->w, 1);

        // Fill the ROI and dstImageSize values required for resize api call
        RpptImagePatchPtr internalDstImgSizes =
            reinterpret_cast<RpptImagePatch*>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
        RpptROI* internalRoiTensorPtrSrc = reinterpret_cast<RpptROI*>(internalDstImgSizes + 2);
        for (Rpp32s i = 0; i < 2; i++) {
            internalDstImgSizes[i] = {srcDescPtr->w, srcDescPtr->h};
            internalRoiTensorPtrSrc[i].xywhROI = {{0, 0}, 1920, 1080};
        }
        RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
        Rpp32f* resizedFogAlphaMaskPtr = reinterpret_cast<Rpp32f*>(internalRoiTensorPtrSrc + 2);
        Rpp32f* resizedFogIntensityMaskPtr =
            resizedFogAlphaMaskPtr + (srcDescPtr->h * srcDescPtr->w);

        // Resize the mask to the maximum size present in the batch
        rppt_resize(&fogMask_1920_1080[0], fogMaskSrcDescPtr, resizedFogAlphaMaskPtr,
                    fogMaskDstDescPtr, internalDstImgSizes, interpolationType,
                    internalRoiTensorPtrSrc, roiType, rppHandle, RPP_HOST_BACKEND);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            fog_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                resizedFogAlphaMaskPtr, resizedFogIntensityMaskPtr, intensityFactor, grayFactor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            fog_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, resizedFogAlphaMaskPtr, resizedFogIntensityMaskPtr, intensityFactor,
                grayFactor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            fog_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, resizedFogAlphaMaskPtr, resizedFogIntensityMaskPtr, intensityFactor,
                grayFactor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            fog_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                resizedFogAlphaMaskPtr, resizedFogIntensityMaskPtr, intensityFactor, grayFactor,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        // Do the setup required for resizing the fog mask based on max size in the batch

        // Initialize and set descriptor for original fog mask
        RpptDesc fogMaskSrcDesc;
        RpptDescPtr fogMaskSrcDescPtr = &fogMaskSrcDesc;
        set_fog_mask_descriptor(fogMaskSrcDescPtr, 2, FOG_MAX_HEIGHT, FOG_MAX_WIDTH, 1);

        // Initialize and set descriptor for resized fog mask
        RpptDesc fogMaskDstDesc;
        RpptDescPtr fogMaskDstDescPtr = &fogMaskDstDesc;
        set_fog_mask_descriptor(fogMaskDstDescPtr, 2, srcDescPtr->h, srcDescPtr->w, 1);

        // Fill the ROI and dstImageSize values required for resize api call
        RpptImagePatchPtr internalDstImgSizes = reinterpret_cast<RpptImagePatch*>(
            handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
        RpptROI* internalRoiTensorPtrSrc = reinterpret_cast<RpptROI*>(internalDstImgSizes + 2);
        for (Rpp32s i = 0; i < 2; i++) {
            internalDstImgSizes[i] = {srcDescPtr->w, srcDescPtr->h};
            internalRoiTensorPtrSrc[i].xywhROI = {{0, 0}, 1920, 1080};
        }
        RpptInterpolationType interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;

        // Set batch size to 2 for computing resized alpha and intensity masks using single resize
        // call
        rppSetBatchSize(rppHandle, 2);

        // Compute the mask size
        Rpp32u maskSize = FOG_MAX_HEIGHT * FOG_MAX_WIDTH;
        Rpp32u maskSizeInBytes = maskSize * sizeof(Rpp32f);

        // Copying fog alpha mask from host to device asynchronously
        Rpp32f *d_fogAlphaMaskPtr, *d_resizedFogAlphaMaskPtr, *d_resizedFogIntensityMaskPtr;
        d_fogAlphaMaskPtr =
            reinterpret_cast<Rpp32f*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        d_resizedFogAlphaMaskPtr = reinterpret_cast<Rpp32f*>(d_fogAlphaMaskPtr + (2 * maskSize));
        d_resizedFogIntensityMaskPtr = d_resizedFogAlphaMaskPtr + (srcDescPtr->h * srcDescPtr->w);

        RPP_HIP_RETURN_IF_ERROR(hipMemcpyAsync(d_fogAlphaMaskPtr, &fogMask_1920_1080[0],
                                               maskSizeInBytes * 2, hipMemcpyHostToDevice,
                                               handle.GetStream()));

        // Resize the mask to the maximum size present in the batch
        rppt_resize(d_fogAlphaMaskPtr, fogMaskSrcDescPtr, d_resizedFogAlphaMaskPtr,
                    fogMaskDstDescPtr, internalDstImgSizes, interpolationType,
                    internalRoiTensorPtrSrc, roiType, rppHandle, RPP_HIP_BACKEND);
        RPP_HIP_RETURN_IF_ERROR(hipStreamSynchronize(handle.GetStream()));

        // Resetting the batch size in handle to match the user passed batch size
        rppSetBatchSize(rppHandle, srcDescPtr->n);

        Rpp32u *maskLocOffsetX, *maskLocOffsetY;
        maskLocOffsetX = reinterpret_cast<Rpp32u*>(internalRoiTensorPtrSrc + 2);
        maskLocOffsetY = reinterpret_cast<Rpp32u*>(maskLocOffsetX + srcDescPtr->n);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_fog_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                                d_resizedFogAlphaMaskPtr, d_resizedFogIntensityMaskPtr,
                                intensityFactor, grayFactor, maskLocOffsetX, maskLocOffsetY,
                                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_fog_tensor(
                reinterpret_cast<half*>((static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes)),
                srcDescPtr,
                reinterpret_cast<half*>((static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes)),
                dstDescPtr, d_resizedFogAlphaMaskPtr, d_resizedFogIntensityMaskPtr, intensityFactor,
                grayFactor, maskLocOffsetX, maskLocOffsetY, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_fog_tensor(reinterpret_cast<Rpp32f*>(
                                    (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes)),
                                srcDescPtr,
                                reinterpret_cast<Rpp32f*>(
                                    (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes)),
                                dstDescPtr, d_resizedFogAlphaMaskPtr, d_resizedFogIntensityMaskPtr,
                                intensityFactor, grayFactor, maskLocOffsetX, maskLocOffsetY,
                                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_fog_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                                d_resizedFogAlphaMaskPtr, d_resizedFogIntensityMaskPtr,
                                intensityFactor, grayFactor, maskLocOffsetX, maskLocOffsetY,
                                roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** rain ********************/

RppStatus rppt_rain(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr, Rpp32f rainPercentage, Rpp32u rainWidth,
                    Rpp32u rainHeight, Rpp32f slantAngle, Rpp32f* alpha, RpptROIPtr roiTensorPtrSrc,
                    RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            rain_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr, rainPercentage, rainWidth, rainHeight, slantAngle,
                                   alpha, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            rain_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rainPercentage, rainWidth, rainHeight, slantAngle, alpha,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            rain_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rainPercentage, rainWidth, rainHeight, slantAngle, alpha,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            rain_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr, rainPercentage, rainWidth, rainHeight, slantAngle,
                                   alpha, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_rain_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rainPercentage,
                rainWidth, rainHeight, slantAngle, alpha, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_rain_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rainPercentage, rainWidth, rainHeight, slantAngle, alpha,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_rain_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, rainPercentage, rainWidth, rainHeight, slantAngle, alpha,
                roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_rain_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, rainPercentage,
                rainWidth, rainHeight, slantAngle, alpha, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** posterize ********************/

RppStatus rppt_posterize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr, Rpp8u* posterizeLevelBits,
                         RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                         RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    for (int i = 0; i < srcDescPtr->n; i++)
        if (posterizeLevelBits[i] > 8) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if (((srcDescPtr->dataType == RpptDataType::U8) &&
             (dstDescPtr->dataType == RpptDataType::U8)) ||
            ((srcDescPtr->dataType == RpptDataType::I8) &&
             (dstDescPtr->dataType == RpptDataType::I8))) {
            posterize_char_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                posterizeLevelBits, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            posterize_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, posterizeLevelBits, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            posterize_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, posterizeLevelBits, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (((srcDescPtr->dataType == RpptDataType::U8) &&
             (dstDescPtr->dataType == RpptDataType::U8)) ||
            ((srcDescPtr->dataType == RpptDataType::I8) &&
             (dstDescPtr->dataType == RpptDataType::I8))) {
            hip_exec_posterize_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                posterizeLevelBits, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_posterize_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, posterizeLevelBits, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_posterize_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, posterizeLevelBits, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** solarize ********************/

RppStatus rppt_solarize(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr, Rpp32f* thresholdTensor, RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    for (int i = 0; i < srcDescPtr->n; i++)
        if (thresholdTensor[i] < 0 || thresholdTensor[i] > 1) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            solarize_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                thresholdTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            solarize_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, thresholdTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            solarize_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, thresholdTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            solarize_i8_i8_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                thresholdTensor, roiTensorPtrSrc, roiType, layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_solarize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr, thresholdTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_solarize_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, thresholdTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_solarize_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, thresholdTensor, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_solarize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr, thresholdTensor, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** snow ********************/

RppStatus rppt_snow(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr, Rpp32f* brightnessCoefficient, Rpp32f* snowThreshold,
                    Rpp32s* darkMode, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                    rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    for (int i = 0; i < srcDescPtr->n; i++) {
        if (brightnessCoefficient[i] <= 1.0f || brightnessCoefficient[i] > 4.0f)
            return RPP_ERROR_INVALID_ARGUMENTS;
        if (snowThreshold[i] <= 0.0f || snowThreshold[i] > 1.0f) return RPP_ERROR_INVALID_ARGUMENTS;
        if (darkMode[i] != 0 && darkMode[i] != 1) return RPP_ERROR_INVALID_ARGUMENTS;
    }

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            snow_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr, brightnessCoefficient, snowThreshold, darkMode,
                                   roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            snow_f16_f16_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, brightnessCoefficient, snowThreshold, darkMode, roiTensorPtrSrc,
                roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            snow_f32_f32_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, brightnessCoefficient, snowThreshold, darkMode, roiTensorPtrSrc,
                roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            snow_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr, brightnessCoefficient, snowThreshold, darkMode,
                                   roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_snow_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr, brightnessCoefficient, snowThreshold, darkMode,
                                 roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_snow_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, brightnessCoefficient, snowThreshold, darkMode, roiTensorPtrSrc,
                roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_snow_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, brightnessCoefficient, snowThreshold, darkMode, roiTensorPtrSrc,
                roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_snow_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr, brightnessCoefficient, snowThreshold, darkMode,
                                 roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** channel_dropout ********************/

RppStatus rppt_channel_dropout(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr, Rpp8u* dropoutTensor,
                               RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                               rppHandle_t rppHandle, RppBackend executionBackend) {
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            channel_dropout_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dropoutTensor,
                roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            channel_dropout_host_tensor(
                (Rpp16f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                (Rpp16f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                dropoutTensor, roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            channel_dropout_host_tensor(
                (Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes), srcDescPtr,
                (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), dstDescPtr,
                dropoutTensor, roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            channel_dropout_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dropoutTensor,
                roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_channel_dropout_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dropoutTensor,
                roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_channel_dropout_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dropoutTensor, roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_channel_dropout_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, dropoutTensor, roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_channel_dropout_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr, dropoutTensor,
                roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** cutout_dropout ********************/

RppStatus rppt_cutout_dropout(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                              RppPtr_t colorsTensor, Rpp32u* numBoxesTensor,
                              RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                              rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            erase_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                              anchorBoxInfoTensor, static_cast<Rpp8u*>(colorsTensor),
                              numBoxesTensor, roiTensorPtrSrc, roiType, layoutParams,
                              rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            erase_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp16f*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            erase_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp32f*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            erase_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                              anchorBoxInfoTensor, static_cast<Rpp8s*>(colorsTensor),
                              numBoxesTensor, roiTensorPtrSrc, roiType, layoutParams,
                              rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_erase_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, static_cast<Rpp8u*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_erase_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<half*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_erase_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp32f*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_erase_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, static_cast<Rpp8s*>(colorsTensor), numBoxesTensor,
                roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** grid_dropout ********************/

RppStatus rppt_grid_dropout(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                            Rpp32u boxesInEachImage, Rpp32u maxHoleW, Rpp32u maxHoleH,
                            RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle,
                            RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            grid_dropout_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH, roiTensorPtrSrc, roiType,
                layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            grid_dropout_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH,
                roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            grid_dropout_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH,
                roiTensorPtrSrc, roiType, layoutParams, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            grid_dropout_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH, roiTensorPtrSrc, roiType,
                layoutParams, rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_grid_dropout_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH, roiTensorPtrSrc, roiType,
                rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_grid_dropout_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH,
                roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_grid_dropout_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH,
                roiTensorPtrSrc, roiType, rpp::deref(rppHandle));
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_grid_dropout_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH, roiTensorPtrSrc, roiType,
                rpp::deref(rppHandle));
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** random_erase ********************/

RppStatus rppt_random_erase(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                            RppPtr_t noiseBuffer, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                            rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            random_erase_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, static_cast<Rpp8u*>(noiseBuffer), roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            random_erase_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp16f*>(noiseBuffer), roiTensorPtrSrc,
                roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            random_erase_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp32f*>(noiseBuffer), roiTensorPtrSrc,
                roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            random_erase_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, static_cast<Rpp8s*>(noiseBuffer), roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_random_erase_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, static_cast<Rpp8u*>(noiseBuffer), roiTensorPtrSrc, roiType,
                handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_random_erase_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<half*>(noiseBuffer), roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_random_erase_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, static_cast<Rpp32f*>(noiseBuffer), roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_random_erase_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, static_cast<Rpp8s*>(noiseBuffer), roiTensorPtrSrc, roiType,
                handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** coarse_dropout ********************/

RppStatus rppt_coarse_dropout(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                              RpptDescPtr dstDescPtr, RpptRoiLtrb* anchorBoxInfoTensor,
                              Rpp32u* numBoxesTensor, Rpp32u maxBoxesPerImage,
                              RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                              rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != dstDescPtr->dataType) return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if ((srcDescPtr->layout != RpptLayout::NCHW) && (srcDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstDescPtr->layout != RpptLayout::NCHW) && (dstDescPtr->layout != RpptLayout::NHWC))
        return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcDescPtr->c != dstDescPtr->c) return RPP_ERROR_INVALID_ARGUMENTS;
    // Validate maxBoxesPerImage
    if (maxBoxesPerImage == 0) return RPP_ERROR_INVALID_ARGUMENTS;

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            coarse_dropout_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, numBoxesTensor, maxBoxesPerImage, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            coarse_dropout_host_tensor(
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, numBoxesTensor, maxBoxesPerImage, roiTensorPtrSrc,
                roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            coarse_dropout_host_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, numBoxesTensor, maxBoxesPerImage, roiTensorPtrSrc,
                roiType, layoutParams, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            coarse_dropout_host_tensor(
                static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                anchorBoxInfoTensor, numBoxesTensor, maxBoxesPerImage, roiTensorPtrSrc, roiType,
                layoutParams, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_coarse_dropout_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr, anchorBoxInfoTensor, numBoxesTensor,
                                           maxBoxesPerImage, roiTensorPtrSrc, roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F16) &&
                   (dstDescPtr->dataType == RpptDataType::F16)) {
            hip_exec_coarse_dropout_tensor(
                reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, numBoxesTensor, maxBoxesPerImage, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::F32) &&
                   (dstDescPtr->dataType == RpptDataType::F32)) {
            hip_exec_coarse_dropout_tensor(
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                srcDescPtr,
                reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                dstDescPtr, anchorBoxInfoTensor, numBoxesTensor, maxBoxesPerImage, roiTensorPtrSrc,
                roiType, handle);
        } else if ((srcDescPtr->dataType == RpptDataType::I8) &&
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            hip_exec_coarse_dropout_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr, anchorBoxInfoTensor, numBoxesTensor,
                                           maxBoxesPerImage, roiTensorPtrSrc, roiType, handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}
