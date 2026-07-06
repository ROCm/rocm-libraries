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

#include "rppt_tensor_bitwise_operations.h"

#include "host_tensor_executors.hpp"
#include "rppdefs.h"
#include "rppt_validate.hpp"

#ifdef GPU_SUPPORT
#include "hip_tensor_executors.hpp"
#endif  // GPU_SUPPORT

/******************** bitwise AND ********************/

RppStatus rppt_bitwise_and(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType, rppHandle_t rppHandle,
                           RppBackend executionBackend) {
    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;
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
            bitwise_and_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_bitwise_and_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                        static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr, roiTensorPtrSrc, roiType, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** bitwise XOR ********************/

RppStatus rppt_bitwise_xor(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType, rppHandle_t rppHandle,
                           RppBackend executionBackend) {
    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;
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
            bitwise_xor_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_bitwise_xor_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                        static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr, roiTensorPtrSrc, roiType, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** bitwise OR ********************/

RppStatus rppt_bitwise_or(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType, rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;
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
            bitwise_or_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_bitwise_or_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                       static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr, roiTensorPtrSrc, roiType, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** bitwise NOT ********************/

RppStatus rppt_bitwise_not(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                           rppHandle_t rppHandle, RppBackend executionBackend) {
    if (srcDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstDescPtr->dataType != RpptDataType::U8) return RPP_ERROR_INVALID_DST_DATATYPE;
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
            bitwise_not_u8_u8_host_tensor(
                static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes, srcDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes, dstDescPtr,
                roiTensorPtrSrc, roiType, layoutParams, handle);
        }
        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if ((srcDescPtr->dataType == RpptDataType::U8) &&
            (dstDescPtr->dataType == RpptDataType::U8)) {
            hip_exec_bitwise_not_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr, roiTensorPtrSrc, roiType, handle);
        }
        return RPP_SUCCESS;
    }
#endif
    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor AND tensor ********************/

RppStatus rppt_tensor_and_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                 RpptGenericDescPtr srcPtr1GenericDescPtr,
                                 RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 RpptBroadcastMode broadcastMode, Rpp32u* roiTensorSrc1,
                                 Rpp32u* roiTensorSrc2, rppHandle_t rppHandle,
                                 RppBackend executionBackend) {
    // Validate that all three data types match
    if (srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    // When broadcast is disabled, ensure that all tensor shapes and ROIs match
    if (broadcastMode == RpptBroadcastMode::RPP_BROADCAST_DISABLE) {
        // Validate number of dimensions match across all tensors
        if ((srcPtr1GenericDescPtr->numDims != srcPtr2GenericDescPtr->numDims) ||
            (srcPtr1GenericDescPtr->numDims != dstGenericDescPtr->numDims)) {
            return RPP_ERROR_INVALID_ARGUMENTS;
        }

        // Validate that each dimension size matches across all tensors
        for (Rpp32u dim = 0; dim < srcPtr1GenericDescPtr->numDims; dim++) {
            if ((srcPtr1GenericDescPtr->dims[dim] != srcPtr2GenericDescPtr->dims[dim]) ||
                (srcPtr1GenericDescPtr->dims[dim] != dstGenericDescPtr->dims[dim])) {
                return RPP_ERROR_INVALID_ARGUMENTS;
            }
        }

        // Validate ROI tensor presence consistency when not broadcasting
        if ((roiTensorSrc1 == nullptr) != (roiTensorSrc2 == nullptr)) {
            return RPP_ERROR_INVALID_ARGUMENTS;
        }
    }

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
             (dstGenericDescPtr->dataType == RpptDataType::U8)) ||
            ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
             (dstGenericDescPtr->dataType == RpptDataType::I8))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_AND, broadcastMode, roiTensorSrc1, roiTensorSrc2, handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U16)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I16))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_AND, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U32)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I32))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_AND, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
             (dstGenericDescPtr->dataType == RpptDataType::U8)) ||
            ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
             (dstGenericDescPtr->dataType == RpptDataType::I8))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_AND, broadcastMode, roiTensorSrc1, roiTensorSrc2, handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U16)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I16))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_AND, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U32)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I32))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_AND, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/********************tensor OR tensor ********************/

RppStatus rppt_tensor_or_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                RpptGenericDescPtr srcPtr1GenericDescPtr,
                                RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                                RpptGenericDescPtr dstGenericDescPtr,
                                RpptBroadcastMode broadcastMode, Rpp32u* roiTensorSrc1,
                                Rpp32u* roiTensorSrc2, rppHandle_t rppHandle,
                                RppBackend executionBackend) {
    // Validate that all three data types match
    if (srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    // When broadcast is disabled, ensure that all tensor shapes and ROIs match
    if (broadcastMode == RpptBroadcastMode::RPP_BROADCAST_DISABLE) {
        // Validate number of dimensions match across all tensors
        if ((srcPtr1GenericDescPtr->numDims != srcPtr2GenericDescPtr->numDims) ||
            (srcPtr1GenericDescPtr->numDims != dstGenericDescPtr->numDims)) {
            return RPP_ERROR_INVALID_ARGUMENTS;
        }

        // Validate that each dimension size matches across all tensors
        for (Rpp32u dim = 0; dim < srcPtr1GenericDescPtr->numDims; dim++) {
            if ((srcPtr1GenericDescPtr->dims[dim] != srcPtr2GenericDescPtr->dims[dim]) ||
                (srcPtr1GenericDescPtr->dims[dim] != dstGenericDescPtr->dims[dim])) {
                return RPP_ERROR_INVALID_ARGUMENTS;
            }
        }

        // Validate ROI tensor presence consistency when not broadcasting
        if ((roiTensorSrc1 == nullptr) != (roiTensorSrc2 == nullptr)) {
            return RPP_ERROR_INVALID_ARGUMENTS;
        }
    }

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
             (dstGenericDescPtr->dataType == RpptDataType::U8)) ||
            ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
             (dstGenericDescPtr->dataType == RpptDataType::I8))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_OR, broadcastMode, roiTensorSrc1, roiTensorSrc2, handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U16)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I16))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_OR, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U32)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I32))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_OR, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
             (dstGenericDescPtr->dataType == RpptDataType::U8)) ||
            ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
             (dstGenericDescPtr->dataType == RpptDataType::I8))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_OR, broadcastMode, roiTensorSrc1, roiTensorSrc2, handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U16)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I16))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_OR, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U32)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I32))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_OR, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}

/******************** tensor XOR tensor ********************/

RppStatus rppt_tensor_xor_tensor(RppPtr_t srcPtr1, RppPtr_t srcPtr2,
                                 RpptGenericDescPtr srcPtr1GenericDescPtr,
                                 RpptGenericDescPtr srcPtr2GenericDescPtr, RppPtr_t dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 RpptBroadcastMode broadcastMode, Rpp32u* roiTensorSrc1,
                                 Rpp32u* roiTensorSrc2, rppHandle_t rppHandle,
                                 RppBackend executionBackend) {
    // Validate that all three data types match
    if (srcPtr1GenericDescPtr->dataType != srcPtr2GenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    if (srcPtr1GenericDescPtr->dataType != dstGenericDescPtr->dataType)
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;

    // When broadcast is disabled, ensure that all tensor shapes and ROIs match
    if (broadcastMode == RpptBroadcastMode::RPP_BROADCAST_DISABLE) {
        // Validate number of dimensions match across all tensors
        if ((srcPtr1GenericDescPtr->numDims != srcPtr2GenericDescPtr->numDims) ||
            (srcPtr1GenericDescPtr->numDims != dstGenericDescPtr->numDims)) {
            return RPP_ERROR_INVALID_ARGUMENTS;
        }

        // Validate that each dimension size matches across all tensors
        for (Rpp32u dim = 0; dim < srcPtr1GenericDescPtr->numDims; dim++) {
            if ((srcPtr1GenericDescPtr->dims[dim] != srcPtr2GenericDescPtr->dims[dim]) ||
                (srcPtr1GenericDescPtr->dims[dim] != dstGenericDescPtr->dims[dim])) {
                return RPP_ERROR_INVALID_ARGUMENTS;
            }
        }

        // Validate ROI tensor presence consistency when not broadcasting
        if ((roiTensorSrc1 == nullptr) != (roiTensorSrc2 == nullptr)) {
            return RPP_ERROR_INVALID_ARGUMENTS;
        }
    }

    rpp::Handle& handle = rpp::deref(rppHandle);
    RppBackend handleBackend = handle.GetBackend();

    if (executionBackend == RppBackend::RPP_HOST_BACKEND) {
        if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
             (dstGenericDescPtr->dataType == RpptDataType::U8)) ||
            ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
             (dstGenericDescPtr->dataType == RpptDataType::I8))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_XOR, broadcastMode, roiTensorSrc1, roiTensorSrc2, handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U16)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I16))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_XOR, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U32)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I32))) {
            tensor_binary_bitwise_op_dispatch_host_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_XOR, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#ifdef GPU_SUPPORT
    else if ((handleBackend == RppBackend::RPP_HIP_BACKEND) &&
             (executionBackend == RppBackend::RPP_HIP_BACKEND)) {
        if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U8) &&
             (dstGenericDescPtr->dataType == RpptDataType::U8)) ||
            ((srcPtr1GenericDescPtr->dataType == RpptDataType::I8) &&
             (dstGenericDescPtr->dataType == RpptDataType::I8))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                static_cast<Rpp8u*>(srcPtr1) + srcPtr1GenericDescPtr->offsetInBytes,
                static_cast<Rpp8u*>(srcPtr2) + srcPtr2GenericDescPtr->offsetInBytes,
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes, dstGenericDescPtr,
                RPP_TENSOR_OP_XOR, broadcastMode, roiTensorSrc1, roiTensorSrc2, handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U16)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I16) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I16))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_XOR, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else if (((srcPtr1GenericDescPtr->dataType == RpptDataType::U32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::U32)) ||
                   ((srcPtr1GenericDescPtr->dataType == RpptDataType::I32) &&
                    (dstGenericDescPtr->dataType == RpptDataType::I32))) {
            tensor_binary_bitwise_op_dispatch_gpu_tensor(
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr1) +
                                          srcPtr1GenericDescPtr->offsetInBytes),
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(srcPtr2) +
                                          srcPtr2GenericDescPtr->offsetInBytes),
                srcPtr1GenericDescPtr, srcPtr2GenericDescPtr,
                reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(dstPtr) +
                                          dstGenericDescPtr->offsetInBytes),
                dstGenericDescPtr, RPP_TENSOR_OP_XOR, broadcastMode, roiTensorSrc1, roiTensorSrc2,
                handle);
        } else
            return RPP_ERROR_NOT_IMPLEMENTED;

        return RPP_SUCCESS;
    }
#endif

    return RPP_ERROR_INCOMPATIBLE_BACKEND;
}
