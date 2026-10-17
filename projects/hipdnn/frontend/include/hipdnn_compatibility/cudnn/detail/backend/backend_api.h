// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/engine_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/execution_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/operation_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/operation_graph_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/parameter_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/tensor_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/type_mapping.h>
#include <hipdnn_data_sdk/Visibility.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

// Per-thread message of the last failed shim call. A trivially initialized
// buffer avoids TLS init guards and thread-exit destructors (fragile across the
// MSVC DLL ABI). HIPDNN_HIDDEN here and on the entry-point layer below keeps one
// buffer, and the code that writes it, per shared object.
HIPDNN_HIDDEN inline char* lastErrorBuffer()
{
    static thread_local std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> s_buffer{};
    return s_buffer.data();
}

// Writes "<entryPoint>: <message>", truncated to the buffer.
HIPDNN_HIDDEN inline void recordLastError(const char* entryPoint, const char* message) noexcept
{
    constexpr std::size_t CAPACITY = HIPDNN_ERROR_STRING_MAX_LENGTH - 1;
    char* buffer = lastErrorBuffer();
    std::size_t length = 0;
    const auto append = [&](const char* text) {
        const std::size_t count = std::min(std::strlen(text), CAPACITY - length);
        std::memcpy(buffer + length, text, count);
        length += count;
    };
    append(entryPoint);
    append(": ");
    append(message);
    buffer[length] = '\0';
}

// The single catch ladder behind every entry point. `body` must not call a
// public cudnnBackend* function; use the C++ implementation instead.
template <typename Body>
HIPDNN_HIDDEN cudnnStatus_t guardedCall(const char* entryPoint, Body&& body) noexcept
{
    try
    {
        std::forward<Body>(body)();
        return CUDNN_STATUS_SUCCESS;
    }
    catch(const ShimError& e)
    {
        recordLastError(entryPoint, e.what());
        return e.status();
    }
    catch(const std::bad_alloc&)
    {
        recordLastError(entryPoint, "allocation failed");
        return CUDNN_STATUS_ALLOC_FAILED;
    }
    catch(const std::exception& e)
    {
        recordLastError(entryPoint, e.what());
        return CUDNN_STATUS_INTERNAL_ERROR;
    }
    catch(...)
    {
        recordLastError(entryPoint, "unknown exception");
        return CUDNN_STATUS_INTERNAL_ERROR;
    }
}

// Maps every descriptor type to its class. Later phases move a type out of the
// storage-only group into its own `case` returning the validating class.
HIPDNN_HIDDEN inline std::shared_ptr<Descriptor>
    newDescriptorObject(cudnnBackendDescriptorType_t type)
{
    switch(type)
    {
    case CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
        return std::make_shared<ConvolutionDescriptor>();
    case CUDNN_BACKEND_MATMUL_DESCRIPTOR:
        return std::make_shared<MatmulDescriptor>();
    case CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
        return std::make_shared<PointwiseDescriptor>();
    case CUDNN_BACKEND_REDUCTION_DESCRIPTOR:
        return std::make_shared<ReductionDescriptor>();
    case CUDNN_BACKEND_RESAMPLE_DESCRIPTOR:
        return std::make_shared<ResampleDescriptor>();
    case CUDNN_BACKEND_RNG_DESCRIPTOR:
        return std::make_shared<RngDescriptor>();
    case CUDNN_BACKEND_TENSOR_DESCRIPTOR:
        return std::make_shared<TensorDescriptor>();
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
        return std::make_shared<ConvolutionForwardOperation>();
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
        return std::make_shared<ConvolutionBackwardDataOperation>();
    case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
        return std::make_shared<ConvolutionBackwardFilterOperation>();
    case CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
        return std::make_shared<MatmulOperation>();
    case CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
        return std::make_shared<NormBackwardOperation>();
    case CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
        return std::make_shared<NormForwardOperation>();
    case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
        return std::make_shared<PointwiseOperation>();
    case CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
        return std::make_shared<ReductionOperation>();
    case CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
        return std::make_shared<ResampleBackwardOperation>();
    case CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
        return std::make_shared<ResampleForwardOperation>();
    case CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
        return std::make_shared<OperationGraphDescriptor>();
    case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
        return std::make_shared<EngineHeuristicDescriptor>();
    case CUDNN_BACKEND_ENGINE_DESCRIPTOR:
        return std::make_shared<EngineDescriptor>();
    case CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
        return std::make_shared<EngineConfigDescriptor>();
    case CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
        return std::make_shared<KnobChoiceDescriptor>();
    case CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
        return std::make_shared<KnobInfoDescriptor>();
    case CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
        return std::make_shared<ExecutionPlanDescriptor>();
    case CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
        return std::make_shared<VariantPackDescriptor>();
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR:
    case CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR:
        return std::make_shared<UnsupportedOperation>(type);
    case CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
    case CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
        return std::make_shared<StorageOnlyDescriptor>(type);
    default:
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        "unknown descriptor type " + std::to_string(static_cast<int>(type)));
    }
}

HIPDNN_HIDDEN inline cudnnStatus_t createDescriptor(cudnnBackendDescriptorType_t descriptorType,
                                                    cudnnBackendDescriptor_t* descriptor) noexcept
{
    return guardedCall("cudnnBackendCreateDescriptor", [&] {
        if(descriptor == nullptr)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, "descriptor is null");
        }
        *descriptor = makeHandle(newDescriptorObject(descriptorType));
    });
}

HIPDNN_HIDDEN inline cudnnStatus_t destroyDescriptor(cudnnBackendDescriptor_t descriptor) noexcept
{
    // Upstream destroys unconditionally, including after a failed create, so a
    // null destroy must not clobber the error that create recorded.
    if(descriptor == nullptr)
    {
        return CUDNN_STATUS_BAD_PARAM;
    }
    return guardedCall("cudnnBackendDestroyDescriptor", [&] {
        Handle& handle = requireHandle(descriptor, "descriptor");
        handle.magic = 0;
        delete &handle;
    });
}

// SET, GET and finalize pin the descriptor, so an in-place GET that re-points
// its own handle cannot destroy the object whose member function is running.
HIPDNN_HIDDEN inline cudnnStatus_t setAttribute(cudnnBackendDescriptor_t descriptor,
                                                cudnnBackendAttributeName_t attributeName,
                                                cudnnBackendAttributeType_t attributeType,
                                                std::int64_t elementCount,
                                                const void* arrayOfElements) noexcept
{
    return guardedCall("cudnnBackendSetAttribute", [&] {
        const std::shared_ptr<Descriptor> self = requireHandle(descriptor, "descriptor").object;
        self->setAttribute(attributeName, attributeType, elementCount, arrayOfElements);
    });
}

HIPDNN_HIDDEN inline cudnnStatus_t getAttribute(cudnnBackendDescriptor_t descriptor,
                                                cudnnBackendAttributeName_t attributeName,
                                                cudnnBackendAttributeType_t attributeType,
                                                std::int64_t requestedElementCount,
                                                std::int64_t* elementCount,
                                                void* arrayOfElements) noexcept
{
    return guardedCall("cudnnBackendGetAttribute", [&] {
        const std::shared_ptr<Descriptor> self = requireHandle(descriptor, "descriptor").object;
        self->getAttribute(
            attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);
    });
}

HIPDNN_HIDDEN inline cudnnStatus_t finalize(cudnnBackendDescriptor_t descriptor) noexcept
{
    return guardedCall("cudnnBackendFinalize", [&] {
        const std::shared_ptr<Descriptor> self = requireHandle(descriptor, "descriptor").object;
        self->finalize();
    });
}

// Hot path: no allocation and exactly one hipDNN call (the variant pack was
// built at its finalize).
HIPDNN_HIDDEN inline cudnnStatus_t execute(cudnnHandle_t handle,
                                           cudnnBackendDescriptor_t executionPlan,
                                           cudnnBackendDescriptor_t variantPack) noexcept
{
    return guardedCall("cudnnBackendExecute", [&] {
        const std::shared_ptr<Descriptor> planObject
            = requireHandle(executionPlan, "executionPlan").object;
        const std::shared_ptr<Descriptor> packObject
            = requireHandle(variantPack, "variantPack").object;
        const auto& plan = descriptorCast<ExecutionPlanDescriptor>(
            *planObject, CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        const auto& pack = descriptorCast<VariantPackDescriptor>(
            *packObject, CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        if(!plan.isFinalized() || !pack.isFinalized())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            !plan.isFinalized() ? "executionPlan is not finalized"
                                                : "variantPack is not finalized");
        }
        // hipDNN would run the plan's context on this handle's stream without
        // noticing the mismatch.
        if(handle != plan.contextHandle())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "handle differs from the handle the execution plan was built for");
        }
        checkHipdnn(hipdnn_frontend::detail::hipdnnBackend()->backendExecute(
                        handle, plan.hipdnnPlan(), pack.hipdnnVariantPack()),
                    "executing the hipDNN plan");
    });
}

// Copies and clears the shim message; when there is none, falls back to the
// hipDNN backend's message (covers v9 entry points such as cudnnCreate).
HIPDNN_HIDDEN inline void getLastErrorString(char* message, std::size_t size) noexcept
{
    if(message == nullptr || size == 0)
    {
        return;
    }
    char* buffer = lastErrorBuffer();
    if(buffer[0] != '\0')
    {
        const std::size_t length = std::min(std::strlen(buffer), size - 1);
        std::memcpy(message, buffer, length);
        message[length] = '\0';
        buffer[0] = '\0';
        return;
    }
    message[0] = '\0';
    try
    {
        hipdnn_frontend::detail::hipdnnBackend()->getLastErrorString(message, size);
    }
    catch(...)
    {
        message[0] = '\0';
    }
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
