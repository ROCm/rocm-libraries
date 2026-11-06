// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceValidation.hpp>

#include <cstdint>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefValidatorHelpers.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <stdexcept>

namespace hipdnn_gpu_ref
{

template <class T>
GpuFpReferenceValidation<T>::GpuFpReferenceValidation(float absoluteTolerance,
                                                      float relativeTolerance)
    : _absoluteTolerance(absoluteTolerance)
    , _relativeTolerance(relativeTolerance)
{
    if(absoluteTolerance < 0.0f || relativeTolerance < 0.0f || std::isnan(absoluteTolerance)
       || std::isnan(relativeTolerance) || std::isinf(absoluteTolerance)
       || std::isinf(relativeTolerance))
    {
        throw std::invalid_argument("Tolerances must be finite and non-negative");
    }
}

template <class T>
bool GpuFpReferenceValidation<T>::allClose(
    hipdnn_data_sdk::utilities::ITensor& reference,
    hipdnn_data_sdk::utilities::ITensor& implementation) const
{
    const auto raggedLayout = reference.raggedIterationInfo();
    if(reference.elementCount() != implementation.elementCount()
       || reference.dims() != implementation.dims()
       || raggedLayout != implementation.raggedIterationInfo())
    {
        return false;
    }

    if(reference.elementCount() == 0)
    {
        return true;
    }

    return gpuAllClose(reference, implementation, raggedLayout);
}

template <class T>
bool GpuFpReferenceValidation<T>::gpuAllClose(
    hipdnn_data_sdk::utilities::ITensor& reference,
    hipdnn_data_sdk::utilities::ITensor& implementation,
    const std::optional<hipdnn_data_sdk::utilities::RaggedIterationInfo>& raggedLayout) const
{
    auto totalElements = static_cast<int64_t>(reference.elementCount());

    // [0] is the failure flag; [1] is the ragged compared flag.
    hipdnn_data_sdk::utilities::MigratableMemory<int> flagBuf(2);
    flagBuf.hostData()[0] = 0;
    flagBuf.hostData()[1] = 0;

    // Get device pointers — triggers host→device migration if needed
    auto* refPtr = reference.rawDeviceData();
    auto* implPtr = implementation.rawDeviceData();

    // Build defines and compile kernel
    auto defines = detail::buildValidatorDefines(detail::HipRtcTypeName<T>::VALUE, "double");

    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefValidator.cpp", defines, "validateAllClose");

    // Build args
    detail::ValidatorArgs args{};
    args.reference = refPtr;
    args.implementation = implPtr;
    args.failureFlag = static_cast<int*>(flagBuf.deviceData());
    args.totalElements = totalElements;
    args.absoluteTolerance = static_cast<double>(_absoluteTolerance);
    args.relativeTolerance = static_cast<double>(_relativeTolerance);

    detail::setStridedLayout(args, reference, implementation);

    hipdnn_data_sdk::utilities::MigratableMemory<long long> rowOffsetsBuf;
    if(raggedLayout.has_value())
    {
        detail::setRaggedLayout(args, *raggedLayout, reference.dims(), rowOffsetsBuf);
        args.comparedFlag = static_cast<int*>(flagBuf.deviceData()) + 1;
    }

    detail::launchValidatorKernel(kernel.function(), args.totalElements, args);

    flagBuf.markDeviceModified();
    const auto* hostFlags = flagBuf.hostData();
    if(hostFlags[0] != 0)
    {
        return false;
    }

    if(raggedLayout.has_value() && hostFlags[1] == 0)
    {
        HIPDNN_SDK_LOG_ERROR("No element of the ragged reference was compared: every "
                             "in-block reference value is NaN.");
        return false;
    }

    return true;
}

// --- Explicit template instantiations ---

template class GpuFpReferenceValidation<float>;
template class GpuFpReferenceValidation<hipdnn_data_sdk::types::half>;
template class GpuFpReferenceValidation<hipdnn_data_sdk::types::bfloat16>;
template class GpuFpReferenceValidation<double>;

} // namespace hipdnn_gpu_ref
