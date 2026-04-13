// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_gpu_ref/GpuFpReferenceValidation.hpp>

#include <cstdint>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefValidatorHelpers.hpp>
#include <hipdnn_gpu_ref/detail/HipRtcTypeName.hpp>
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
    if(reference.elementCount() != implementation.elementCount()
       || reference.dims() != implementation.dims())
    {
        return false;
    }

    if(reference.elementCount() == 0)
    {
        return true;
    }

    // GPU kernel uses linear indexing — only valid for packed (contiguous) tensors.
    if(!reference.isPacked() || !implementation.isPacked())
    {
        throw std::runtime_error("GPU validator requires packed (contiguous) tensors");
    }

    return gpuAllClose(reference, implementation);
}

template <class T>
bool GpuFpReferenceValidation<T>::gpuAllClose(
    hipdnn_data_sdk::utilities::ITensor& reference,
    hipdnn_data_sdk::utilities::ITensor& implementation) const
{
    auto totalElements = static_cast<int64_t>(reference.elementCount());

    // Allocate single failure flag using MigratableMemory
    hipdnn_data_sdk::utilities::MigratableMemory<int> flagBuf(1);
    flagBuf.hostData()[0] = 0;

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

    detail::launchValidatorKernel(kernel.function(), totalElements, args);

    // Read back single failure flag
    flagBuf.markDeviceModified();
    auto hostFlag = flagBuf.hostData()[0];

    return hostFlag == 0;
}

// --- Explicit template instantiations ---

template class GpuFpReferenceValidation<float>;
template class GpuFpReferenceValidation<hipdnn_data_sdk::types::half>;
template class GpuFpReferenceValidation<hipdnn_data_sdk::types::bfloat16>;
template class GpuFpReferenceValidation<double>;

} // namespace hipdnn_gpu_ref
