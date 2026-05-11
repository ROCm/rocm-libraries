// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_gpu_ref/GpuIntReferenceValidation.hpp>

#include <cstdint>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefValidatorHelpers.hpp>
#include <hipdnn_gpu_ref/detail/HipRtcTypeName.hpp>
#include <stdexcept>

namespace hipdnn_gpu_ref
{

template <class T>
bool GpuIntReferenceValidation<T>::allClose(
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

    return gpuExact(reference, implementation);
}

template <class T>
bool GpuIntReferenceValidation<T>::gpuExact(
    hipdnn_data_sdk::utilities::ITensor& reference,
    hipdnn_data_sdk::utilities::ITensor& implementation) const
{
    auto totalElements = static_cast<int64_t>(reference.elementCount());

    // Allocate single failure flag using MigratableMemory
    hipdnn_data_sdk::utilities::MigratableMemory<int> flagBuf(1);
    flagBuf.hostData()[0] = 0;

    auto* refPtr = reference.rawDeviceData();
    auto* implPtr = implementation.rawDeviceData();

    auto defines = detail::buildValidatorDefines(detail::HipRtcTypeName<T>::VALUE, "double");

    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefValidator.cpp", defines, "validateExact");

    detail::ValidatorArgs args{};
    args.reference = refPtr;
    args.implementation = implPtr;
    args.failureFlag = static_cast<int*>(flagBuf.deviceData());
    args.totalElements = totalElements;
    args.absoluteTolerance = 0.0;
    args.relativeTolerance = 0.0;

    // Populate stride/dim fields for non-contiguous tensors.
    if(!reference.isPacked() || !implementation.isPacked())
    {
        const auto& refStrides = reference.strides();
        const auto& implStrides = implementation.strides();
        const auto& dims = reference.dims();
        auto ndim = dims.size();
        if(ndim > 8)
        {
            throw std::runtime_error("GPU validator supports up to 8 dimensions, got "
                                     + std::to_string(ndim));
        }
        args.ndim = static_cast<int>(ndim);
        for(size_t d = 0; d < ndim; ++d)
        {
            args.refStrides[d] = static_cast<long long>(refStrides[d]);
            args.implStrides[d] = static_cast<long long>(implStrides[d]);
            args.dims[d] = static_cast<long long>(dims[d]);
        }
    }

    detail::launchValidatorKernel(kernel.function(), totalElements, args);

    // Read back single failure flag
    flagBuf.markDeviceModified();
    auto hostFlag = flagBuf.hostData()[0];

    return hostFlag == 0;
}

// --- Explicit template instantiations ---

template class GpuIntReferenceValidation<int8_t>;
template class GpuIntReferenceValidation<uint8_t>;
template class GpuIntReferenceValidation<int32_t>;

} // namespace hipdnn_gpu_ref
