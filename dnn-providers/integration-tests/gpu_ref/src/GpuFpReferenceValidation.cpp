// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_gpu_ref/GpuFpReferenceValidation.hpp>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_gpu_ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

// Shared argument struct — single definition used by both host and device (HipRTC).
#include <GpuRefValidatorArgs.h> // NOLINT(misc-include-cleaner)

std::vector<std::string> buildValidatorDefines(const char* dataType, const char* computeType)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DDATA_TYPE=") + dataType);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + computeType);
    return defines;
}

void launchValidatorKernel(hipFunction_t function, int64_t totalElements, ValidatorArgs& args)
{
    const int64_t blockSize = 256;
    auto gridSize = (totalElements + blockSize - 1) / blockSize;

    if(gridSize > static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error("Grid size exceeds hipModuleLaunchKernel limit");
    }

    auto argsSize = sizeof(ValidatorArgs);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    throwOnHipError(hipModuleLaunchKernel(function,
                                          static_cast<unsigned int>(gridSize),
                                          1,
                                          1,
                                          static_cast<unsigned int>(blockSize),
                                          1,
                                          1,
                                          0,
                                          nullptr,
                                          nullptr,
                                          config),
                    "validateAllClose: hipModuleLaunchKernel failed");

    throwOnHipError(hipDeviceSynchronize(), "validateAllClose: hipDeviceSynchronize failed");
}

} // namespace detail

// --- GpuFpReferenceValidation<T> ---

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
    // Non-packed tensors have gaps in memory that would cause incorrect comparisons.
    if(!reference.isPacked() || !implementation.isPacked())
    {
        HIPDNN_SDK_LOG_INFO("Tensors are not packed, falling back to CPU validator for "
                            "stride-aware comparison");
        return cpuFallback(reference, implementation);
    }

    try
    {
        return gpuAllClose(reference, implementation);
    }
    catch(const std::exception& e)
    {
        HIPDNN_SDK_LOG_WARN("GPU validation failed, falling back to CPU validator: " << e.what());
        return cpuFallback(reference, implementation);
    }
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

    if(hostFlag != 0)
    {
        HIPDNN_SDK_LOG_INFO(
            "GPU validation detected failure, falling back to CPU for detailed diagnostics");
        return cpuFallback(reference, implementation);
    }

    return true;
}

template <class T>
bool GpuFpReferenceValidation<T>::cpuFallback(
    hipdnn_data_sdk::utilities::ITensor& reference,
    hipdnn_data_sdk::utilities::ITensor& implementation) const
{
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<T> cpuValidator(_absoluteTolerance,
                                                                               _relativeTolerance);
    return cpuValidator.allClose(reference, implementation);
}

// --- GpuIntReferenceValidation<T> ---

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

    if(!reference.isPacked() || !implementation.isPacked())
    {
        HIPDNN_SDK_LOG_INFO("Tensors are not packed, falling back to CPU validator for "
                            "stride-aware comparison");
        return cpuFallback(reference, implementation);
    }

    try
    {
        return gpuExact(reference, implementation);
    }
    catch(const std::exception& e)
    {
        HIPDNN_SDK_LOG_WARN("GPU validation failed, falling back to CPU validator: " << e.what());
        return cpuFallback(reference, implementation);
    }
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

    detail::launchValidatorKernel(kernel.function(), totalElements, args);

    // Read back single failure flag
    flagBuf.markDeviceModified();
    auto hostFlag = flagBuf.hostData()[0];

    if(hostFlag != 0)
    {
        HIPDNN_SDK_LOG_INFO("GPU integer validation detected failure, falling back to CPU "
                            "for detailed diagnostics");
        return cpuFallback(reference, implementation);
    }

    return true;
}

template <class T>
bool GpuIntReferenceValidation<T>::cpuFallback(
    hipdnn_data_sdk::utilities::ITensor& reference,
    hipdnn_data_sdk::utilities::ITensor& implementation) const
{
    const hipdnn_test_sdk::utilities::CpuIntReferenceValidation<T> cpuValidator;
    return cpuValidator.allClose(reference, implementation);
}

// --- Non-template factory ---

std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createGpuAllCloseValidator(hipdnn_data_sdk::data_objects::DataType dataType,
                               float absoluteTolerance,
                               float relativeTolerance)
{
    switch(dataType)
    {
    case hipdnn_data_sdk::data_objects::DataType::FLOAT:
        return std::make_unique<GpuFpReferenceValidation<float>>(absoluteTolerance,
                                                                 relativeTolerance);
    case hipdnn_data_sdk::data_objects::DataType::HALF:
        return std::make_unique<GpuFpReferenceValidation<hipdnn_data_sdk::types::half>>(
            absoluteTolerance, relativeTolerance);
    case hipdnn_data_sdk::data_objects::DataType::BFLOAT16:
        return std::make_unique<GpuFpReferenceValidation<hipdnn_data_sdk::types::bfloat16>>(
            absoluteTolerance, relativeTolerance);
    case hipdnn_data_sdk::data_objects::DataType::DOUBLE:
        return std::make_unique<GpuFpReferenceValidation<double>>(absoluteTolerance,
                                                                  relativeTolerance);
    case hipdnn_data_sdk::data_objects::DataType::INT8:
        return std::make_unique<GpuIntReferenceValidation<int8_t>>();
    case hipdnn_data_sdk::data_objects::DataType::UINT8:
        return std::make_unique<GpuIntReferenceValidation<uint8_t>>();
    case hipdnn_data_sdk::data_objects::DataType::INT32:
        return std::make_unique<GpuIntReferenceValidation<int32_t>>();
    default:
        throw std::runtime_error("Unsupported data type for GPU allClose validator");
    }
}

// --- Explicit template instantiations ---

template class GpuFpReferenceValidation<float>;
template class GpuFpReferenceValidation<hipdnn_data_sdk::types::half>;
template class GpuFpReferenceValidation<hipdnn_data_sdk::types::bfloat16>;
template class GpuFpReferenceValidation<double>;

template class GpuIntReferenceValidation<int8_t>;
template class GpuIntReferenceValidation<uint8_t>;
template class GpuIntReferenceValidation<int32_t>;

} // namespace hipdnn_gpu_ref
