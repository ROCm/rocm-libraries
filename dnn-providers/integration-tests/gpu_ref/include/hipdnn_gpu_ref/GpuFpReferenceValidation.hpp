// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_gpu_ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

// RAII wrapper for HIP device memory
class GpuValidatorBuffer
{
public:
    explicit GpuValidatorBuffer(size_t bytes)
        : _bytes(bytes)
    {
        if(bytes > 0)
        {
            throwOnHipError(hipMalloc(&_ptr, bytes), "GpuValidatorBuffer: hipMalloc failed");
        }
    }

    ~GpuValidatorBuffer()
    {
        if(_ptr != nullptr)
        {
            // Best-effort cleanup; ignore errors during destruction
            static_cast<void>(hipFree(_ptr));
        }
    }

    GpuValidatorBuffer(const GpuValidatorBuffer&) = delete;
    GpuValidatorBuffer& operator=(const GpuValidatorBuffer&) = delete;
    GpuValidatorBuffer(GpuValidatorBuffer&&) = delete;
    GpuValidatorBuffer& operator=(GpuValidatorBuffer&&) = delete;

    void* get() const
    {
        return _ptr;
    }

    size_t bytes() const
    {
        return _bytes;
    }

private:
    void* _ptr = nullptr;
    size_t _bytes;
};

#include <GpuRefValidatorArgs.h> // NOLINT(misc-include-cleaner)

inline std::vector<std::string> buildValidatorDefines(const char* dataType, const char* computeType)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DDATA_TYPE=") + dataType);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + computeType);
    // GpuRefTypes.h toAccum/fromAccum overloads require X_TYPE defined
    defines.emplace_back(std::string("-DX_TYPE=") + dataType);
    defines.emplace_back(std::string("-DW_TYPE=") + dataType);
    defines.emplace_back(std::string("-DY_TYPE=") + dataType);
    return defines;
}

inline void
    launchValidatorKernel(hipFunction_t function, int64_t totalElements, ValidatorArgs& args)
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

// GPU-based floating-point tensor validator implementing IReferenceValidation.
// Launches a HipRTC kernel to perform element-wise tolerance comparison on the GPU
// using a single atomic failure flag. On failure, falls back to CpuFpReferenceValidation
// for detailed per-element diagnostics. Also falls back on GPU errors.
template <class T>
class GpuFpReferenceValidation : public hipdnn_test_sdk::utilities::IReferenceValidation
{
public:
    // NOLINTNEXTLINE(readability-redundant-casting) - cast needed for non-float T types
    GpuFpReferenceValidation(float absoluteTolerance = float(std::numeric_limits<T>::epsilon()),
                             // NOLINTNEXTLINE(readability-redundant-casting)
                             float relativeTolerance = float(std::numeric_limits<T>::epsilon()))
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

    ~GpuFpReferenceValidation() override = default;

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
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
            HIPDNN_SDK_LOG_WARN(
                "GPU validation failed, falling back to CPU validator: " << e.what());
            return cpuFallback(reference, implementation);
        }
    }

private:
    bool gpuAllClose(hipdnn_data_sdk::utilities::ITensor& reference,
                     hipdnn_data_sdk::utilities::ITensor& implementation) const
    {
        auto totalElements = static_cast<int64_t>(reference.elementCount());

        // Allocate single failure flag on device
        const detail::GpuValidatorBuffer flagBuf(sizeof(int));
        detail::throwOnHipError(hipMemset(flagBuf.get(), 0, sizeof(int)),
                                "validateAllClose: hipMemset failureFlag failed");

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
        args.failureFlag = static_cast<int*>(flagBuf.get());
        args.totalElements = totalElements;
        args.absoluteTolerance = static_cast<double>(_absoluteTolerance);
        args.relativeTolerance = static_cast<double>(_relativeTolerance);

        detail::launchValidatorKernel(kernel.function(), totalElements, args);

        // Read back single failure flag
        int hostFlag = 0;
        detail::throwOnHipError(
            hipMemcpy(&hostFlag, flagBuf.get(), sizeof(int), hipMemcpyDeviceToHost),
            "validateAllClose: hipMemcpy failureFlag failed");

        if(hostFlag != 0)
        {
            HIPDNN_SDK_LOG_INFO(
                "GPU validation detected failure, falling back to CPU for detailed diagnostics");
            return cpuFallback(reference, implementation);
        }

        return true;
    }

    bool cpuFallback(hipdnn_data_sdk::utilities::ITensor& reference,
                     hipdnn_data_sdk::utilities::ITensor& implementation) const
    {
        const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<T> cpuValidator(
            _absoluteTolerance, _relativeTolerance);
        return cpuValidator.allClose(reference, implementation);
    }

    float _absoluteTolerance;
    float _relativeTolerance;
};

// GPU-based integer tensor validator implementing IReferenceValidation.
// Requires exact equality between reference and implementation tensors.
// Falls back to CpuIntReferenceValidation on GPU errors.
template <class T>
class GpuIntReferenceValidation : public hipdnn_test_sdk::utilities::IReferenceValidation
{
public:
    GpuIntReferenceValidation() = default;
    ~GpuIntReferenceValidation() override = default;

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
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
            HIPDNN_SDK_LOG_WARN(
                "GPU validation failed, falling back to CPU validator: " << e.what());
            return cpuFallback(reference, implementation);
        }
    }

private:
    bool gpuExact(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const
    {
        auto totalElements = static_cast<int64_t>(reference.elementCount());

        // Allocate single failure flag on device
        const detail::GpuValidatorBuffer flagBuf(sizeof(int));
        detail::throwOnHipError(hipMemset(flagBuf.get(), 0, sizeof(int)),
                                "validateExact: hipMemset failureFlag failed");

        auto* refPtr = reference.rawDeviceData();
        auto* implPtr = implementation.rawDeviceData();

        auto defines = detail::buildValidatorDefines(detail::HipRtcTypeName<T>::VALUE, "double");

        auto& compiler = detail::GpuRefKernelCompiler::instance();
        auto& kernel = compiler.getOrCompile("GpuRefValidator.cpp", defines, "validateExact");

        detail::ValidatorArgs args{};
        args.reference = refPtr;
        args.implementation = implPtr;
        args.failureFlag = static_cast<int*>(flagBuf.get());
        args.totalElements = totalElements;
        args.absoluteTolerance = 0.0;
        args.relativeTolerance = 0.0;

        detail::launchValidatorKernel(kernel.function(), totalElements, args);

        // Read back single failure flag
        int hostFlag = 0;
        detail::throwOnHipError(
            hipMemcpy(&hostFlag, flagBuf.get(), sizeof(int), hipMemcpyDeviceToHost),
            "validateExact: hipMemcpy failureFlag failed");

        if(hostFlag != 0)
        {
            HIPDNN_SDK_LOG_INFO("GPU integer validation detected failure, falling back to CPU "
                                "for detailed diagnostics");
            return cpuFallback(reference, implementation);
        }

        return true;
    }

    bool cpuFallback(hipdnn_data_sdk::utilities::ITensor& reference,
                     hipdnn_data_sdk::utilities::ITensor& implementation) const
    {
        const hipdnn_test_sdk::utilities::CpuIntReferenceValidation<T> cpuValidator;
        return cpuValidator.allClose(reference, implementation);
    }
};

// Factory function to create a GPU allClose validator for the given data type.
// Mirrors the createAllCloseValidator() API from CpuFpReferenceValidation.hpp.
inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createGpuAllCloseValidator(hipdnn_data_sdk::data_objects::DataType dataType,
                               float absoluteTolerance = std::numeric_limits<float>::epsilon(),
                               float relativeTolerance = std::numeric_limits<float>::epsilon())
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

// Templated factory function to create a GPU allClose validator.
template <typename T>
inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    // NOLINTNEXTLINE(readability-redundant-casting) - cast needed for non-float T types
    createGpuAllCloseValidator(float absoluteTolerance = float(std::numeric_limits<T>::epsilon()),
                               // NOLINTNEXTLINE(readability-redundant-casting)
                               float relativeTolerance = float(std::numeric_limits<T>::epsilon()))
{
    if constexpr(std::is_integral_v<T>)
    {
        return std::make_unique<GpuIntReferenceValidation<T>>();
    }
    else
    {
        return std::make_unique<GpuFpReferenceValidation<T>>(absoluteTolerance, relativeTolerance);
    }
}

} // namespace hipdnn_gpu_ref
