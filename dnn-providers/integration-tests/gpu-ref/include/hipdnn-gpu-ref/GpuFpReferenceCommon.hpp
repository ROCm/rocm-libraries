// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#if defined(USE_ROCRAND)
#include <hip/hip_fp16.h>
#include <rocrand/rocrand.h>
#endif

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace common
{

// Shared argument structs — single definition used by both host and device (HipRTC).
#include <GpuRefCommonArgs.h> // NOLINT(misc-include-cleaner)

using namespace hipdnn_gpu_ref::detail;

namespace detail
{

#if defined(USE_ROCRAND)

inline void throwOnRocRandError(rocrand_status status, const char* what)
{
    if(status != ROCRAND_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(what) + " failed with rocRAND status "
                                 + std::to_string(static_cast<int>(status)));
    }
}

// RAII wrapper for rocrand_generator
struct RocRandGenerator
{
    explicit RocRandGenerator(rocrand_rng_type type)
    {
        throwOnRocRandError(rocrand_create_generator(&generator, type), "rocrand_create_generator");
    }

    ~RocRandGenerator()
    {
        if(generator != nullptr)
        {
            (void)rocrand_destroy_generator(generator);
            generator = nullptr;
        }
    }

    RocRandGenerator(const RocRandGenerator&) = delete;
    RocRandGenerator& operator=(const RocRandGenerator&) = delete;

    RocRandGenerator(RocRandGenerator&&) = delete;
    RocRandGenerator& operator=(RocRandGenerator&&) = delete;

    rocrand_generator generator{};
};

// RAII wrapper for hipMalloc and hipFree
template <class T>
struct HipDeviceBuffer
{
    explicit HipDeviceBuffer(size_t count)
    {
        throwOnHipError(hipMalloc(&data, count * sizeof(T)), "hipMalloc");
    }

    ~HipDeviceBuffer()
    {
        if(data != nullptr)
        {
            (void)hipFree(data);
            data = nullptr;
        }
    }

    HipDeviceBuffer(const HipDeviceBuffer&) = delete;
    HipDeviceBuffer& operator=(const HipDeviceBuffer&) = delete;

    HipDeviceBuffer(HipDeviceBuffer&&) = delete;
    HipDeviceBuffer& operator=(HipDeviceBuffer&&) = delete;

    T* data = nullptr;
};

#endif // defined(USE_ROCRAND)

} // namespace detail

namespace gpu_fp_reference_tensor
{
static constexpr unsigned int BLOCK_SIZE = 256;

#if defined(USE_ROCRAND)
template <class T>
static void
    launchScaleUniform(const void* srcPtr, void* dstPtr, size_t count, T minValue, T maxValue)
{
    if(count == 0)
    {
        return;
    }

    // For bfloat16, we use float as the source type as we generate random floats
    // and then convert them to bfloat16
    using SrcType
        = std::conditional_t<std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>, float, T>;

    const std::vector<std::string> defines{
        std::string("-DTARGET_TYPE=") + HipRtcTypeName<T>::VALUE,
        std::string("-DSOURCE_TYPE=") + HipRtcTypeName<SrcType>::VALUE,
        std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<double>::VALUE};

    auto& compiler = GpuRefKernelCompiler::instance();
    const auto& kernel = compiler.getOrCompile("GpuRefScaleUniform.cpp", defines, "ScaleUniform");

    ScaleUniformArgs args{srcPtr,
                          dstPtr,
                          static_cast<long long>(count),
                          static_cast<double>(minValue),
                          static_cast<double>(maxValue)};
    size_t argsSize = sizeof(args);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    // Check the device limits for grid size
    int deviceId;
    throwOnHipError(hipGetDevice(&deviceId), "hipGetDevice failed");

    hipDeviceProp_t deviceProps;
    throwOnHipError(hipGetDeviceProperties(&deviceProps, deviceId),
                    "hipGetDeviceProperties failed");

    const size_t gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if(gridSize > static_cast<size_t>(deviceProps.maxGridSize[0]))
    {
        throw std::runtime_error("Grid size exceeds device limit: " + std::to_string(gridSize)
                                 + " > " + std::to_string(deviceProps.maxGridSize[0]));
    }

    throwOnHipError(hipModuleLaunchKernel(kernel.function(),
                                          static_cast<unsigned int>(gridSize),
                                          1,
                                          1,
                                          BLOCK_SIZE,
                                          1,
                                          1,
                                          0,
                                          nullptr,
                                          nullptr,
                                          config),
                    "hipModuleLaunchKernel failed");
}

template <class T>
static void gpuFillWithRandomValues(hipdnn_data_sdk::utilities::TensorBase<T>& tensor,
                                    T minValue,
                                    T maxValue,
                                    unsigned int seed)
{
    tensor.memory().markDeviceModified();

    // Using elementSpace() to fill the entire allocated buffer
    // which could be larger than the number of elements in the
    // tensor's elementCount(), for example, in the case of
    // strided tensors.
    const auto count = tensor.elementSpace();
    auto* dstPtr = tensor.memory().deviceData();

    const detail::RocRandGenerator gen(ROCRAND_RNG_PSEUDO_DEFAULT);

    detail::throwOnRocRandError(rocrand_set_seed(gen.generator, seed), "rocrand_set_seed");

    // Launch the appropriate rocrand_generate_uniform function based on the data type
    if constexpr(std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>)
    {
        const detail::HipDeviceBuffer<float> scratch(count);

        detail::throwOnRocRandError(rocrand_generate_uniform(gen.generator, scratch.data, count),
                                    "rocrand_generate_uniform");

        launchScaleUniform<T>(scratch.data, dstPtr, count, minValue, maxValue);

        throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        detail::throwOnRocRandError(
            rocrand_generate_uniform_double(gen.generator, static_cast<double*>(dstPtr), count),
            "rocrand_generate_uniform_double");

        if(minValue != 0.0 || maxValue != 1.0)
        {
            launchScaleUniform<T>(dstPtr, dstPtr, count, minValue, maxValue);
        }

        throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
    }
    else if constexpr(std::is_same_v<T, hipdnn_data_sdk::types::half>)
    {
        detail::throwOnRocRandError(
            rocrand_generate_uniform_half(gen.generator, static_cast<__half*>(dstPtr), count),
            "rocrand_generate_uniform_half");

        if(minValue != static_cast<hipdnn_data_sdk::types::half>(0.0f)
           || maxValue != static_cast<hipdnn_data_sdk::types::half>(1.0f))
        {
            launchScaleUniform<T>(dstPtr, dstPtr, count, minValue, maxValue);
        }

        throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
    }
    else // float or other unsupported types
    {
        static_assert(std::is_same_v<T, float>, "Unsupported type for gpuFillWithRandomValues");

        detail::throwOnRocRandError(
            rocrand_generate_uniform(gen.generator, static_cast<float*>(dstPtr), count),
            "rocrand_generate_uniform");

        if(minValue != 0.0f || maxValue != 1.0f)
        {
            launchScaleUniform<T>(dstPtr, dstPtr, count, minValue, maxValue);
        }

        throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
    }
}
#endif // USE_ROCRAND

template <class T>
static void fillWithRandomValues(hipdnn_data_sdk::utilities::TensorBase<T>& tensor,
                                 T minValue,
                                 T maxValue,
                                 unsigned int seed = std::random_device{}())
{
#if defined(USE_ROCRAND)
    gpuFillWithRandomValues(tensor, minValue, maxValue, seed);
#else
    tensor.fillWithRandomValues(minValue, maxValue, seed);
#endif
}

}; // namespace gpu_fp_reference_tensor

} // namespace common

} // namespace hipdnn_gpu_ref
