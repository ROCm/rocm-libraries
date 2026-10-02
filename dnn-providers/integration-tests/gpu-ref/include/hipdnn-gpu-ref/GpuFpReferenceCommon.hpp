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

    // The grid is one thread per element, so the element count is bounded by the
    // device's grid limit. hipDeviceGetAttribute rather than hipGetDeviceProperties:
    // the latter fills in the whole property block, a real cost for a call made once
    // per input tensor of every test case.
    int deviceId = 0;
    throwOnHipError(hipGetDevice(&deviceId), "hipGetDevice failed");

    int maxGridSizeX = 0;
    throwOnHipError(hipDeviceGetAttribute(&maxGridSizeX, hipDeviceAttributeMaxGridDimX, deviceId),
                    "hipDeviceGetAttribute failed");

    const size_t gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if(gridSize > static_cast<size_t>(maxGridSizeX))
    {
        throw std::runtime_error("Grid size exceeds device limit: " + std::to_string(gridSize)
                                 + " > " + std::to_string(maxGridSizeX));
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

// One rocRAND generator per thread, created on first use. Creating a generator
// allocates its state on the device, and a fill runs once per input tensor of every
// test case, so building a fresh one each time costs more than generating the values
// for all but the largest tensors. The seed is set again on every fill, which restarts
// the sequence, so reuse does not change what a given seed produces.
//
// Destroyed at thread exit, which for the main thread is before the HIP and rocRAND
// libraries unload.
inline const detail::RocRandGenerator& threadRocRandGenerator()
{
    thread_local const detail::RocRandGenerator s_generator(ROCRAND_RNG_PSEUDO_DEFAULT);
    return s_generator;
}

// `synchronize` false leaves the fill in flight on the device, for a caller that fills
// several tensors and waits once afterwards. Nothing may read the tensor on another
// stream, or on the host, until a hipDeviceSynchronize() has returned.
template <class T>
static void gpuFillWithRandomValues(hipdnn_data_sdk::utilities::TensorBase<T>& tensor,
                                    T minValue,
                                    T maxValue,
                                    unsigned int seed,
                                    bool synchronize)
{
    tensor.memory().markDeviceModified();

    // Using elementSpace() to fill the entire allocated buffer
    // which could be larger than the number of elements in the
    // tensor's elementCount(), for example, in the case of
    // strided tensors.
    const auto count = tensor.elementSpace();
    auto* dstPtr = tensor.memory().deviceData();

    const auto& gen = threadRocRandGenerator();

    detail::throwOnRocRandError(rocrand_set_seed(gen.generator, seed), "rocrand_set_seed");

    // Launch the appropriate rocrand_generate_uniform function based on the data type
    if constexpr(std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>)
    {
        const detail::HipDeviceBuffer<float> scratch(count);

        detail::throwOnRocRandError(rocrand_generate_uniform(gen.generator, scratch.data, count),
                                    "rocrand_generate_uniform");

        launchScaleUniform<T>(scratch.data, dstPtr, count, minValue, maxValue);

        // Not deferred: the scaling kernel is still reading `scratch`, which is freed
        // when this block ends.
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
    }

    if(synchronize)
    {
        throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
    }
}
#endif // USE_ROCRAND

// Fills `tensor` with uniform random values in [minValue, maxValue]: on the device
// with rocRAND when it is available, on the host otherwise. With rocRAND the data
// lives on the device afterwards and is migrated to the host by the first non-const
// host access; a const access cannot migrate and throws.
//
// `synchronize` is only honoured on the rocRAND path; see gpuFillWithRandomValues().
template <class T>
static void fillWithRandomValues(hipdnn_data_sdk::utilities::TensorBase<T>& tensor,
                                 T minValue,
                                 T maxValue,
                                 unsigned int seed = std::random_device{}(),
                                 [[maybe_unused]] bool synchronize = true)
{
#if defined(USE_ROCRAND)
    gpuFillWithRandomValues(tensor, minValue, maxValue, seed, synchronize);
#else
    tensor.fillWithRandomValues(minValue, maxValue, seed);
#endif
}

}; // namespace gpu_fp_reference_tensor

} // namespace common

} // namespace hipdnn_gpu_ref
