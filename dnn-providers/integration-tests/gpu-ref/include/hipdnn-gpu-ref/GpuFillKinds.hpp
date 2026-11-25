// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3Fnuz.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2Fnuz.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(USE_ROCRAND)

namespace hipdnn_gpu_ref
{

namespace common
{

namespace gpu_fp_reference_tensor
{

/// The element types gpuFillKind() writes: the ones whose values the kernel can build.
template <class T>
inline constexpr bool IS_FILL_KIND_TYPE_V
    = std::is_same_v<T, float> || std::is_same_v<T, hipdnn_data_sdk::types::half>
      || std::is_same_v<T, hipdnn_data_sdk::types::bfloat16>
      || std::is_same_v<T, hipdnn_data_sdk::types::fp8_e4m3>
      || std::is_same_v<T, hipdnn_data_sdk::types::fp8_e5m2>
      || std::is_same_v<T, hipdnn_data_sdk::types::fp8_e4m3_fnuz>
      || std::is_same_v<T, hipdnn_data_sdk::types::fp8_e5m2_fnuz>;

/// Most dimensions a tensor given to gpuFillKind() can have.
inline constexpr std::size_t FILL_KIND_MAX_RANK = GPU_REF_FILL_MAX_RANK;

/// Fills every element of `tensor` on the device by the kind in `args` (GpuRefFillKinds.cpp;
/// FillKindArgs says which fields each kind reads). The caller sets `kind`, `seed`, `lo`, `hi`,
/// `fallback`, `maxSubnormalCode`, `quota` and `plantedBits`; the rest is filled in here.
///
/// The draws come from `generator`, which the caller owns, seeded with `args.seed`: two uniform
/// floats per element, in one rocrand_generate_uniform call. A tensor with padding between its
/// elements keeps the bytes of the padding: its host copy is uploaded first when it is the
/// current one, and the kernel writes only the elements.
///
/// Blocks until the fill has finished when it drew into a scratch buffer, which is freed on the
/// way out; a ZEROS fill is left in flight, so the caller waits before reading the tensor.
template <class T>
static void gpuFillKind(hipdnn_data_sdk::utilities::TensorBase<T>& tensor,
                        FillKindArgs args,
                        const RocRandGenerator& generator)
{
    static_assert(IS_FILL_KIND_TYPE_V<T>, "Unsupported type for gpuFillKind");

    const auto count = tensor.elementCount();
    const auto& dims = tensor.dims();
    const auto& strides = tensor.strides();
    if(count == 0)
    {
        return;
    }
    if(dims.size() > FILL_KIND_MAX_RANK || dims.size() != strides.size())
    {
        throw std::runtime_error("gpuFillKind: a tensor of rank " + std::to_string(dims.size())
                                 + " is not supported");
    }

    // Uploading first keeps the padding's bytes. A tensor without padding has none to keep,
    // and its host copy is about to be wholly overwritten, so it is not moved.
    void* dstPtr = nullptr;
    if(tensor.elementSpace() != count)
    {
        dstPtr = tensor.memory().deviceData();
        tensor.memory().markDeviceModified();
    }
    else
    {
        tensor.memory().markDeviceModified();
        dstPtr = tensor.memory().deviceData();
    }

    const bool drawsRandom = static_cast<FillKind>(args.kind) != FillKind::ZEROS;
    std::optional<detail::HipDeviceBuffer<float>> scratch;
    if(drawsRandom)
    {
        const size_t draws = 2 * count;
        scratch.emplace(draws);
        detail::throwOnRocRandError(rocrand_set_seed(generator.generator, args.seed),
                                    "rocrand_set_seed");
        detail::throwOnRocRandError(
            rocrand_generate_uniform(generator.generator, scratch->data, draws),
            "rocrand_generate_uniform");
    }

    args.uniform = drawsRandom ? scratch->data : nullptr;
    args.dst = dstPtr;
    args.count = static_cast<long long>(count);
    args.rank = static_cast<long long>(dims.size());
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        args.dims[axis] = dims[axis];
        args.strides[axis] = strides[axis];
    }

    const std::vector<std::string> defines{std::string("-DTARGET_TYPE=") + HipRtcTypeName<T>::VALUE,
                                           std::string("-DCOMPUTE_TYPE=")
                                               + HipRtcTypeName<double>::VALUE};
    auto& compiler = GpuRefKernelCompiler::instance();
    const auto& kernel = compiler.getOrCompile("GpuRefFillKinds.cpp", defines, "FillKinds");

    size_t argsSize = sizeof(args);
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    // One thread per element, bounded by the device's grid limit, as in launchScaleUniform.
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

    // Not deferred when `scratch` holds the draws: the kernel is still reading it, and it is
    // freed when this function returns.
    if(drawsRandom)
    {
        throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
    }
}

} // namespace gpu_fp_reference_tensor

} // namespace common

} // namespace hipdnn_gpu_ref

#endif // defined(USE_ROCRAND)
