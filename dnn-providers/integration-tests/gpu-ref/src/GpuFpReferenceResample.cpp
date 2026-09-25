// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn-gpu-ref/GpuFpReferenceResample.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHelpers.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefResampleArgs.h> // NOLINT(misc-include-cleaner)

void launchKernel(hipFunction_t function, int64_t gridSize, void* argsPtr, size_t argsSize)
{
    // Check the device limits for grid size
    detail::assertValidGridSize(gridSize, 1, 1);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    detail::throwOnHipError(hipModuleLaunchKernel(function,
                                                  static_cast<unsigned int>(gridSize),
                                                  1,
                                                  1,
                                                  GpuFpReferenceResample::BLOCK_SIZE,
                                                  1,
                                                  1,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  config),
                            "hipModuleLaunchKernel failed");

    detail::throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
}

} // namespace

// --- Kernel launchers ---

void GpuFpReferenceResample::launchForward(const void* xPtr,
                                           const std::vector<int64_t>& xSpatialDims,
                                           const std::vector<int64_t>& xStrides,
                                           void* yPtr,
                                           const std::vector<int64_t>& ySpatialDims,
                                           const std::vector<int64_t>& yStrides,
                                           const std::array<int64_t, 2>& batchChannelDims,
                                           const ResampleSpatialParams& spatialParams,
                                           const ResampleModeParams& modeParams,
                                           std::vector<std::string>& defines,
                                           void* indexPtr)
{
    ResampleMode resampleMode;
    switch(modeParams.resampleMode)
    {
    case hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL:
        resampleMode = ResampleMode::MAXPOOL;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING:
        resampleMode = ResampleMode::AVGPOOL_EXCLUDE_PADDING;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING:
        resampleMode = ResampleMode::AVGPOOL_INCLUDE_PADDING;
        break;
    default:
        throw std::invalid_argument("Unsupported resample mode: "
                                    + std::to_string(static_cast<int>(modeParams.resampleMode)));
    }
    defines.emplace_back(std::string("-DRESAMPLE_MODE=")
                         + std::to_string(static_cast<int>(resampleMode)));

    PaddingMode paddingMode;
    switch(modeParams.paddingMode)
    {
    case hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD:
        paddingMode = PaddingMode::ZERO_PAD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PaddingMode::NEG_INF_PAD:
        paddingMode = PaddingMode::NEG_INF_PAD;
        break;
    default:
        throw std::invalid_argument("Unsupported padding mode: "
                                    + std::to_string(static_cast<int>(modeParams.paddingMode)));
    }
    defines.emplace_back(std::string("-DPADDING_MODE=")
                         + std::to_string(static_cast<int>(paddingMode)));
    defines.emplace_back(std::string("-DSPATIAL_DIMS=") + std::to_string(xSpatialDims.size()));

    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel = compiler.getOrCompile("GpuRefResampleFwd.cpp", defines, "ResampleFwdRef");

    ResampleFwdArgs args{};
    args.x = xPtr;
    args.y = yPtr;
    args.index = indexPtr;
    std::memcpy(args.xSpatialDims, xSpatialDims.data(), xSpatialDims.size() * sizeof(int64_t));
    std::memcpy(args.xStrides, xStrides.data(), xStrides.size() * sizeof(int64_t));
    std::memcpy(args.ySpatialDims, ySpatialDims.data(), ySpatialDims.size() * sizeof(int64_t));
    std::memcpy(args.yStrides, yStrides.data(), yStrides.size() * sizeof(int64_t));
    std::memcpy(args.prePadding,
                spatialParams.prePadding.data(),
                spatialParams.prePadding.size() * sizeof(int64_t));
    std::memcpy(
        args.stride, spatialParams.stride.data(), spatialParams.stride.size() * sizeof(int64_t));
    std::memcpy(
        args.window, spatialParams.window.data(), spatialParams.window.size() * sizeof(int64_t));
    const auto outputElementCount
        = batchChannelDims[0] * batchChannelDims[1]
          * std::accumulate(
              ySpatialDims.begin(), ySpatialDims.end(), int64_t{1}, std::multiplies<>());
    args.n = static_cast<long long>(batchChannelDims[0]);
    args.c = static_cast<long long>(batchChannelDims[1]);

    launchKernel(
        kernel.function(), (outputElementCount + BLOCK_SIZE - 1) / BLOCK_SIZE, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
