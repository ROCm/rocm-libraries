// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn-gpu-ref/GpuFpReferenceResample.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefResampleArgs.h> // NOLINT(misc-include-cleaner)

void addModesToCompileDefines(std::vector<std::string>& defines,
                              hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode,
                              hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode)
{
    ResampleMode resampleModeEnum;
    switch(resampleMode)
    {
    case hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL:
        resampleModeEnum = ResampleMode::MAXPOOL;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING:
        resampleModeEnum = ResampleMode::AVGPOOL_EXCLUDE_PADDING;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING:
        resampleModeEnum = ResampleMode::AVGPOOL_INCLUDE_PADDING;
        break;
    default:
        throw std::invalid_argument("Unsupported resample mode: "
                                    + std::to_string(static_cast<int>(resampleMode)));
    }
    defines.emplace_back(std::string("-DRESAMPLE_MODE=")
                         + std::to_string(static_cast<int>(resampleModeEnum)));

    PaddingMode paddingModeEnum;
    switch(paddingMode)
    {
    case hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD:
        paddingModeEnum = PaddingMode::ZERO_PAD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PaddingMode::NEG_INF_PAD:
        paddingModeEnum = PaddingMode::NEG_INF_PAD;
        break;
    default:
        throw std::invalid_argument("Unsupported padding mode: "
                                    + std::to_string(static_cast<int>(paddingMode)));
    }
    defines.emplace_back(std::string("-DPADDING_MODE=")
                         + std::to_string(static_cast<int>(paddingModeEnum)));
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
    addModesToCompileDefines(defines, modeParams.resampleMode, modeParams.paddingMode);
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

    detail::launchKernel1d(kernel.function(),
                           (outputElementCount + BLOCK_SIZE - 1) / BLOCK_SIZE,
                           BLOCK_SIZE,
                           &args,
                           sizeof(args));
}

void GpuFpReferenceResample::launchBackward(const void* dyPtr,
                                            const std::vector<int64_t>& dySpatialDims,
                                            const std::vector<int64_t>& dyStrides,
                                            void* dxPtr,
                                            const std::vector<int64_t>& dxSpatialDims,
                                            const std::vector<int64_t>& dxStrides,
                                            const std::array<int64_t, 2>& batchChannelDims,
                                            const ResampleSpatialParams& spatialParams,
                                            const ResampleModeParams& modeParams,
                                            std::vector<std::string>& defines,
                                            const void* indexPtr)
{
    addModesToCompileDefines(defines, modeParams.resampleMode, modeParams.paddingMode);
    defines.emplace_back(std::string("-DSPATIAL_DIMS=") + std::to_string(dySpatialDims.size()));

    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel = compiler.getOrCompile("GpuRefResampleBwd.cpp", defines, "ResampleBwdRef");

    ResampleBwdArgs args{};
    args.dy = dyPtr;
    args.dx = dxPtr;
    args.index = indexPtr;
    std::memcpy(args.dySpatialDims, dySpatialDims.data(), dySpatialDims.size() * sizeof(int64_t));
    std::memcpy(args.dyStrides, dyStrides.data(), dyStrides.size() * sizeof(int64_t));
    std::memcpy(args.dxSpatialDims, dxSpatialDims.data(), dxSpatialDims.size() * sizeof(int64_t));
    std::memcpy(args.dxStrides, dxStrides.data(), dxStrides.size() * sizeof(int64_t));
    std::memcpy(args.prePadding,
                spatialParams.prePadding.data(),
                spatialParams.prePadding.size() * sizeof(int64_t));
    std::memcpy(
        args.stride, spatialParams.stride.data(), spatialParams.stride.size() * sizeof(int64_t));
    std::memcpy(
        args.window, spatialParams.window.data(), spatialParams.window.size() * sizeof(int64_t));
    const auto numDxElements
        = batchChannelDims[0] * batchChannelDims[1]
          * std::accumulate(
              dxSpatialDims.begin(), dxSpatialDims.end(), int64_t{1}, std::multiplies<>());
    args.n = static_cast<long long>(batchChannelDims[0]);
    args.c = static_cast<long long>(batchChannelDims[1]);

    detail::launchKernel1d(kernel.function(),
                           (numDxElements + BLOCK_SIZE - 1) / BLOCK_SIZE,
                           BLOCK_SIZE,
                           &args,
                           sizeof(args));
}

} // namespace hipdnn_gpu_ref
