// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hipdnn-gpu-ref/GpuFpReferenceMatmul.hpp>

#include "hipdnn-gpu-ref/detail/GpuRefHelpers.hpp"
#include "hipdnn-gpu-ref/detail/GpuRefHipError.hpp"
#include "hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp"

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefMatmulArgs.h> // NOLINT(misc-include-cleaner)

void launchKernel(hipFunction_t function,
                  int64_t mTiles,
                  int64_t nTiles,
                  int64_t tileSize,
                  void* argsPtr,
                  size_t argsSize)
{
    const int64_t xlocalsize = tileSize;
    const int64_t xgridsize = mTiles;
    const int64_t ylocalsize = tileSize;
    const int64_t ygridsize = nTiles;
    const int64_t zlocalsize = 1;
    const int64_t zgridsize = 1;

    // Check the device limits for grid size
    detail::assertValidGridSize(xgridsize, ygridsize, zgridsize);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    detail::throwOnHipError(hipModuleLaunchKernel(function,
                                                  static_cast<unsigned int>(xgridsize),
                                                  static_cast<unsigned int>(ygridsize),
                                                  static_cast<unsigned int>(zgridsize),
                                                  static_cast<unsigned int>(xlocalsize),
                                                  static_cast<unsigned int>(ylocalsize),
                                                  static_cast<unsigned int>(zlocalsize),
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  config),
                            "hipModuleLaunchKernel failed");

    detail::throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
}

} // namespace

void GpuFpReferenceMatmul::launchMatmul(const void* aPtr,
                                        const std::vector<int64_t>& aDims,
                                        const std::vector<int64_t>& aStrides,
                                        const void* bPtr,
                                        const std::vector<int64_t>& bDims,
                                        const std::vector<int64_t>& bStrides,
                                        void* cPtr,
                                        const std::vector<int64_t>& cDims,
                                        const std::vector<int64_t>& cStrides,
                                        const int64_t tileSize,
                                        const std::vector<std::string>& defines)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefMatmul.cpp", defines, "MatmulRef");

    MatmulArgs args{};
    args.a = aPtr;
    args.b = bPtr;
    args.c = cPtr;
    std::memcpy(args.aDims, aDims.data(), aDims.size() * sizeof(int64_t));
    std::memcpy(args.aStrides, aStrides.data(), aStrides.size() * sizeof(int64_t));
    std::memcpy(args.bDims, bDims.data(), bDims.size() * sizeof(int64_t));
    std::memcpy(args.bStrides, bStrides.data(), bStrides.size() * sizeof(int64_t));
    std::memcpy(args.cDims, cDims.data(), cDims.size() * sizeof(int64_t));
    std::memcpy(args.cStrides, cStrides.data(), cStrides.size() * sizeof(int64_t));

    auto mTiles = (aDims[aDims.size() - 2] + tileSize - 1) / tileSize;
    auto nTiles = (bDims[bDims.size() - 1] + tileSize - 1) / tileSize;

    launchKernel(kernel.function(), mTiles, nTiles, tileSize, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
