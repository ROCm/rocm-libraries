// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hipdnn-gpu-ref/GpuFpReferenceMatmul.hpp>

#include "hipdnn-gpu-ref/detail/GpuRefHelpers.hpp"
#include "hipdnn-gpu-ref/detail/GpuRefHipError.hpp"
#include "hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp"
#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefMatmulArgs.h> // NOLINT(misc-include-cleaner)

} // namespace

namespace hipdnn_gpu_ref
{

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

    if(aDims.size() > 5 || aStrides.size() > 5 || bDims.size() > 5 || bStrides.size() > 5
       || cDims.size() > 5 || cStrides.size() > 5)
    {
        throw std::runtime_error(
            "Rank of dimensions and/or strides for A, B and/or C is too large");
    }

    MatmulArgs args{};
    static_assert(std::size(args.aDims) == 5);
    static_assert(std::size(args.aStrides) == 5);
    static_assert(std::size(args.bDims) == 5);
    static_assert(std::size(args.bStrides) == 5);
    static_assert(std::size(args.cDims) == 5);
    static_assert(std::size(args.cStrides) == 5);
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

    int64_t batches = 1;
    for(size_t i = 0; i < cDims.size() - 2; ++i)
    {
        batches *= cDims[i];
    }

    detail::launchKernel(
        kernel.function(), {mTiles, nTiles, batches}, {tileSize, tileSize, 1}, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
