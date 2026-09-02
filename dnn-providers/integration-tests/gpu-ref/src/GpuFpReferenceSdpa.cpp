// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefSdpaArgs.h> // NOLINT(misc-include-cleaner)

SdpaStrides toSdpaStrides(const std::vector<int64_t>& strides)
{
    SdpaStrides result{};
    for(size_t i = 0; i < 4 && i < strides.size(); ++i)
    {
        result.s[i] = static_cast<long long>(strides[i]);
    }
    return result;
}

void launchKernel(hipFunction_t function, int64_t totalElements, void* argsPtr, size_t argsSize)
{
    const int64_t blockSize = 256;
    auto gridSize = (totalElements + blockSize - 1) / blockSize;

    if(gridSize > static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error("Grid size exceeds hipModuleLaunchKernel limit");
    }

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
                                                  static_cast<unsigned int>(blockSize),
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

// --- SDPA forward kernel launcher ---

void GpuFpReferenceSdpa::launchSdpaFwd(const void* qPtr,
                                       const void* kPtr,
                                       const void* vPtr,
                                       const void* maskPtr,
                                       void* oPtr,
                                       void* lsePtr,
                                       const std::vector<int64_t>& qTensorStrides,
                                       const std::vector<int64_t>& kTensorStrides,
                                       const std::vector<int64_t>& vTensorStrides,
                                       const std::vector<int64_t>& oTensorStrides,
                                       const std::vector<int64_t>& maskTensorStrides,
                                       const std::vector<int64_t>& maskDims,
                                       const std::vector<int64_t>& lseTensorStrides,
                                       int64_t batch,
                                       int64_t numHeads,
                                       int64_t numHeadsK,
                                       int64_t numHeadsV,
                                       int64_t seqQ,
                                       int64_t seqKv,
                                       int64_t headDim,
                                       int64_t headDimV,
                                       float scale,
                                       int64_t leftBound,
                                       int64_t rightBound,
                                       bool topLeftAlignment,
                                       const std::vector<std::string>& defines)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefSdpaFwd.cpp", defines, "sdpaFwdRef");

    SdpaFwdArgs args{};
    args.q = qPtr;
    args.k = kPtr;
    args.v = vPtr;
    args.mask = maskPtr;
    args.o = oPtr;
    args.lse = lsePtr;
    args.qStr = toSdpaStrides(qTensorStrides);
    args.kStr = toSdpaStrides(kTensorStrides);
    args.vStr = toSdpaStrides(vTensorStrides);
    args.oStr = toSdpaStrides(oTensorStrides);
    args.maskStr = toSdpaStrides(maskTensorStrides);
    args.lseStr = toSdpaStrides(lseTensorStrides);
    args.batch = static_cast<long long>(batch);
    args.numHeads = static_cast<long long>(numHeads);
    args.numHeadsK = static_cast<long long>(numHeadsK);
    args.numHeadsV = static_cast<long long>(numHeadsV);
    args.seqQ = static_cast<long long>(seqQ);
    args.seqKv = static_cast<long long>(seqKv);
    args.headDim = static_cast<long long>(headDim);
    args.headDimV = static_cast<long long>(headDimV);
    args.maskRank = static_cast<int>(maskDims.size());
    for(size_t i = 0; i < 4 && i < maskDims.size(); ++i)
    {
        args.maskDims[i] = static_cast<long long>(maskDims[i]);
    }
    args.scale = scale;
    args.leftBound = static_cast<long long>(leftBound);
    args.rightBound = static_cast<long long>(rightBound);
    args.topLeftAlignment = topLeftAlignment ? 1 : 0;

    auto totalElements = batch * numHeads * seqQ * headDimV;
    launchKernel(kernel.function(), totalElements, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
