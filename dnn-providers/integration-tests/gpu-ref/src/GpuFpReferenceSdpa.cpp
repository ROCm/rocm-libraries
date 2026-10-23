// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
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

SdpaBwdArgs makeSdpaBwddQdKdVArgs(const void* qPtr,
                                  const void* kPtr,
                                  const void* vPtr,
                                  const void* dOPtr,
                                  const void* lsePtr,
                                  const void* maskPtr,
                                  void* dBuffPtr,
                                  const std::vector<int64_t>& qStrides,
                                  const std::vector<int64_t>& kStrides,
                                  const std::vector<int64_t>& vStrides,
                                  const std::vector<int64_t>& dOStrides,
                                  const std::vector<int64_t>& lseStrides,
                                  const std::vector<int64_t>& maskStrides,
                                  const std::vector<int64_t>& dBuffStrides,
                                  const std::vector<int64_t>& maskDims,
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
                                  bool topLeftAlignment)
{
    SdpaBwdArgs args{};
    args.q = qPtr;
    args.k = kPtr;
    args.v = vPtr;
    args.dO = dOPtr;
    args.lse = lsePtr;
    args.mask = maskPtr;
    args.dBuff = dBuffPtr;
    args.qStr = toSdpaStrides(qStrides);
    args.kStr = toSdpaStrides(kStrides);
    args.vStr = toSdpaStrides(vStrides);
    args.dOStr = toSdpaStrides(dOStrides);
    args.lseStr = toSdpaStrides(lseStrides);
    args.maskStr = toSdpaStrides(maskStrides);
    args.dBuffStr = toSdpaStrides(dBuffStrides);
    args.batch = batch;
    args.numHeads = numHeads;
    args.numHeadsK = numHeadsK;
    args.numHeadsV = numHeadsV;
    args.seqQ = seqQ;
    args.seqKv = seqKv;
    args.headDim = headDim;
    args.headDimV = headDimV;
    args.maskRank = static_cast<int>(maskDims.size());
    for(size_t i = 0; i < 4 && i < maskDims.size(); ++i)
    {
        args.maskDims[i] = maskDims[i];
    }
    args.scale = scale;
    args.leftBound = leftBound;
    args.rightBound = rightBound;
    args.topLeftAlignment = topLeftAlignment ? 1 : 0;
    return args;
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
    detail::launchKernelForElements(kernel.function(), totalElements, &args, sizeof(args));
}

// --- SDPA backward kernel launchers ---

void GpuFpReferenceSdpa::launchSdpaBwdD(const void* oPtr,
                                        const void* dOPtr,
                                        void* dBuffPtr,
                                        const std::vector<int64_t>& oStrides,
                                        const std::vector<int64_t>& dOStrides,
                                        const std::vector<int64_t>& dBuffStrides,
                                        int64_t batch,
                                        int64_t numHeads,
                                        int64_t seqQ,
                                        int64_t headDimV,
                                        const std::vector<std::string>& defines)
{
    auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefSdpaBwd.cpp", defines, "sdpaBwdRefD");

    SdpaBwdArgs args{};
    args.o = oPtr;
    args.dO = dOPtr;
    args.dBuff = dBuffPtr;
    args.oStr = toSdpaStrides(oStrides);
    args.dOStr = toSdpaStrides(dOStrides);
    args.dBuffStr = toSdpaStrides(dBuffStrides);
    args.batch = batch;
    args.numHeads = numHeads;
    args.seqQ = seqQ;
    args.headDimV = headDimV;

    detail::launchKernelForElements(
        kernel.function(), batch * numHeads * seqQ, &args, sizeof(args));
}

void GpuFpReferenceSdpa::launchSdpaBwdDQ(const void* qPtr,
                                         const void* kPtr,
                                         const void* vPtr,
                                         const void* dOPtr,
                                         void* dQPtr,
                                         const void* lsePtr,
                                         const void* maskPtr,
                                         void* dBuffPtr,
                                         const std::vector<int64_t>& qStrides,
                                         const std::vector<int64_t>& kStrides,
                                         const std::vector<int64_t>& vStrides,
                                         const std::vector<int64_t>& dOStrides,
                                         const std::vector<int64_t>& dQStrides,
                                         const std::vector<int64_t>& lseStrides,
                                         const std::vector<int64_t>& maskStrides,
                                         const std::vector<int64_t>& dBuffStrides,
                                         const std::vector<int64_t>& maskDims,
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
    auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefSdpaBwd.cpp", defines, "sdpaBwdRefdQ");

    auto args = makeSdpaBwddQdKdVArgs(qPtr,
                                      kPtr,
                                      vPtr,
                                      dOPtr,
                                      lsePtr,
                                      maskPtr,
                                      dBuffPtr,
                                      qStrides,
                                      kStrides,
                                      vStrides,
                                      dOStrides,
                                      lseStrides,
                                      maskStrides,
                                      dBuffStrides,
                                      maskDims,
                                      batch,
                                      numHeads,
                                      numHeadsK,
                                      numHeadsV,
                                      seqQ,
                                      seqKv,
                                      headDim,
                                      headDimV,
                                      scale,
                                      leftBound,
                                      rightBound,
                                      topLeftAlignment);
    args.dQ = dQPtr;
    args.dQStr = toSdpaStrides(dQStrides);

    detail::launchKernelForElements(
        kernel.function(), batch * numHeads * seqQ * headDim, &args, sizeof(args));
}

void GpuFpReferenceSdpa::launchSdpaBwdDK(const void* qPtr,
                                         const void* kPtr,
                                         const void* vPtr,
                                         const void* dOPtr,
                                         void* dKPtr,
                                         const void* lsePtr,
                                         const void* maskPtr,
                                         void* dBuffPtr,
                                         const std::vector<int64_t>& qStrides,
                                         const std::vector<int64_t>& kStrides,
                                         const std::vector<int64_t>& vStrides,
                                         const std::vector<int64_t>& dOStrides,
                                         const std::vector<int64_t>& dKStrides,
                                         const std::vector<int64_t>& lseStrides,
                                         const std::vector<int64_t>& maskStrides,
                                         const std::vector<int64_t>& dBuffStrides,
                                         const std::vector<int64_t>& maskDims,
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
    auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefSdpaBwd.cpp", defines, "sdpaBwdRefdK");

    auto args = makeSdpaBwddQdKdVArgs(qPtr,
                                      kPtr,
                                      vPtr,
                                      dOPtr,
                                      lsePtr,
                                      maskPtr,
                                      dBuffPtr,
                                      qStrides,
                                      kStrides,
                                      vStrides,
                                      dOStrides,
                                      lseStrides,
                                      maskStrides,
                                      dBuffStrides,
                                      maskDims,
                                      batch,
                                      numHeads,
                                      numHeadsK,
                                      numHeadsV,
                                      seqQ,
                                      seqKv,
                                      headDim,
                                      headDimV,
                                      scale,
                                      leftBound,
                                      rightBound,
                                      topLeftAlignment);
    args.dK = dKPtr;
    args.dKStr = toSdpaStrides(dKStrides);

    detail::launchKernelForElements(
        kernel.function(), batch * numHeadsK * seqKv * headDim, &args, sizeof(args));
}

void GpuFpReferenceSdpa::launchSdpaBwdDV(const void* qPtr,
                                         const void* kPtr,
                                         const void* vPtr,
                                         const void* dOPtr,
                                         void* dVPtr,
                                         const void* lsePtr,
                                         const void* maskPtr,
                                         void* dBuffPtr,
                                         const std::vector<int64_t>& qStrides,
                                         const std::vector<int64_t>& kStrides,
                                         const std::vector<int64_t>& vStrides,
                                         const std::vector<int64_t>& dOStrides,
                                         const std::vector<int64_t>& dVStrides,
                                         const std::vector<int64_t>& lseStrides,
                                         const std::vector<int64_t>& maskStrides,
                                         const std::vector<int64_t>& dBuffStrides,
                                         const std::vector<int64_t>& maskDims,
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
    auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefSdpaBwd.cpp", defines, "sdpaBwdRefdV");

    auto args = makeSdpaBwddQdKdVArgs(qPtr,
                                      kPtr,
                                      vPtr,
                                      dOPtr,
                                      lsePtr,
                                      maskPtr,
                                      dBuffPtr,
                                      qStrides,
                                      kStrides,
                                      vStrides,
                                      dOStrides,
                                      lseStrides,
                                      maskStrides,
                                      dBuffStrides,
                                      maskDims,
                                      batch,
                                      numHeads,
                                      numHeadsK,
                                      numHeadsV,
                                      seqQ,
                                      seqKv,
                                      headDim,
                                      headDimV,
                                      scale,
                                      leftBound,
                                      rightBound,
                                      topLeftAlignment);
    args.dV = dVPtr;
    args.dVStr = toSdpaStrides(dVStrides);

    detail::launchKernelForElements(
        kernel.function(), batch * numHeadsV * seqKv * headDimV, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
