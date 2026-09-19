// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>

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

// Ragged q/k/v/o are rank-4; copy up to 4 strides, leaving unused entries zero. Mirrors the dense
// launcher's helper (kept file-local to avoid coupling the two units).
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

// --- Ragged SDPA forward kernel launcher ---

void GpuFpReferenceSdpaRagged::launchSdpaRaggedFwd(const void* qPtr,
                                                   const void* kPtr,
                                                   const void* vPtr,
                                                   void* oPtr,
                                                   void* lsePtr,
                                                   const void* raggedOffsetQPtr,
                                                   const void* raggedOffsetKvPtr,
                                                   int64_t seqStrideQ,
                                                   int64_t seqStrideKv,
                                                   const void* descaleQPtr,
                                                   int64_t descaleQBatchStride,
                                                   int64_t descaleQHeadStride,
                                                   const void* descaleKPtr,
                                                   int64_t descaleKBatchStride,
                                                   int64_t descaleKHeadStride,
                                                   const void* descaleVPtr,
                                                   int64_t descaleVBatchStride,
                                                   int64_t descaleVHeadStride,
                                                   const std::vector<int64_t>& qTensorStrides,
                                                   const std::vector<int64_t>& kTensorStrides,
                                                   const std::vector<int64_t>& vTensorStrides,
                                                   const std::vector<int64_t>& oTensorStrides,
                                                   const std::vector<int64_t>& lseTensorStrides,
                                                   int64_t batch,
                                                   int64_t numHeads,
                                                   int64_t numHeadsK,
                                                   int64_t numHeadsV,
                                                   int64_t headDim,
                                                   int64_t headDimV,
                                                   float scale,
                                                   int64_t leftBound,
                                                   int64_t rightBound,
                                                   bool topLeftAlignment,
                                                   const std::vector<std::string>& defines)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefSdpaRaggedFwd.cpp", defines, "sdpaRaggedFwdRef");

    // total_q (packed query-token count) = ragged_offset[B] / seqStrideQ, read from device so this
    // works whether the aux is a host-backed Tensor or a device-only view (plan path).
    int lastOffsetQ = 0;
    detail::throwOnHipError(hipMemcpy(&lastOffsetQ,
                                      static_cast<const int*>(raggedOffsetQPtr) + batch,
                                      sizeof(int),
                                      hipMemcpyDeviceToHost),
                            "failed to read ragged_offset[B]");
    const int64_t totalQ = static_cast<int64_t>(lastOffsetQ) / seqStrideQ;

    SdpaRaggedFwdArgs args{};
    args.q = qPtr;
    args.k = kPtr;
    args.v = vPtr;
    args.o = oPtr;
    args.lse = lsePtr;
    args.raggedOffsetQ = static_cast<const int*>(raggedOffsetQPtr);
    args.raggedOffsetKv = static_cast<const int*>(raggedOffsetKvPtr);
    args.seqStrideQ = static_cast<long long>(seqStrideQ);
    args.seqStrideKv = static_cast<long long>(seqStrideKv);
    args.descaleQ = static_cast<const float*>(descaleQPtr);
    args.descaleK = static_cast<const float*>(descaleKPtr);
    args.descaleV = static_cast<const float*>(descaleVPtr);
    args.descaleQBatchStride = static_cast<long long>(descaleQBatchStride);
    args.descaleQHeadStride = static_cast<long long>(descaleQHeadStride);
    args.descaleKBatchStride = static_cast<long long>(descaleKBatchStride);
    args.descaleKHeadStride = static_cast<long long>(descaleKHeadStride);
    args.descaleVBatchStride = static_cast<long long>(descaleVBatchStride);
    args.descaleVHeadStride = static_cast<long long>(descaleVHeadStride);
    args.qStr = toSdpaStrides(qTensorStrides);
    args.kStr = toSdpaStrides(kTensorStrides);
    args.vStr = toSdpaStrides(vTensorStrides);
    args.oStr = toSdpaStrides(oTensorStrides);
    args.lseStr = toSdpaStrides(lseTensorStrides);
    args.batch = static_cast<long long>(batch);
    args.totalQ = static_cast<long long>(totalQ);
    args.numHeads = static_cast<long long>(numHeads);
    args.numHeadsK = static_cast<long long>(numHeadsK);
    args.numHeadsV = static_cast<long long>(numHeadsV);
    args.headDim = static_cast<long long>(headDim);
    args.headDimV = static_cast<long long>(headDimV);
    args.scale = scale;
    args.leftBound = static_cast<long long>(leftBound);
    args.rightBound = static_cast<long long>(rightBound);
    args.topLeftAlignment = topLeftAlignment ? 1 : 0;

    auto totalElements = totalQ * numHeads * headDimV;
    launchKernel(kernel.function(), totalElements, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
