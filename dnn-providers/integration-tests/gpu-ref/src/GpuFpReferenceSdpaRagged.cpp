// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
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

// Argument structs shared with the HipRTC kernel.
#include <GpuRefSdpaArgs.h> // NOLINT(misc-include-cleaner)

// Same helper as the dense launcher's.
SdpaStrides toSdpaStrides(const std::vector<int64_t>& strides)
{
    SdpaStrides result{};
    for(size_t i = 0; i < 4 && i < strides.size(); ++i)
    {
        result.s[i] = static_cast<long long>(strides[i]);
    }
    return result;
}

} // namespace

// --- Ragged offset read-back for host validation ---

std::vector<int64_t> GpuFpReferenceSdpaRagged::readRaggedOffsets(const void* raggedOffsetPtr,
                                                                 int64_t count)
{
    std::vector<int32_t> offsets(static_cast<size_t>(count));
    detail::throwOnHipError(hipMemcpy(offsets.data(),
                                      raggedOffsetPtr,
                                      offsets.size() * sizeof(int32_t),
                                      hipMemcpyDeviceToHost),
                            "failed to read ragged_offset");
    return {offsets.begin(), offsets.end()};
}

// --- Ragged SDPA forward kernel launcher ---

void GpuFpReferenceSdpaRagged::launchSdpaRaggedFwd(const void* qPtr,
                                                   const void* kPtr,
                                                   const void* vPtr,
                                                   void* oPtr,
                                                   void* lsePtr,
                                                   const void* raggedOffsetLsePtr,
                                                   const void* raggedOffsetQPtr,
                                                   const void* raggedOffsetKvPtr,
                                                   int64_t offsetMultiplierQ,
                                                   int64_t offsetMultiplierKv,
                                                   int64_t offsetMultiplierLse,
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
                                                   int64_t totalQ,
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

    SdpaRaggedFwdArgs args{};
    args.q = qPtr;
    args.k = kPtr;
    args.v = vPtr;
    args.o = oPtr;
    args.lse = lsePtr;
    args.raggedOffsetLse = static_cast<const int*>(raggedOffsetLsePtr);
    args.raggedOffsetQ = static_cast<const int*>(raggedOffsetQPtr);
    args.raggedOffsetKv = static_cast<const int*>(raggedOffsetKvPtr);
    args.offsetMultiplierQ = static_cast<long long>(offsetMultiplierQ);
    args.offsetMultiplierKv = static_cast<long long>(offsetMultiplierKv);
    args.offsetMultiplierLse = static_cast<long long>(offsetMultiplierLse);
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
    detail::launchKernelForElements(kernel.function(), totalElements, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
