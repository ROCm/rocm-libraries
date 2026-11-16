// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <stdexcept>
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

// --- Ragged offset and seq_len read-back for host validation ---

std::vector<int64_t> GpuFpReferenceSdpaRagged::readStridedValues(const void* ptr,
                                                                 int64_t count,
                                                                 int64_t stride,
                                                                 const std::string& name)
{
    const auto span = hipdnn_test_sdk::detail::stridedAuxSpan(count, stride, name);
    if(span == 0)
    {
        return {};
    }
    std::vector<int32_t> raw(span);
    detail::throwOnHipError(
        hipMemcpy(raw.data(), ptr, raw.size() * sizeof(int32_t), hipMemcpyDeviceToHost),
        "failed to read ragged_offset or seq_len");
    return hipdnn_test_sdk::detail::gatherStridedAux(raw.data(), count, stride, name);
}

// --- Ragged SDPA forward kernel launcher ---

void GpuFpReferenceSdpaRagged::launchSdpaRaggedFwd(const RaggedLaunchTensors& tensors,
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
                                                   int64_t seqMaxQ,
                                                   int64_t seqMaxKv,
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
    args.q = tensors.q;
    args.k = tensors.k;
    args.v = tensors.v;
    args.o = tensors.o;
    args.lse = tensors.lse;
    args.raggedOffsetQ = static_cast<const int*>(tensors.offsetQ);
    args.raggedOffsetK = static_cast<const int*>(tensors.offsetK);
    args.raggedOffsetV = static_cast<const int*>(tensors.offsetV);
    args.raggedOffsetO = static_cast<const int*>(tensors.offsetO);
    args.raggedOffsetLse = static_cast<const int*>(tensors.offsetLse);
    args.offsetMultiplierQ = static_cast<long long>(tensors.multipliers.q);
    args.offsetMultiplierK = static_cast<long long>(tensors.multipliers.k);
    args.offsetMultiplierV = static_cast<long long>(tensors.multipliers.v);
    args.offsetMultiplierO = static_cast<long long>(tensors.multipliers.o);
    args.offsetMultiplierLse = static_cast<long long>(tensors.multipliers.lse);
    args.offsetStrideQ = static_cast<long long>(tensors.offsetStrideQ);
    args.offsetStrideK = static_cast<long long>(tensors.offsetStrideK);
    args.offsetStrideV = static_cast<long long>(tensors.offsetStrideV);
    args.offsetStrideO = static_cast<long long>(tensors.offsetStrideO);
    args.offsetStrideLse = static_cast<long long>(tensors.offsetStrideLse);
    args.seqLenQ = static_cast<const int*>(tensors.seqLenQ);
    args.seqLenKv = static_cast<const int*>(tensors.seqLenKv);
    args.seqLenQStride = static_cast<long long>(tensors.seqLenQStride);
    args.seqLenKvStride = static_cast<long long>(tensors.seqLenKvStride);
    args.seqMaxQ = static_cast<long long>(seqMaxQ);
    args.seqMaxKv = static_cast<long long>(seqMaxKv);
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
    args.numHeads = static_cast<long long>(numHeads);
    args.numHeadsK = static_cast<long long>(numHeadsK);
    args.numHeadsV = static_cast<long long>(numHeadsV);
    args.headDim = static_cast<long long>(headDim);
    args.headDimV = static_cast<long long>(headDimV);
    args.scale = scale;
    args.leftBound = static_cast<long long>(leftBound);
    args.rightBound = static_cast<long long>(rightBound);
    args.topLeftAlignment = topLeftAlignment ? 1 : 0;

    auto totalElements = batch * seqMaxQ * numHeads * headDimV;
    detail::launchKernelForElements(kernel.function(), totalElements, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
