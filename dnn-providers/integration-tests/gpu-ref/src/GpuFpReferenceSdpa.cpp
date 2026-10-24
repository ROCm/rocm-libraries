// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
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

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefSdpaArgs.h> // NOLINT(misc-include-cleaner)

// RAII fp32 device scratch for the backward row stats.
struct DeviceScratch
{
    explicit DeviceScratch(int64_t count)
    {
        detail::throwOnHipError(
            hipMalloc(&data, static_cast<size_t>(count > 0 ? count : 1) * sizeof(float)),
            "hipMalloc");
    }
    ~DeviceScratch()
    {
        static_cast<void>(hipFree(data));
    }
    DeviceScratch(const DeviceScratch&) = delete;
    DeviceScratch& operator=(const DeviceScratch&) = delete;
    DeviceScratch(DeviceScratch&&) = delete;
    DeviceScratch& operator=(DeviceScratch&&) = delete;
    float* data = nullptr;
};

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
                                       const std::vector<std::string>& defines,
                                       const int32_t* seqLenQPtr,
                                       const int32_t* seqLenKvPtr,
                                       int64_t seqLenQStride,
                                       int64_t seqLenKvStride,
                                       const SdpaPagedSide& pagedK,
                                       const SdpaPagedSide& pagedV,
                                       const float* sinkPtr,
                                       int64_t sinkStride,
                                       const float* dropoutMaskPtr,
                                       const std::vector<int64_t>& dropoutMaskStrides,
                                       float dropoutScale,
                                       float softcap,
                                       void* rowMaxPtr,
                                       const std::vector<int64_t>& rowMaxStrides,
                                       void* rowSumExpPtr,
                                       const std::vector<int64_t>& rowSumExpStrides,
                                       bool alibi,
                                       const SdpaFp8Softmax& fp8)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefSdpaFwd.cpp", defines, "sdpaFwdRef");

    SdpaFwdArgs args{};
    args.q = qPtr;
    args.k = kPtr;
    args.v = vPtr;
    args.score.mask = maskPtr;
    args.o = oPtr;
    args.lse = lsePtr;
    args.qStr = toSdpaStrides(qTensorStrides);
    args.kStr = toSdpaStrides(kTensorStrides);
    args.vStr = toSdpaStrides(vTensorStrides);
    args.oStr = toSdpaStrides(oTensorStrides);
    args.score.maskStr = toSdpaStrides(maskTensorStrides);
    args.lseStr = toSdpaStrides(lseTensorStrides);
    args.batch = static_cast<long long>(batch);
    args.numHeads = static_cast<long long>(numHeads);
    args.numHeadsK = static_cast<long long>(numHeadsK);
    args.numHeadsV = static_cast<long long>(numHeadsV);
    args.seqQ = static_cast<long long>(seqQ);
    args.seqKv = static_cast<long long>(seqKv);
    args.headDim = static_cast<long long>(headDim);
    args.headDimV = static_cast<long long>(headDimV);
    args.score.maskRank = static_cast<int>(maskDims.size());
    for(size_t i = 0; i < 4 && i < maskDims.size(); ++i)
    {
        args.score.maskDims[i] = static_cast<long long>(maskDims[i]);
    }
    args.score.scale = scale;
    args.score.leftBound = static_cast<long long>(leftBound);
    args.score.rightBound = static_cast<long long>(rightBound);
    args.score.topLeftAlignment = topLeftAlignment ? 1 : 0;
    args.seqLenQ = seqLenQPtr;
    args.seqLenKv = seqLenKvPtr;
    args.seqLenQStride = static_cast<long long>(seqLenQStride);
    args.seqLenKvStride = static_cast<long long>(seqLenKvStride);
    args.pageTableK = pagedK.table;
    args.pageTableKStrB = static_cast<long long>(pagedK.strideBatch);
    args.pageTableKStrT = static_cast<long long>(pagedK.strideTable);
    args.blockSizeK = static_cast<long long>(pagedK.blockSize);
    args.pageTableV = pagedV.table;
    args.pageTableVStrB = static_cast<long long>(pagedV.strideBatch);
    args.pageTableVStrT = static_cast<long long>(pagedV.strideTable);
    args.blockSizeV = static_cast<long long>(pagedV.blockSize);
    args.sink = sinkPtr;
    args.sinkStride = static_cast<long long>(sinkStride);
    args.dropoutMask = dropoutMaskPtr;
    args.dropoutMaskStr = toSdpaStrides(dropoutMaskStrides);
    args.dropoutScale = dropoutScale;
    args.score.softcap = softcap;
    args.rowMax = rowMaxPtr;
    args.rowSumExp = rowSumExpPtr;
    args.rowMaxStr = toSdpaStrides(rowMaxStrides);
    args.rowSumExpStr = toSdpaStrides(rowSumExpStrides);
    args.score.alibi = alibi ? 1 : 0;
    args.pFp8Format = static_cast<int>(fp8.format);
    args.scaleS = fp8.scaleS;
    args.descaleS = fp8.descaleS;
    args.amaxS = fp8.amaxS;
    if(fp8.amaxS != nullptr)
    {
        // The kernel max-reduces into amaxS (values >= 0), so it starts at 0.
        detail::throwOnHipError(hipMemset(fp8.amaxS, 0, sizeof(float)), "hipMemset");
    }

    auto totalElements = batch * numHeads * seqQ * headDimV;
    detail::launchKernelForElements(kernel.function(), totalElements, &args, sizeof(args));
}

// --- SDPA backward kernel launcher ---

void GpuFpReferenceSdpa::launchSdpaBwd(const SdpaBwdLaunch& launch,
                                       const std::vector<std::string>& defines)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto rows = launch.batch * launch.numHeads * launch.seqQ;
    // Row-stats scratch: D, and the LSE or the recomputed max and exp-sum (fp32 [B, H, Sq]).
    const DeviceScratch rowD(rows);
    const DeviceScratch rowM(rows);
    const DeviceScratch rowL(rows);

    SdpaBwdArgs args{};
    args.q = launch.q;
    args.k = launch.k;
    args.v = launch.v;
    args.o = launch.o;
    args.dO = launch.dO;
    args.score.mask = launch.mask;
    args.lse = launch.lse;
    args.dQ = launch.dQ;
    args.dK = launch.dK;
    args.dV = launch.dV;
    args.dBias = launch.dBias;
    args.rowD = rowD.data;
    args.rowM = rowM.data;
    args.rowL = rowL.data;
    const auto& st = launch.strides;
    args.qStr = toSdpaStrides(st[0]);
    args.kStr = toSdpaStrides(st[1]);
    args.vStr = toSdpaStrides(st[2]);
    args.oStr = toSdpaStrides(st[3]);
    args.dOStr = toSdpaStrides(st[4]);
    args.dQStr = toSdpaStrides(st[5]);
    args.dKStr = toSdpaStrides(st[6]);
    args.dVStr = toSdpaStrides(st[7]);
    args.score.maskStr = toSdpaStrides(st[8]);
    args.dBiasStr = toSdpaStrides(st[9]);
    args.lseStr = toSdpaStrides(st[10]);
    args.batch = static_cast<long long>(launch.batch);
    args.numHeads = static_cast<long long>(launch.numHeads);
    args.numHeadsK = static_cast<long long>(launch.numHeadsK);
    args.numHeadsV = static_cast<long long>(launch.numHeadsV);
    args.seqQ = static_cast<long long>(launch.seqQ);
    args.seqKv = static_cast<long long>(launch.seqKv);
    args.headDim = static_cast<long long>(launch.headDim);
    args.headDimV = static_cast<long long>(launch.headDimV);
    args.score.maskRank = static_cast<int>(launch.maskDims.size());
    for(size_t i = 0; i < launch.maskDims.size() && i < 4; ++i)
    {
        args.score.maskDims[i] = static_cast<long long>(launch.maskDims[i]);
    }
    args.score.scale = launch.scale;
    args.score.leftBound = static_cast<long long>(launch.leftBound);
    args.score.rightBound = static_cast<long long>(launch.rightBound);
    args.score.topLeftAlignment = launch.topLeftAlignment ? 1 : 0;
    args.seqLenQ = launch.seqLenQ;
    args.seqLenKv = launch.seqLenKv;
    args.seqLenQStride = static_cast<long long>(launch.seqLenQStride);
    args.seqLenKvStride = static_cast<long long>(launch.seqLenKvStride);
    args.dropoutMask = launch.dropoutMask;
    args.dropoutMaskStr = toSdpaStrides(launch.dropoutMaskStrides);
    args.dropoutScale = launch.dropoutScale;
    args.dropoutScaleInv = launch.dropoutScaleInv;
    args.score.softcap = launch.softcap;
    args.score.alibi = launch.alibi ? 1 : 0;

    const auto run = [&](const char* name, int64_t elements) {
        auto& kernel = compiler.getOrCompile("GpuRefSdpaBwd.cpp", defines, name);
        detail::launchKernelForElements(kernel.function(), elements, &args, sizeof(args));
    };
    run("sdpaBwdRowStatsRef", rows);
    run("sdpaBwdDqRef", rows * launch.headDim);
    run("sdpaBwdDkRef", launch.batch * launch.numHeadsK * launch.seqKv * launch.headDim);
    run("sdpaBwdDvRef", launch.batch * launch.numHeadsV * launch.seqKv * launch.headDimV);
    if(launch.dBias != nullptr)
    {
        int64_t count = 1;
        for(const auto dim : launch.maskDims)
        {
            count *= dim;
        }
        run("sdpaBwdDBiasRef", count);
    }
    // The scratch buffers are freed on return; finish the launches that read them first.
    detail::throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

// --- fp8 dequantize launcher ---

void GpuFpReferenceSdpa::launchSdpaDequant(const void* in,
                                           const std::vector<int64_t>& dims,
                                           const std::vector<int64_t>& strides,
                                           float* out,
                                           const float* descale,
                                           int64_t descaleBatchStride,
                                           int64_t descaleHeadStride,
                                           int64_t headsPerKv,
                                           const char* inType)
{
    const std::vector<std::string> defines
        = {std::string("-DIN_TYPE=") + inType, std::string("-DCOMPUTE_TYPE=float")};
    auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefSdpaFp8.cpp", defines, "sdpaDequantRef");

    SdpaDequantArgs args{};
    args.in = in;
    args.out = out;
    args.inStr = toSdpaStrides(strides);
    int64_t elements = 1;
    for(size_t i = 0; i < 4; ++i)
    {
        args.dims[i] = static_cast<long long>(dims[i]);
        elements *= dims[i];
    }
    args.descale = descale;
    args.descaleBatchStride = static_cast<long long>(descaleBatchStride);
    args.descaleHeadStride = static_cast<long long>(descaleHeadStride);
    args.headsPerKv = static_cast<long long>(headsPerKv);
    detail::launchKernelForElements(kernel.function(), elements, &args, sizeof(args));
}

// --- fp8 output scale launcher ---

void GpuFpReferenceSdpa::launchSdpaScaleOut(const float* in,
                                            const std::vector<int64_t>& dims,
                                            const std::vector<int64_t>& outStrides,
                                            void* out,
                                            float scale,
                                            float* amax,
                                            const char* outType)
{
    const std::vector<std::string> defines
        = {std::string("-DO_TYPE=") + outType, std::string("-DCOMPUTE_TYPE=float")};
    auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefSdpaFp8.cpp", defines, "sdpaScaleOutRef");

    SdpaScaleOutArgs args{};
    args.in = in;
    args.out = out;
    args.outStr = toSdpaStrides(outStrides);
    int64_t elements = 1;
    for(size_t i = 0; i < 4; ++i)
    {
        args.dims[i] = static_cast<long long>(dims[i]);
        elements *= dims[i];
    }
    args.scale = scale;
    args.amax = amax;
    if(amax != nullptr)
    {
        // The kernel max-reduces |O_f| into amax, so it starts at 0.
        detail::throwOnHipError(hipMemset(amax, 0, sizeof(float)), "hipMemset");
    }
    detail::launchKernelForElements(kernel.function(), elements, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
