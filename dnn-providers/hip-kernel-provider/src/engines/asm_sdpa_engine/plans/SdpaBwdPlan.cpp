// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/SdpaBwdPlan.hpp"
#include "plans/SdpaBwdArgsBuilder.hpp"
#include "plans/SdpaBwdLaunchParams.hpp"
#include "plans/SdpaKernelUtils.hpp"
#include "plans/SdpaPlanUtils.hpp"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>
#include <unordered_map>
#include <utility>

namespace
{

// Zeroes the tensor at `base` that `plan` describes (planBhsdZero()).
hipError_t zeroStrided(void* base, const asm_sdpa_engine::StridedZeroPlan& plan, hipStream_t stream)
{
    const auto& [outer, middle, inner] = plan.loops;
    for(size_t i = 0; i < outer.extent; ++i)
    {
        for(size_t j = 0; j < middle.extent; ++j)
        {
            for(size_t k = 0; k < inner.extent; ++k)
            {
                void* region = static_cast<char*>(base) + i * outer.strideBytes
                               + j * middle.strideBytes + k * inner.strideBytes;
                const hipError_t err = hipMemset2DAsync(
                    region, plan.pitchBytes, 0, plan.widthBytes, plan.height, stream);
                if(err != hipSuccess)
                {
                    return err;
                }
            }
        }
    }
    return hipSuccess;
}

} // anonymous namespace

namespace asm_sdpa_engine
{

// =============================================================================
// Constructors
// =============================================================================

SdpaBwdPlan::SdpaBwdPlan(CachedModule odoKernel,
                         CachedModule dqdkdvKernel,
                         std::optional<CachedModule> postKernel,
                         SdpaBwdParams params)
    : _odoKernel(std::move(odoKernel))
    , _dqdkdvKernel(std::move(dqdkdvKernel))
    , _postKernel(std::move(postKernel))
    , _params(std::move(params))
{
}

// =============================================================================
// getWorkspaceSize
// =============================================================================

size_t SdpaBwdPlan::getWorkspaceSize(const Handle& /*handle*/) const
{
    return sdpaBwdWorkspaceSize(_params.batchSize,
                                _params.numHeadsQ,
                                _params.seqLenQ,
                                _params.headDimQk,
                                _params.accumulatorType,
                                _postKernel.has_value());
}

// =============================================================================
// execute — ODO, DQDKDV, then the post kernel when there is one
// =============================================================================

void SdpaBwdPlan::execute(const Handle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    // 1. Validate workspace.  getMaxWorkspaceSize() always reports a non-zero
    // size for backward SDPA, so a null workspace pointer here is a contract
    // violation by the caller.
    if(workspace == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "SdpaBwdPlan::execute: workspace is null but backward SDPA requires a non-zero "
            "workspace (see getMaxWorkspaceSize())");
    }

    // 2. Build UID->ptr map from device buffers
    std::unordered_map<int64_t, void*> uidToPtrMap;
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    // 3. Resolve tensor pointers
    void* qPtr = uidToPtrMap.at(_params.qUid);
    void* kPtr = uidToPtrMap.at(_params.kUid);
    void* vPtr = uidToPtrMap.at(_params.vUid);
    void* oPtr = uidToPtrMap.at(_params.oUid);
    void* doPtr = uidToPtrMap.at(_params.doUid);
    void* lsePtr = uidToPtrMap.at(_params.statsUid);
    void* dqPtr = uidToPtrMap.at(_params.dqUid);
    void* dkPtr = uidToPtrMap.at(_params.dkUid);
    void* dvPtr = uidToPtrMap.at(_params.dvUid);

    // 4. Carve workspace into sub-buffers. Layout: D buffer | dq_acc, where
    // dq_acc exists only when a post kernel writes dQ from it.
    auto* dBufPtr = workspace;
    const size_t wsOffset
        = sdpaBwdDBufferSize(_params.batchSize, _params.numHeadsQ, _params.seqLenQ);
    const bool needPostProcessing = _postKernel.has_value();
    void* dqAccPtr = needPostProcessing ? static_cast<char*>(workspace) + wsOffset : nullptr;

    // 5. Resolve attention scale at execute time (runtime pass-by-value support).
    // Throws BAD_PARAM before the first launch for a scale the kernels mishandle.
    const float resolvedScale = plan_utils::resolveSupportedAttnScale(
        _params.attnScale, deviceBuffers, numDeviceBuffers, "SdpaBwdPlan::execute");

    // 6. Build convenience args struct (mirrors AITER mha_bwd_args).
    // Byte-stride uint32 overflow was already rejected by isApplicable (A32)
    // or buildPlan (A16).
    const MhaBwdArgs mhaArgs = buildMhaBwdArgs(_params,
                                               resolvedScale,
                                               qPtr,
                                               kPtr,
                                               vPtr,
                                               oPtr,
                                               doPtr,
                                               lsePtr,
                                               dqPtr,
                                               dkPtr,
                                               dvPtr,
                                               dBufPtr,
                                               dqAccPtr,
                                               needPostProcessing);

    // 7. Launch kernels on the same stream, which orders them without
    // explicit barriers.
    auto stream = handle.getStream();

    // 7a. Build args and launch kernel 1: ODO
    auto odoArgs = buildOdoArgs(mhaArgs);

    const auto odoLaunch = computeBwdLaunchParams(_params, SdpaBwdStage::ODO);

    if(!launchKernel("SDPA backward ODO",
                     _odoKernel->function(),
                     &odoArgs,
                     sizeof(odoArgs),
                     odoLaunch.gridDimX,
                     odoLaunch.gridDimY,
                     odoLaunch.gridDimZ,
                     odoLaunch.blockDimX,
                     stream))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlan::execute: hipModuleLaunchKernel failed for SDPA backward ODO");
    }

    // 7b. Build args and launch kernel 2: DQDKDV
    auto dqdkdvArgs = buildDqdkdvArgs(mhaArgs, _params.dqdkdvTiles.ts, _params);

    const auto dqdkdvLaunch = computeBwdLaunchParams(_params, SdpaBwdStage::DQDKDV);

    // DQDKDV adds each K/V tile's dQ contribution atomically into what it
    // writes and does not zero it first; stale data there would corrupt dQ.
    // - dq_acc: AITER allocates it with workspace_alloc(..., zero_init=true)
    //   (mha_bwd.cu:595).
    // - dQ, when DQDKDV writes it directly (gfx942 A16): AITER's Python layer
    //   zeroes dQ before every ASM backward (aiter/ops/mha.py:2885-2886).
    hipError_t memsetErr = hipSuccess;
    const char* zeroed = nullptr;
    if(needPostProcessing)
    {
        zeroed = "dq_acc workspace";
        memsetErr = hipMemsetAsync(dqAccPtr,
                                   0,
                                   sdpaBwdDqAccLayout(_params.batchSize,
                                                      _params.numHeadsQ,
                                                      _params.seqLenQ,
                                                      _params.headDimQk,
                                                      _params.accumulatorType)
                                       .bytes,
                                   stream);
    }
    else
    {
        zeroed = "dQ output";
        memsetErr = zeroStrided(
            dqPtr,
            planBhsdZero({_params.batchSize, _params.numHeadsQ, _params.seqLenQ},
                         {_params.dqStrideBatch, _params.dqStrideHead, _params.dqStrideSeq},
                         _params.headDimQk,
                         K_BWD_BF16_SIZE),
            stream);
    }
    if(memsetErr != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string("SdpaBwdPlan::execute: failed to zero the ") + zeroed
                + " before SDPA backward DQDKDV, error: " + hipGetErrorString(memsetErr));
    }

    if(!launchKernel("SDPA backward DQDKDV",
                     _dqdkdvKernel->function(),
                     &dqdkdvArgs,
                     sizeof(dqdkdvArgs),
                     dqdkdvLaunch.gridDimX,
                     dqdkdvLaunch.gridDimY,
                     dqdkdvLaunch.gridDimZ,
                     dqdkdvLaunch.blockDimX,
                     stream))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "SdpaBwdPlan::execute: hipModuleLaunchKernel failed for SDPA backward DQDKDV");
    }

    // 7c. Post kernel: DQ_CONVERT (A32) or DQ_SHUFFLE (gfx950 A16) writes dQ
    // from dq_acc (AITER mha_bwd.cu:752-759).
    if(needPostProcessing)
    {
        auto postArgs = buildPostArgs(mhaArgs);

        const auto postLaunch = computeBwdLaunchParams(_params, SdpaBwdStage::POST);

        if(!launchKernel("SDPA backward post",
                         (*_postKernel)->function(),
                         &postArgs,
                         sizeof(postArgs),
                         postLaunch.gridDimX,
                         postLaunch.gridDimY,
                         postLaunch.gridDimZ,
                         postLaunch.blockDimX,
                         stream))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "SdpaBwdPlan::execute: hipModuleLaunchKernel failed for the SDPA backward post "
                "kernel");
        }
    }
}

} // namespace asm_sdpa_engine
