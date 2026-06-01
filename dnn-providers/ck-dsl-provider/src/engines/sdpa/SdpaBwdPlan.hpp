// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstddef>
#include <cstdint>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <memory>

#include "../../CkDslHandle.hpp"
#include "../../runtime/HipModule.hpp"

namespace ck_dsl_provider {

/// IPlan for one compiled FMHA-backward kernel plus the LSE-prep kernel
/// that precedes it.
///
/// The plan owns shared references to the two ``HipModule`` instances
/// produced by ``JitCache``: ``_bwdModule`` (the main backward kernel)
/// and ``_prepModule`` (the LSE-prep pass that derives the M/L scratch
/// from the natural-log stats buffer). All launch metadata (grid, block,
/// ldsBytes, argSchema) lives on each module. The plan additionally
/// remembers the eight tensor UIDs (resolved to device pointers at
/// execute time), the problem dims, the two attention scales, and the
/// launch-time strides the kernels consume.
///
/// **Schema-driven packing.** The plan packs its arguments per launch
/// via ``LaunchAbi::pack`` against each module's argSchema. The schemas
/// are validated in the ctor:
///   * bwd: 24 slots -- 9 Pointer (Q, K, V, dO, M, L, dQ, dK, dV),
///     2 F32 (scale_log2, scale_inv), 13 I32 (seqlen_q, seqlen_k, then
///     the eleven token/head strides)
///   * prep: 6 slots -- 3 Pointer (stats, M, L), 3 I32 (B, Hq, Sq)
///
/// **Workspace.** The bwd path needs a host-managed scratch buffer for
/// the two per-(B,Hq,Sq) reductions the prep kernel writes (M_saved and
/// L_saved). ``getWorkspaceSize`` returns ``2 * B * Sq * Hq *
/// sizeof(float)``.
class SdpaBwdPlan : public hipdnn_plugin_sdk::IPlan<::CkDslHandle> {
   public:
    SdpaBwdPlan(std::shared_ptr<HipModule> bwdModule, std::shared_ptr<HipModule> prepModule,
                std::int64_t qUid, std::int64_t kUid, std::int64_t vUid, std::int64_t doUid,
                std::int64_t statsUid, std::int64_t dqUid, std::int64_t dkUid, std::int64_t dvUid,
                std::int32_t B, std::int32_t Hq, std::int32_t Hkv, std::int32_t Sq,
                std::int32_t Skv, std::int32_t D, float scaleLog2, float scaleInv,
                std::int32_t strideQToken, std::int32_t strideQHead, std::int32_t strideKToken,
                std::int32_t strideKHead, std::int32_t strideVToken, std::int32_t strideVHead,
                std::int32_t strideDoToken, std::int32_t strideDoHead, std::int32_t strideDqToken,
                std::int32_t strideDkToken, std::int32_t strideDvToken);

    /// Scratch for the two per-(B, Hq, Sq) reductions the prep kernel
    /// writes (M_saved + L_saved), each ``B * Sq * Hq`` floats.
    std::size_t getWorkspaceSize(const ::CkDslHandle& handle) const override;

    /// Resolve the eight device pointers by uid, carve the M/L scratch
    /// out of ``workspace``, zero-initialise the gradient buffers, run
    /// the LSE-prep kernel, then launch the backward kernel once per
    /// batch (the kernels carry no batch stride, so the host folds the
    /// batch offset into the base pointers).
    ///
    /// All launches are enqueued on the handle's stream, so their
    /// ordering is guaranteed: prep before the bwd reads M/L, and the
    /// zero-init before the bwd accumulates into the gradients.
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` if a UID is
    /// missing from ``deviceBuffers``, if ``workspace`` is null while a
    /// nonzero scratch is required, or if a HIP call fails.
    void execute(const ::CkDslHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 std::uint32_t numDeviceBuffers, void* workspace = nullptr) const override;

    /// Test-only accessors. Production callers use ``execute``.
    const HipModule& bwdModuleForTesting() const {
        return *_bwdModule;
    }
    const HipModule& prepModuleForTesting() const {
        return *_prepModule;
    }
    std::int64_t qUidForTesting() const {
        return _qUid;
    }
    std::int64_t statsUidForTesting() const {
        return _statsUid;
    }
    std::int64_t dqUidForTesting() const {
        return _dqUid;
    }
    std::int64_t dkUidForTesting() const {
        return _dkUid;
    }
    std::int64_t dvUidForTesting() const {
        return _dvUid;
    }
    float scaleLog2ForTesting() const {
        return _scaleLog2;
    }
    float scaleInvForTesting() const {
        return _scaleInv;
    }

   private:
    std::shared_ptr<HipModule> _bwdModule;
    std::shared_ptr<HipModule> _prepModule;
    std::int64_t _qUid;
    std::int64_t _kUid;
    std::int64_t _vUid;
    std::int64_t _doUid;
    std::int64_t _statsUid;
    std::int64_t _dqUid;
    std::int64_t _dkUid;
    std::int64_t _dvUid;
    std::int32_t _B;
    std::int32_t _Hq;
    std::int32_t _Hkv;
    std::int32_t _Sq;
    std::int32_t _Skv;
    std::int32_t _D;
    float _scaleLog2;
    float _scaleInv;
    std::int32_t _strideQToken;
    std::int32_t _strideQHead;
    std::int32_t _strideKToken;
    std::int32_t _strideKHead;
    std::int32_t _strideVToken;
    std::int32_t _strideVHead;
    std::int32_t _strideDoToken;
    std::int32_t _strideDoHead;
    std::int32_t _strideDqToken;
    std::int32_t _strideDkToken;
    std::int32_t _strideDvToken;
};

}  // namespace ck_dsl_provider
