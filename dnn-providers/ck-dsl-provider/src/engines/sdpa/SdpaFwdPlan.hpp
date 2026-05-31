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

/// IPlan for one compiled FMHA-forward kernel.
///
/// The plan owns a shared reference to the ``HipModule`` produced by
/// ``JitCache``; the same module may back multiple plans built for the
/// same signature. All launch metadata (grid, block, ldsBytes,
/// argSchema) lives on the module. The plan additionally remembers the
/// four tensor UIDs (resolved to device pointers at execute time) and
/// the launch-time scalars the kernel consumes: the attention scale (in
/// log2 space), the two sequence lengths, and the eight token/head
/// strides for Q/K/V/O.
///
/// **Schema-driven packing.** The plan packs its arguments per call via
/// ``LaunchAbi::pack`` against the module's argSchema. The kernel's ABI is
/// validated in the ctor:
///   slots 0..3  -> Pointer (Q, K, V, O)
///   slot  4     -> F32     (scale_log2)
///   slots 5..14 -> I32     (seqlen_q, seqlen_k, then the eight strides)
///   slot  15    -> Pointer (LSE_out) -- present iff opt-in stats are
///                  enabled (a 16-slot schema); otherwise the schema is
///                  the byte-identical 15-slot ABI.
///
/// **Opt-in stats (LSE).** When ``hasStats`` is set the kernel writes the
/// natural-log LSE to a head-major [B, Hq, Sq] f32 buffer resolved by
/// ``statsUid`` at execute time and bound to the 16th arg slot. The
/// non-stats path is unchanged (15 args, no LSE pointer resolved).
class SdpaFwdPlan : public hipdnn_plugin_sdk::IPlan<::CkDslHandle> {
   public:
    SdpaFwdPlan(std::shared_ptr<HipModule> module, std::int64_t qUid, std::int64_t kUid,
                std::int64_t vUid, std::int64_t oUid, float scaleLog2, std::int32_t seqlenQ,
                std::int32_t seqlenK, std::int32_t strideQToken, std::int32_t strideQHead,
                std::int32_t strideKToken, std::int32_t strideKHead, std::int32_t strideVToken,
                std::int32_t strideVHead, std::int32_t strideOToken, std::int32_t strideOHead,
                bool hasStats = false, std::int64_t statsUid = -1);

    std::size_t getWorkspaceSize(const ::CkDslHandle& handle) const override;

    /// Resolve Q/K/V/O device pointers from ``deviceBuffers`` by uid (plus
    /// the LSE_out pointer by ``statsUid`` when opt-in stats are enabled),
    /// pack the 15- or 16-slot argument buffer via ``LaunchAbi::pack``, and
    /// launch via ``HipModule::launch`` on the handle's stream.
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` if a UID is
    /// missing from ``deviceBuffers`` or if the underlying HIP launch
    /// fails. Workspace is unused (this kernel allocates its scratch in
    /// static LDS).
    void execute(const ::CkDslHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 std::uint32_t numDeviceBuffers, void* workspace = nullptr) const override;

    /// Test-only: expose the loaded module so the plan-builder test can
    /// confirm the plan actually wraps a HipModule with the expected
    /// kernel name. Production callers use ``execute``.
    const HipModule& moduleForTesting() const {
        return *_module;
    }
    std::int64_t qUidForTesting() const {
        return _qUid;
    }
    std::int64_t oUidForTesting() const {
        return _oUid;
    }
    float scaleLog2ForTesting() const {
        return _scaleLog2;
    }

   private:
    std::shared_ptr<HipModule> _module;
    std::int64_t _qUid;
    std::int64_t _kUid;
    std::int64_t _vUid;
    std::int64_t _oUid;
    float _scaleLog2;
    std::int32_t _seqlenQ;
    std::int32_t _seqlenK;
    std::int32_t _strideQToken;
    std::int32_t _strideQHead;
    std::int32_t _strideKToken;
    std::int32_t _strideKHead;
    std::int32_t _strideVToken;
    std::int32_t _strideVHead;
    std::int32_t _strideOToken;
    std::int32_t _strideOHead;
    bool _hasStats;
    std::int64_t _statsUid;
};

}  // namespace ck_dsl_provider
