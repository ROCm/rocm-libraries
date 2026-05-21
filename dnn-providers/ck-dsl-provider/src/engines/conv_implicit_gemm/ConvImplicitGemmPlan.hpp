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

/// IPlan for one compiled implicit-GEMM convolution kernel.
///
/// The plan owns a shared reference to the ``HipModule`` produced by
/// ``JitCache``; the same module may back multiple plans built for
/// the same signature. All launch metadata (grid, block, ldsBytes,
/// argSchema) lives on the module. The plan additionally remembers
/// the three tensor UIDs (resolved to device pointers at execute
/// time) and the per-buffer byte sizes the kernel needs for its
/// buffer-rsrc OOB-clamping bounds (``A_bytes`` / ``B_bytes`` /
/// ``D_bytes`` in the conv kernel signature).
///
/// Byte sizes are computed at ``buildPlan`` time from the tensor dims
/// + dtype size, not at ``execute`` time -- the graph is not in scope
/// once execute runs, and the sizes are static per signature anyway.
class ConvImplicitGemmPlan : public hipdnn_plugin_sdk::IPlan<::CkDslHandle> {
   public:
    ConvImplicitGemmPlan(std::shared_ptr<HipModule> module, std::int64_t xUid, std::int64_t wUid,
                         std::int64_t yUid, std::int32_t xBytes, std::int32_t wBytes,
                         std::int32_t yBytes);

    std::size_t getWorkspaceSize(const ::CkDslHandle& handle) const override;

    /// Resolve X/W/Y device pointers from ``deviceBuffers`` by uid,
    /// pack the 6-arg kernel buffer (3 ptrs + 3 i32 bytes-bounds) per
    /// the module's ArgSchema, and launch via ``HipModule::launch`` on
    /// the handle's stream.
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` if a UID
    /// is missing from ``deviceBuffers`` or if the underlying HIP
    /// launch fails. Workspace is unused (this kernel allocates its
    /// scratch in static LDS).
    void execute(const ::CkDslHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 std::uint32_t numDeviceBuffers, void* workspace = nullptr) const override;

    /// Test-only: expose the loaded module so the I-7 plan-builder
    /// test can confirm the plan actually wraps a HipModule with the
    /// expected kernel name. Production callers use ``execute``.
    const HipModule& moduleForTesting() const {
        return *_module;
    }
    std::int64_t xUidForTesting() const {
        return _xUid;
    }
    std::int64_t yUidForTesting() const {
        return _yUid;
    }
    std::int32_t xBytesForTesting() const {
        return _xBytes;
    }

   private:
    std::shared_ptr<HipModule> _module;
    std::int64_t _xUid;
    std::int64_t _wUid;
    std::int64_t _yUid;
    std::int32_t _xBytes;
    std::int32_t _wBytes;
    std::int32_t _yBytes;
};

}  // namespace ck_dsl_provider
