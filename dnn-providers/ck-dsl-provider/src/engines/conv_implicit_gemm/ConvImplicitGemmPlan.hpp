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
/// argSchema) lives on the module, so the plan only needs the module
/// plus the three tensor UIDs to resolve device buffers at
/// ``execute()`` time.
///
/// **Step I-7 ships a stub** -- the plan stores the references and
/// reports zero workspace, but ``execute()`` throws. Wiring the
/// uid-keyed device-buffer lookup + LaunchAbi packing + HipModule
/// launch is step I-8.
class ConvImplicitGemmPlan : public hipdnn_plugin_sdk::IPlan<::CkDslHandle> {
   public:
    ConvImplicitGemmPlan(std::shared_ptr<HipModule> module, std::int64_t xUid, std::int64_t wUid,
                         std::int64_t yUid);

    std::size_t getWorkspaceSize(const ::CkDslHandle& handle) const override;

    /// **Stub for I-7** -- throws HipdnnPluginException to signal that
    /// the launch path lands in I-8. The stub still validates that
    /// the device-buffer array is non-null so misuse (call execute
    /// before I-8 ships) surfaces a clear "not implemented" message
    /// rather than a segfault.
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

   private:
    std::shared_ptr<HipModule> _module;
    std::int64_t _xUid;
    std::int64_t _wUid;
    std::int64_t _yUid;
};

}  // namespace ck_dsl_provider
