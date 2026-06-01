// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <array>
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
/// **Pre-packed argument buffer.** The plan ctor validates that the
/// kernel's argSchema is exactly ``(Pointer, Pointer, Pointer, I32,
/// I32, I32)`` and constructs a 36-byte template containing the three
/// i32 byte counts at their natural-alignment offsets. ``execute`` then
/// copies the template into a stack-resident scratch buffer, patches
/// in the three resolved pointers at offsets 0/8/16, and launches.
/// This avoids two heap allocations (initializer-list vector +
/// LaunchAbi::pack's returned vector) that the generic packer would
/// have done on every call.
///
/// Byte sizes are computed at ``buildPlan`` time from the tensor dims
/// + dtype size, not at ``execute`` time -- the graph is not in scope
/// once execute runs, and the sizes are static per signature anyway.
class ConvImplicitGemmPlan : public hipdnn_plugin_sdk::IPlan<::CkDslHandle> {
   public:
    /// Layout constants for the pre-packed argument buffer.
    /// 3 pointers (8 bytes each, naturally aligned) followed by 3
    /// i32s (4 bytes each, naturally aligned). Total 36 bytes.
    static constexpr std::size_t kArgBufferSize = 36;
    static constexpr std::size_t kXPtrOffset = 0;
    static constexpr std::size_t kWPtrOffset = 8;
    static constexpr std::size_t kYPtrOffset = 16;
    static constexpr std::size_t kXBytesOffset = 24;
    static constexpr std::size_t kWBytesOffset = 28;
    static constexpr std::size_t kYBytesOffset = 32;

    ConvImplicitGemmPlan(std::shared_ptr<HipModule> module, std::int64_t xUid, std::int64_t wUid,
                         std::int64_t yUid, std::int32_t xBytes, std::int32_t wBytes,
                         std::int32_t yBytes);

    std::size_t getWorkspaceSize(const ::CkDslHandle& handle) const override;

    /// Resolve X/W/Y device pointers from ``deviceBuffers`` by uid,
    /// patch them into the pre-packed argument template, and launch
    /// via ``HipModule::launch`` on the handle's stream.
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException`` if a UID
    /// is missing from ``deviceBuffers`` or if the underlying HIP
    /// launch fails. Workspace is unused (this kernel allocates its
    /// scratch in static LDS).
    void execute(const ::CkDslHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 std::uint32_t numDeviceBuffers, void* workspace = nullptr) const override;

    /// Test-only: expose the loaded module so the plan-builder test
    /// can confirm the plan actually wraps a HipModule with the
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

    /// 36-byte launch-arg template with the three i32 byte counts
    /// pre-written at offsets 24/28/32. The three pointer slots
    /// (offsets 0/8/16) stay zero in the template; ``execute`` copies
    /// the template into a stack scratch buffer and patches in the
    /// per-call pointers there.
    std::array<std::byte, kArgBufferSize> _argTemplate{};
};

}  // namespace ck_dsl_provider
