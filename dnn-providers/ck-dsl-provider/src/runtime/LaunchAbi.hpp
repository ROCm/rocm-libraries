// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "KernelArtifact.hpp"

namespace ck_dsl_provider {

/// Caller-supplied value for one kernel argument slot. The variant tag
/// must match the corresponding ``ArgSchema::Kind`` produced by the
/// compile service; ``LaunchAbi::pack`` cross-checks them and rejects
/// any mismatch (so a caller cannot silently pass an I32 where the
/// kernel expects a Pointer and corrupt the launch).
///
/// Pointers are passed as ``void*`` so the caller can hand in device
/// allocations directly without casting. The packer copies the raw 8
/// bytes; no dereference happens on the host side.
struct ArgValue {
    enum class Tag : std::uint8_t {
        Pointer,
        I32,
        I64,
        F32,
        F16,
    };

    Tag tag{Tag::Pointer};

    /// Underlying scalar / pointer payload. F16 stored as the raw
    /// 16-bit bit pattern (uint16_t) so the host code doesn't need to
    /// depend on _Float16 / __fp16 availability for the wire format.
    std::variant<void*, std::int32_t, std::int64_t, float, std::uint16_t> value{};

    static ArgValue pointer(void* ptr) {
        ArgValue v;
        v.tag = Tag::Pointer;
        v.value = ptr;
        return v;
    }

    static ArgValue i32(std::int32_t x) {
        ArgValue v;
        v.tag = Tag::I32;
        v.value = x;
        return v;
    }

    static ArgValue i64(std::int64_t x) {
        ArgValue v;
        v.tag = Tag::I64;
        v.value = x;
        return v;
    }

    static ArgValue f32(float x) {
        ArgValue v;
        v.tag = Tag::F32;
        v.value = x;
        return v;
    }

    static ArgValue f16FromBits(std::uint16_t bits) {
        ArgValue v;
        v.tag = Tag::F16;
        v.value = bits;
        return v;
    }
};

/// Schema-driven kernel-argument packing.
///
/// hipModuleLaunchKernel's "extras" path
/// (``HIP_LAUNCH_PARAM_BUFFER_POINTER`` /
/// ``HIP_LAUNCH_PARAM_BUFFER_SIZE`` / ``HIP_LAUNCH_PARAM_END``) expects
/// a single contiguous byte buffer whose layout matches the AMDGPU
/// host-side calling convention -- naturally-aligned, packed args, no
/// trailing padding required by the runtime itself. The existing DSL
/// launcher (``projects/composablekernel/example/ck_tile/dsl/common/
/// launcher.cpp``) bakes that layout per-kind; ``LaunchAbi::pack``
/// generalises it so any artifact whose ``argSchema`` matches the
/// caller's ``ArgValue`` list packs correctly without per-kind code.
class LaunchAbi {
   public:
    /// Pack one launch's arguments into a buffer ready to hand to
    /// hipModuleLaunchKernel via the BUFFER_POINTER extras path.
    ///
    /// Validates:
    ///   * ``schema.size() == values.size()``
    ///   * each ``values[i].tag`` matches the corresponding
    ///     ``schema[i].kind``
    ///   * each ``schema[i].size`` matches the natural size of the
    ///     declared kind (defends against the Python side fabricating
    ///     an inconsistent schema)
    ///
    /// Throws ``hipdnn_plugin_sdk::HipdnnPluginException``
    /// (INTERNAL_ERROR) on any validation failure; the message names
    /// the offending arg by ``schema[i].name`` so the operator can
    /// trace it back to the kernel signature without reading the
    /// packer.
    static std::vector<std::byte> pack(const std::vector<ArgSchema>& schema,
                                       const std::vector<ArgValue>& values);
};

}  // namespace ck_dsl_provider
