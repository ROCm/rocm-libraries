// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ck_dsl_provider {

/// Schema-driven description of a kernel-argument slot.
///
/// The CK DSL kernels are launched through hipModuleLaunchKernel's
/// ``HIP_LAUNCH_PARAM_BUFFER_*`` "extras" path, which expects a
/// contiguous byte buffer laid out the same way the AMDGPU host-side
/// calling convention would lay the args. Natural-alignment, no
/// padding beyond what each slot's ``align`` demands. This
/// generalises the launcher.cpp's hardcoded per-kind packing.
///
/// The Python compile service emits one ArgSchema per kernel parameter
/// (see ``ck_dsl_provider.compile_service.compile_smoke``); the C++
/// ``LaunchAbi::pack`` walks the schema with caller-supplied
/// ``ArgValue`` instances to produce the buffer handed to HIP.
struct ArgSchema {
    /// Argument kind discriminator. The set is intentionally small for
    /// M1 (smoke + first-conv); extend as new instances surface new
    /// scalar/vector types. Kept as an enum class so the C++ side
    /// rejects unknown tags at translate time.
    enum class Kind : std::uint8_t {
        Pointer,  ///< 64-bit device pointer
        I32,      ///< 32-bit signed integer
        I64,      ///< 64-bit signed integer
        F32,      ///< 32-bit float
        F16,      ///< 16-bit float (passed as 2-byte slot)
    };

    /// Human-readable parameter name (e.g. "A", "C", "N"). Diagnostic
    /// only -- not used by the packer. Kept so logs / errors can name
    /// the offending slot when packing fails.
    std::string name;

    Kind kind{Kind::Pointer};

    /// Size of the slot in bytes. Must match the slot's natural size
    /// for ``kind`` (validated in LaunchAbi). Stored explicitly because
    /// the Python side already knows it and round-tripping it makes the
    /// wire shape self-describing.
    std::uint16_t size{0};

    /// Required alignment of the slot's offset within the arg buffer.
    /// AMDGPU host-side calling convention is natural alignment.
    std::uint16_t align{0};
};

/// Translate the Python "kind" string into the C++ ``ArgSchema::Kind``
/// enum. Returns the parsed Kind on success; throws
/// ``hipdnn_plugin_sdk::HipdnnPluginException`` (INTERNAL_ERROR) on an
/// unknown tag so the bridge surfaces a clear failure at translate
/// time rather than producing a malformed buffer that crashes later
/// inside hipModuleLaunchKernel.
ArgSchema::Kind parseArgKind(const std::string& kindStr);

/// Everything the runtime needs to load a compiled DSL kernel and
/// launch it. Produced by ``CompileServiceBridge::compileSmoke`` for
/// the I-4 smoke path and by ``CompileServiceBridge::compile`` (I-7)
/// for production JIT compiles.
///
/// The artifact owns the HSACO bytes; ``HipModule`` is responsible for
/// passing them to ``hipModuleLoadData`` and may release the bytes
/// after the load returns (the HIP runtime copies what it needs --
/// confirmed by launcher.cpp:571-575).
///
/// Grid / block are stored as plain tuples here; the per-op
/// ``SpecBuilder`` derives them from the problem shape at JIT time
/// (the artifact does not need to know how they were computed). For
/// kernels that take their grid from a single launch the artifact's
/// grid is the canonical value; kernels that vary their grid per
/// invocation will set the artifact's grid to a "default" and the
/// caller will override via ``HipModule::launch`` -- see P-1 for the
/// design memo.
struct KernelArtifact {
    /// Three-component grid dimensions. Mirrors the
    /// ``gridDim{X,Y,Z}`` args of ``hipModuleLaunchKernel``. Stored as
    /// uint32_t to match HIP's API; the Python side emits ints which
    /// fit cleanly.
    struct GridSpec {
        std::uint32_t x{1};
        std::uint32_t y{1};
        std::uint32_t z{1};
    };

    /// Three-component block dimensions. Mirrors
    /// ``blockDim{X,Y,Z}``. The first dim is the kernel's wave-aligned
    /// thread count (a multiple of 64 on gfx9); y and z are typically
    /// 1 for the kernels we emit today.
    struct BlockSpec {
        std::uint32_t x{1};
        std::uint32_t y{1};
        std::uint32_t z{1};
    };

    /// HSA code object bytes. Loaded once into a ``HipModule`` and
    /// thereafter not needed at launch time, but kept so a future
    /// disk-cache layer (M3) can persist them without re-compiling.
    std::vector<std::byte> hsaco;

    /// Mangled kernel symbol passed to ``hipModuleGetFunction``. Comes
    /// straight from ``ck_dsl.helpers.compile.KernelArtifact.kernel_name``.
    std::string kernelName;

    /// Free-form kind tag for diagnostics ("elementwise_copy_smoke",
    /// "conv_implicit_gemm", ...). Not consumed by the launcher.
    std::string kind;

    GridSpec grid{};
    BlockSpec block{};

    /// Dynamic shared-memory bytes (``sharedMemBytes`` arg). Zero for
    /// kernels that don't allocate dynamic LDS. Closing
    /// launcher.cpp's gap (always 0 there) was an explicit P-1
    /// recommendation.
    std::uint32_t ldsBytes{0};

    /// Per-arg layout schema. Order matches the kernel's positional
    /// parameter list. ``LaunchAbi::pack`` walks this in order.
    std::vector<ArgSchema> argSchema;

    /// Comgr ISA triple the artifact was built for. Stored for
    /// diagnostics / cache-key composition. The DSL targets
    /// gfx942/gfx950/gfx1151, so this records whichever arch was
    /// requested (e.g. ``"amdgcn-amd-amdhsa--gfx942"``).
    std::string isa;
};

}  // namespace ck_dsl_provider
