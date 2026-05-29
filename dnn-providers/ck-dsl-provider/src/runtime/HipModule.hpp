// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#include "KernelArtifact.hpp"
#include "LaunchAbi.hpp"

namespace ck_dsl_provider {

/// RAII wrapper around a loaded ``hipModule_t`` plus the resolved
/// ``hipFunction_t`` for one named kernel.
///
/// One ``HipModule`` per ``KernelArtifact``. Constructing it calls
/// ``hipModuleLoadData`` followed by ``hipModuleGetFunction``;
/// destruction calls ``hipModuleUnload``. Both ctor failure paths
/// throw ``hipdnn_plugin_sdk::HipdnnPluginException`` with the
/// underlying ``hipError_t`` in the message so a failed load surfaces
/// the HIP error name (via ``hipGetErrorString``) rather than a bare
/// status code.
///
/// Thread safety: the underlying HIP runtime calls are documented as
/// thread-safe; the wrapper holds no mutable state beyond the two HIP
/// handles, so ``launch`` on a single HipModule from multiple threads
/// is as safe as HIP itself.
///
/// Lifetime: the module owns its ``hipModule_t``. The associated
/// ``hipFunction_t`` is a non-owning view into the module and becomes
/// dangling as soon as ``hipModuleUnload`` runs. Callers must not
/// outlive the ``HipModule`` they hold a function pointer from --
/// which is why the launch interface lives on the module itself, not
/// on a separately-handed-out HipFunction type.
class HipModule {
   public:
    /// Load ``artifact.hsaco`` into a HIP module and resolve
    /// ``artifact.kernelName``. The artifact reference is captured by
    /// the caller; the ctor only reads ``hsaco`` / ``kernelName`` and
    /// stores ``kernelName`` for diagnostic logging.
    explicit HipModule(const KernelArtifact& artifact);

    ~HipModule() noexcept;

    HipModule(const HipModule&) = delete;
    HipModule& operator=(const HipModule&) = delete;
    HipModule(HipModule&& other) noexcept;
    /// Move-assignment is deliberately not provided. Implementing it
    /// would require unloading ``this``'s current module before
    /// overwriting it -- the same sequence the destructor runs, but
    /// in a context where we don't want the failure-swallowing
    /// noexcept behaviour. Since the cache stores ``shared_ptr<HipModule>``
    /// and the plan layer holds the module by shared pointer too, there
    /// is no concrete call site that needs move-assignment today. If a
    /// future caller wants to hold ``HipModule`` by value inside a
    /// vector for re-pointing, add the operator here with explicit
    /// unload-of-the-overwritten-handle semantics rather than
    /// re-enabling it without that bookkeeping.
    HipModule& operator=(HipModule&&) = delete;

    /// Launch the kernel with the supplied packed argument buffer
    /// (raw pointer + size variant). The buffer is typically produced
    /// either by ``LaunchAbi::pack`` (for tests / generic launch sites)
    /// or by an op-specific per-plan pre-packed template (for hot-path
    /// launches that want to avoid the per-call vector allocation).
    /// An empty buffer (``args == nullptr && argsSize == 0``) is
    /// permitted for kernels with no parameters.
    ///
    /// Throws ``HipdnnPluginException`` with the HIP error name if
    /// ``hipModuleLaunchKernel`` returns non-success.
    void launch(const std::byte* args, std::size_t argsSize, const KernelArtifact::GridSpec& grid,
                const KernelArtifact::BlockSpec& block, std::uint32_t ldsBytes, hipStream_t stream);

    /// Vector overload preserved for callers (mostly tests) that
    /// already hold a ``std::vector`` from ``LaunchAbi::pack``.
    void launch(const std::vector<std::byte>& packedArgs, const KernelArtifact::GridSpec& grid,
                const KernelArtifact::BlockSpec& block, std::uint32_t ldsBytes,
                hipStream_t stream) {
        launch(packedArgs.data(), packedArgs.size(), grid, block, ldsBytes, stream);
    }

    /// Convenience overload: pull grid / block / lds from the
    /// artifact's defaults. Used by smoke tests and by future per-op
    /// engines whose artifacts encode the canonical launch shape.
    void launch(const KernelArtifact& artifact, const std::vector<std::byte>& packedArgs,
                hipStream_t stream) {
        launch(packedArgs.data(), packedArgs.size(), artifact.grid, artifact.block,
               artifact.ldsBytes, stream);
    }

    /// Test-only accessors. Production code launches via ``launch``;
    /// exposing the raw handles lets the unit suite drive the HIP
    /// runtime directly when verifying the load path.
    hipModule_t moduleHandle() const noexcept {
        return _module;
    }
    hipFunction_t functionHandle() const noexcept {
        return _function;
    }
    const std::string& kernelName() const noexcept {
        return _kernelName;
    }

    /// Launch metadata captured from the artifact at ctor time. The
    /// plan layer reads these on every ``execute`` so callers don't
    /// have to thread the original ``KernelArtifact`` alongside the
    /// module. The HSACO bytes themselves are not retained -- the HIP
    /// runtime copied what it needs during ``hipModuleLoadData``.
    const KernelArtifact::GridSpec& grid() const noexcept {
        return _grid;
    }
    const KernelArtifact::BlockSpec& block() const noexcept {
        return _block;
    }
    std::uint32_t ldsBytes() const noexcept {
        return _ldsBytes;
    }
    const std::vector<ArgSchema>& argSchema() const noexcept {
        return _argSchema;
    }
    const std::string& kind() const noexcept {
        return _kind;
    }

   private:
    hipModule_t _module{nullptr};
    hipFunction_t _function{nullptr};
    std::string _kernelName;
    std::string _kind;
    KernelArtifact::GridSpec _grid{};
    KernelArtifact::BlockSpec _block{};
    std::uint32_t _ldsBytes{0};
    std::vector<ArgSchema> _argSchema;
};

}  // namespace ck_dsl_provider
