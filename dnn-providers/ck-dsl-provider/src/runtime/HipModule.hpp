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
    HipModule& operator=(HipModule&&) = delete;

    /// Launch the kernel with the supplied packed argument buffer,
    /// grid, block, dynamic LDS size, and stream. The packed buffer
    /// is typically produced by ``LaunchAbi::pack``; an empty buffer
    /// is permitted for kernels with no parameters.
    ///
    /// Throws ``HipdnnPluginException`` with the HIP error name if
    /// ``hipModuleLaunchKernel`` returns non-success.
    void launch(const std::vector<std::byte>& packedArgs, const KernelArtifact::GridSpec& grid,
                const KernelArtifact::BlockSpec& block, std::uint32_t ldsBytes, hipStream_t stream);

    /// Convenience overload: pull grid / block / lds from the
    /// artifact's defaults. Used by smoke tests and by future per-op
    /// engines whose artifacts encode the canonical launch shape.
    void launch(const KernelArtifact& artifact, const std::vector<std::byte>& packedArgs,
                hipStream_t stream) {
        launch(packedArgs, artifact.grid, artifact.block, artifact.ldsBytes, stream);
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

   private:
    hipModule_t _module{nullptr};
    hipFunction_t _function{nullptr};
    std::string _kernelName;
};

}  // namespace ck_dsl_provider
