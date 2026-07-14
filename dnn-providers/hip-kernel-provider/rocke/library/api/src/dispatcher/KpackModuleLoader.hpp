// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <rocm_kpack/kpack.h>

#include <string>

namespace rocke_client::dispatcher
{

/// Result of a kpack → hipModule load attempt.
///
/// On success, both module and fn are valid handles owned by the caller, who
/// must call hipModuleUnload(module) when done. On failure, exactly one of
/// kpackError or hipError is non-zero; the other reflects the API stage that
/// was not reached.
struct KpackLoadResult
{
    hipModule_t module{nullptr};
    hipFunction_t fn{nullptr};
    kpack_error_t kpackError{KPACK_SUCCESS};
    hipError_t hipError{hipSuccess};

    /// Return true if both handles are valid (load succeeded end-to-end).
    bool ok() const noexcept
    {
        return module != nullptr && fn != nullptr;
    }
};

/// Open a .kpack archive, extract the HSACO bytes for (tocKey, arch), load
/// the image into a hipModule, and look up kernelName in that module.
///
/// The archive is closed and the kpack kernel buffer freed before returning,
/// regardless of outcome. On success the caller owns the returned
/// module/function and must call hipModuleUnload(result.module) when done.
///
/// Calling this function is the direct reference to kpack_* symbols that
/// guarantees DT_NEEDED librocm_kpack.so.0 survives --as-needed linking.
///
/// @param kpackPath   Filesystem path to the .kpack archive.
/// @param tocKey      Table-of-contents key used when the archive was packed
///                    (binary_name in the kpack C API). For rocKE bundles
///                    this is "rocke/<op>/<family>/<name>".
/// @param arch        GFX architecture string (e.g. "gfx942").
/// @param kernelName  HIP kernel function name to resolve in the loaded module.
/// @return            KpackLoadResult with valid handles on success, or error
///                    codes identifying the failure stage.
KpackLoadResult loadKernelFromKpack(const std::string& kpackPath,
                                    const std::string& tocKey,
                                    const std::string& arch,
                                    const std::string& kernelName);

} // namespace rocke_client::dispatcher
