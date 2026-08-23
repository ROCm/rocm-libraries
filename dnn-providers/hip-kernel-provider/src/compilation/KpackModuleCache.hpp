// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "KpackArchive.hpp"
#include "KpackModule.hpp"
#include "ModuleCache.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>

namespace hip_kernel_provider::compilation
{

/// A staged failure escaping the cache's load(). The message already describes what went
/// wrong -- but not *who asked*, because the cache is keyed on (archive, tocKey, arch) and
/// deliberately never sees the descriptor or the symbol. KpackKernelLoader catches this and
/// prefixes both, so every message names the descriptor and the symbol without either
/// entering the key.
///
/// stage() is carried alongside the message so a failure can be told apart by machine
/// rather than by matching message text; KpackKernelLoader itself rewraps on the message.
class KpackModuleLoadFailure : public std::runtime_error
{
public:
    KpackModuleLoadFailure(KpackLoadStage stage, const std::string& message)
        : std::runtime_error(message)
        , _stage(stage)
    {
    }

    KpackLoadStage stage() const
    {
        return _stage;
    }

private:
    KpackLoadStage _stage;
};

using CachedKpackModule = std::shared_ptr<const KpackModule>;

/// One hipModule_t per (archive path, toc_key, device arch), loaded lazily and shared.
///
/// The key deliberately excludes the kernel symbol. `toc_key` is content-addressed on
/// (source, build) only, so two kernels that differ solely by entry point name the same
/// blob and must share one module. `symbol` applies one layer up, at
/// hipModuleGetFunction in KpackProgram. Do not add it here even though
/// KpackKernelLoader::load() receives one; it takes that parameter purely so its error
/// messages can name it.
///
/// Why not rocm-kpack's own cache: kpack_cache_* caches the decompressed code-object
/// *blob*, not the loaded hipModule_t, so it would still leave a hipModuleLoadData on
/// every dispatch. Building on compilation::ModuleCache also matches SdpaModuleCache.
///
/// Not keyed by device ordinal, which is safe only because every device in a process
/// that shares an arch also shares this provider's single HIP context. A hipModule_t
/// belongs to the device current when it was loaded, so keying by arch alone would be
/// wrong under a per-device context; arch is in the key, so the different-arch case is
/// correct either way. SdpaModuleCache has the identical property.
class KpackModuleCache : public ModuleCache<KpackModuleCache,
                                            CachedKpackModule,
                                            const std::string& /*archivePath*/,
                                            const std::string& /*tocKey*/,
                                            const std::string& /*deviceArch*/>
{
public:
    KpackModuleCache() = default;

    // Both members are public because MakeKeyFormatsCorrectly calls makeKey directly;
    // the precedent is SdpaModuleCache.hpp.

    static std::string makeKey(const std::string& archivePath,
                               const std::string& tocKey,
                               const std::string& deviceArch)
    {
        return archivePath + "::" + tocKey + "::" + deviceArch;
    }

    /// @throws KpackModuleLoadFailure on any stage that fails. Never returns a null
    ///         module: ModuleCache would decline to cache it, but with no message, and
    ///         the caller could not tell which stage gave up.
    static CachedKpackModule load(const std::string& archivePath,
                                  const std::string& tocKey,
                                  const std::string& deviceArch)
    {
        KpackArchive archive;
        KpackError error;

        if(!archive.open(archivePath, error))
        {
            if(error.archiveAbsent)
            {
                throw KpackModuleLoadFailure(error.stage,
                                             "kpack archive '" + archivePath + "' does not exist ("
                                                 + error.codeName + ")");
            }
            throw KpackModuleLoadFailure(error.stage,
                                         "kpack archive '" + archivePath + "' could not be read ("
                                             + error.codeName + ")");
        }

        std::vector<std::string> arches;
        if(!archive.architectures(arches, error))
        {
            throw KpackModuleLoadFailure(error.stage,
                                         "cannot read the architecture list of kpack archive '"
                                             + archivePath + "' (" + error.codeName + ")");
        }

        // Deliberate pre-check rather than letting kpack_get_kernel fail: a bare
        // KERNEL_NOT_FOUND cannot distinguish "wrong GPU" from "wrong toc_key", and
        // those two send a reader to entirely different places.
        const std::string* matched = nullptr;
        for(const auto& candidate : arches)
        {
            if(hipdnn_plugin_sdk::archMatches(
                   deviceArch, candidate, hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
            {
                matched = &candidate;
                break;
            }
        }
        if(matched == nullptr)
        {
            std::string available;
            for(const auto& candidate : arches)
            {
                available += (available.empty() ? "" : ", ") + candidate;
            }
            throw KpackModuleLoadFailure(
                KpackLoadStage::ARCH_LOOKUP,
                "kpack archive '" + archivePath + "' holds no binary for device arch '" + deviceArch
                    + "'; the archive provides: " + (available.empty() ? "(none)" : available));
        }

        KpackCodeObject codeObject;
        if(!archive.codeObject(tocKey, *matched, codeObject, error))
        {
            if(error.stage == KpackLoadStage::ENTRY_LOOKUP)
            {
                throw KpackModuleLoadFailure(error.stage,
                                             "kpack archive '" + archivePath
                                                 + "' has no entry for toc_key '" + tocKey
                                                 + "' at arch '" + *matched + "' (" + error.codeName
                                                 + "); this usually means the packer and the "
                                                   "descriptor disagree");
            }
            throw KpackModuleLoadFailure(error.stage,
                                         "cannot decompress toc_key '" + tocKey + "' at arch '"
                                             + *matched + "' from kpack archive '" + archivePath
                                             + "' (" + error.codeName + ")");
        }

        hipModule_t module = nullptr;
        const hipError_t status = hipModuleLoadData(&module, codeObject.data());
        if(status != hipSuccess)
        {
            throw KpackModuleLoadFailure(KpackLoadStage::MODULE_LOAD,
                                         "hipModuleLoadData rejected the code object for toc_key '"
                                             + tocKey + "' at arch '" + *matched
                                             + "' from kpack archive '" + archivePath
                                             + "': " + hipGetErrorString(status));
        }

        return std::make_shared<const KpackModule>(module);
    }
};

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
