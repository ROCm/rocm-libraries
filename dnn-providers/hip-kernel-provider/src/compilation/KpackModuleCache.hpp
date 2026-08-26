// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "KpackArchive.hpp"
#include "KpackModule.hpp"
#include "ModuleCache.hpp"
#include "device/ScopedDevice.hpp"

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
/// wrong -- but not *who asked*, because the cache is keyed on (archive, tocKey, arch,
/// ordinal) and never sees the descriptor or the symbol. KpackKernelLoader catches this and
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

/// One hipModule_t per (archive path, toc_key, device arch, device ordinal), loaded lazily
/// and shared.
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
/// The device ordinal is both a key component and a binding -- keying alone separates the
/// entries without placing them. load() makes it current across the load (see ScopedDevice)
/// and KpackModule carries it for the unload.
class KpackModuleCache : public ModuleCache<KpackModuleCache,
                                            CachedKpackModule,
                                            const std::string& /*archivePath*/,
                                            const std::string& /*tocKey*/,
                                            const std::string& /*deviceArch*/,
                                            int /*deviceOrdinal*/>
{
public:
    KpackModuleCache() = default;

    // Both members are public because MakeKeyFormatsCorrectly calls makeKey directly;
    // the precedent is SdpaModuleCache.hpp.

    /// The arch component is feature-stripped: flags like ":sramecc+:xnack-" describe the
    /// device, not the code object, and archMatches already gates on the bare name, so
    /// "gfx90a" and "gfx90a:xnack-" must reach one entry rather than load the same blob
    /// twice. load() keeps the decorated string -- it feeds archMatches and names the
    /// device arch in its diagnostics.
    static std::string makeKey(const std::string& archivePath,
                               const std::string& tocKey,
                               const std::string& deviceArch,
                               int deviceOrdinal)
    {
        return archivePath + "::" + tocKey
               + "::" + std::string(hipdnn_plugin_sdk::stripArchFeatures(deviceArch))
               + "::" + std::to_string(deviceOrdinal);
    }

    /// @throws KpackModuleLoadFailure on any stage that fails. Never returns a null
    ///         module: ModuleCache would decline to cache it, but with no message, and
    ///         the caller could not tell which stage gave up.
    static CachedKpackModule load(const std::string& archivePath,
                                  const std::string& tocKey,
                                  const std::string& deviceArch,
                                  int deviceOrdinal)
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

        // Bound before the load, not after: the device current at hipModuleLoadData is
        // the one the module belongs to for the rest of its life. A refused bind fails
        // the load, because an entry cached under one ordinal and resident on another is
        // a wrong answer every later dispatch reuses.
        const device::ScopedDevice binding(deviceOrdinal);
        if(!binding.bound())
        {
            throw KpackModuleLoadFailure(KpackLoadStage::MODULE_LOAD,
                                         "cannot make device " + std::to_string(deviceOrdinal)
                                             + " current to load toc_key '" + tocKey
                                             + "' from kpack archive '" + archivePath + "'");
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

        return std::make_shared<const KpackModule>(module, deviceOrdinal);
    }
};

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
