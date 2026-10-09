// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "KpackArchive.hpp"
#include "KpackModule.hpp"
#include "ModuleCache.hpp"
#include "device/ScopedDevice.hpp"
#include "utilities/Digest.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>

namespace hip_kernel_provider::compilation
{

/// A staged failure escaping the cache's load(). The message already describes what went
/// wrong -- but not *who asked*, because the cache is keyed on (archive, tocKey, arch,
/// ordinal, sha256) and never sees the descriptor or the symbol. KpackKernelLoader catches this and
/// prefixes both, so every message names the descriptor and the symbol without either
/// entering the key.
///
/// stage() is carried alongside the message so a failure can be told apart by machine
/// rather than by matching message text; KpackKernelLoader branches on it when choosing
/// the status it reports.
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

/// An open archive and the architecture list read from it once, shared by every load that
/// names the same path.
struct OpenKpackArchive
{
    KpackArchive archive;
    std::vector<std::string> arches;
};

/// The kpack archives loads have opened, by path, kept open while at least one lease() is
/// alive. Each plugin Container holds a lease, so the archives close when the last hipDNN
/// handle on this provider goes. Retaining saves a kpack_open per module-cache miss (for a
/// zstd archive, a read of the whole compressed blob), and kpack_get_kernel is thread-safe
/// on one handle. With no lease alive, open() returns a handle nothing else keeps.
///
/// A failed open is not kept. A handle whose archive fails to yield an intact entry is
/// evicted, so a damaged or replaced archive is re-read on the next load. Each path has its
/// own open slot, so opens of different archives run in parallel and concurrent first opens
/// of one archive share a single handle.
class SharedKpackArchives
{
public:
    /// Keeps archives open until every lease is released. A load already holding a handle
    /// keeps it.
    static std::shared_ptr<void> lease()
    {
        std::shared_ptr<State> shared = state();
        {
            const std::lock_guard<std::mutex> guard(shared->mutex);
            ++shared->leases;
        }
        // The deleter owns the state, so a lease released during static destruction still
        // finds it.
        State* const token = shared.get();
        return {token, [owner = std::move(shared)](void*) { release(*owner); }};
    }

    /// @throws KpackModuleLoadFailure at OPEN_ARCHIVE or ARCH_LOOKUP.
    static std::shared_ptr<const OpenKpackArchive> open(const std::string& archivePath)
    {
        auto& shared = *state();
        std::shared_ptr<Slot> slot;
        {
            const std::lock_guard<std::mutex> guard(shared.mutex);
            if(shared.leases != 0)
            {
                auto& entry = shared.slots[archivePath];
                if(entry == nullptr)
                {
                    entry = std::make_shared<Slot>();
                }
                slot = entry;
            }
        }
        if(slot == nullptr)
        {
            return openUnretained(archivePath);
        }

        // A slot dropped from the map meanwhile (last lease released, reset) is harmless:
        // whatever it retains goes when the last caller holding the slot returns.
        const std::lock_guard<std::mutex> guard(slot->mutex);
        if(slot->archive == nullptr)
        {
            slot->archive = openUnretained(archivePath);
        }
        return slot->archive;
    }

    /// Stops retaining @p handle for @p archivePath, so the next open() reads the archive
    /// from disk again. A no-op when the path's slot holds a different handle -- one a
    /// concurrent load already reopened -- or none.
    static void evict(const std::string& archivePath,
                      const std::shared_ptr<const OpenKpackArchive>& handle)
    {
        auto& shared = *state();
        std::shared_ptr<Slot> slot;
        {
            const std::lock_guard<std::mutex> guard(shared.mutex);
            const auto found = shared.slots.find(archivePath);
            if(found == shared.slots.end())
            {
                return;
            }
            slot = found->second;
        }

        // The caller still holds @p handle, so dropping the slot's reference never closes
        // the archive under this lock.
        const std::lock_guard<std::mutex> guard(slot->mutex);
        if(slot->archive == handle)
        {
            slot->archive.reset();
        }
    }

    /// Tests only: closes every retained archive, so the next load reopens it from disk.
    /// Leases stay held, so later opens are retained again.
    static void resetForTesting()
    {
        auto& shared = *state();
        Slots closed;
        {
            const std::lock_guard<std::mutex> guard(shared.mutex);
            closed.swap(shared.slots);
        }
    }

    /// Tests only: how many archive opens this process has attempted, failed ones included.
    static std::size_t opensForTesting()
    {
        return state()->opens.load(std::memory_order_relaxed);
    }

private:
    /// One path's retained handle. Its mutex serializes opens of that path only.
    struct Slot
    {
        std::mutex mutex;
        std::shared_ptr<const OpenKpackArchive> archive;
    };

    using Slots = std::unordered_map<std::string, std::shared_ptr<Slot>>;

    /// `mutex` guards `leases` and the map, never an open: a slot's own mutex does that.
    struct State
    {
        std::mutex mutex;
        std::size_t leases = 0;
        Slots slots;
        std::atomic<std::size_t> opens{0}; ///< read by opensForTesting(); needs no lock
    };

    static const std::shared_ptr<State>& state()
    {
        static const std::shared_ptr<State> s_state = std::make_shared<State>();
        return s_state;
    }

    static void release(State& shared)
    {
        // Swapped out under the lock and destroyed after it, so closing archives does not
        // hold up a concurrent lease() or open().
        Slots closed;
        const std::lock_guard<std::mutex> guard(shared.mutex);
        if(--shared.leases == 0)
        {
            closed.swap(shared.slots);
        }
    }

    /// @throws KpackModuleLoadFailure at OPEN_ARCHIVE or ARCH_LOOKUP.
    static std::shared_ptr<const OpenKpackArchive> openUnretained(const std::string& archivePath)
    {
        state()->opens.fetch_add(1, std::memory_order_relaxed);
        auto opened = std::make_shared<OpenKpackArchive>();
        KpackError error;
        if(!opened->archive.open(archivePath, error))
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
        if(!opened->archive.architectures(opened->arches, error))
        {
            throw KpackModuleLoadFailure(error.stage,
                                         "cannot read the architecture list of kpack archive '"
                                             + archivePath + "' (" + error.codeName + ")");
        }
        if(opened->arches.empty())
        {
            throw KpackModuleLoadFailure(KpackLoadStage::ARCH_LOOKUP,
                                         "kpack archive '" + archivePath
                                             + "' declares no architectures; its gfx_arches "
                                               "entry is absent or malformed");
        }
        return opened;
    }
};

/// One hipModule_t per (archive path, toc_key, device arch, device ordinal, declared
/// sha256), loaded lazily and shared.
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
class KpackModuleCache : public ModuleCache<KpackModuleCache,
                                            CachedKpackModule,
                                            const std::string& /*archivePath*/,
                                            const std::string& /*tocKey*/,
                                            const std::string& /*deviceArch*/,
                                            int /*deviceOrdinal*/,
                                            const std::string& /*expectedSha256*/>
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
    ///
    /// `expectedSha256` is in the key so that no cache hit can bypass the digest check: a
    /// hit answers from the key alone, so a digest outside it would hand a second
    /// descriptor declaring a different one the first's module, unverified. It does not
    /// fragment the cache -- two descriptors naming one entry agree on its digest unless
    /// one is wrong, and a wrong one throws in load() before anything is cached.
    static std::string makeKey(const std::string& archivePath,
                               const std::string& tocKey,
                               const std::string& deviceArch,
                               int deviceOrdinal,
                               const std::string& expectedSha256)
    {
        return archivePath + "::" + tocKey
               + "::" + std::string(hipdnn_plugin_sdk::stripArchFeatures(deviceArch))
               + "::" + std::to_string(deviceOrdinal) + "::" + expectedSha256;
    }

    /// @throws KpackModuleLoadFailure on any stage that fails. Never returns a null
    ///         module: ModuleCache would decline to cache it, but with no message, and
    ///         the caller could not tell which stage gave up.
    static CachedKpackModule load(const std::string& archivePath,
                                  const std::string& tocKey,
                                  const std::string& deviceArch,
                                  int deviceOrdinal,
                                  const std::string& expectedSha256)
    {
        const auto opened = SharedKpackArchives::open(archivePath);
        const auto& arches = opened->arches;
        KpackError error;

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
        if(!opened->archive.codeObject(tocKey, *matched, codeObject, error))
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
            // The archive failed to yield an entry its TOC lists, so the next miss re-reads it.
            // A missing entry or arch keeps the handle: both are answered from the TOC read at
            // open, and with discovery memoized a reopen would only repeat the miss. Revisit
            // if discovery ever rescans.
            SharedKpackArchives::evict(archivePath, opened);
            throw KpackModuleLoadFailure(error.stage,
                                         "cannot decompress toc_key '" + tocKey + "' at arch '"
                                             + *matched + "' from kpack archive '" + archivePath
                                             + "' (" + error.codeName + ")");
        }

        // Before hipModuleLoadData, so bytes that fail never reach the driver. The reader
        // cannot catch this itself: a TOC entry pointing at the wrong offset decompresses
        // cleanly and returns another entry's code object rather than an error.
        const std::string actualSha256 = utilities::sha256Hex(codeObject.data(), codeObject.size());
        if(actualSha256 != expectedSha256)
        {
            SharedKpackArchives::evict(archivePath, opened);
            throw KpackModuleLoadFailure(KpackLoadStage::DIGEST_MISMATCH,
                                         "the code object for toc_key '" + tocKey + "' at arch '"
                                             + *matched + "' in kpack archive '" + archivePath
                                             + "' hashes to " + actualSha256
                                             + ", but the descriptor declares " + expectedSha256
                                             + "; the archive and the descriptor disagree about "
                                               "what this entry contains");
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
