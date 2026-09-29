// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace TensileLite::hip
{
    class SolutionAdapter;
}

namespace hipblaslt_jit
{
    // Pre-built Tensile indices are dense from 0 and rocRoller indices are
    // negative, so JIT solution library indices use the rest of the int32 range.
    constexpr int32_t jitIndexBase = 0x40000000;
    constexpr bool    isJitIndex(int32_t index) noexcept
    {
        return index >= jitIndexBase;
    }

    // Libraries live under "v<jitLibrarySchema>"; other versions are never read.
    constexpr int jitLibrarySchema = 1;

    using GemmMaster = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

    // The variables of a NAME=VALUE environment that can change comgr output:
    // HIP_PATH, LLVM_PATH and every AMD_COMGR_* variable except the logging,
    // cache and temporary-file controls.
    std::map<std::string, std::string> compilerEnvironment(const char* const* environment);

    // Everything that can change a generated code object. Each key has its own
    // directory, and other keys' directories are never read or deleted.
    struct CacheKey
    {
        std::string                        targetId, isa, libraryArch;
        int                                wavefrontSize = 0;
        std::string                        backendId, backendVersion;
        int                                codeObjectVersion = jitCodeObjectVersion;
        std::string                        comgr; // code_object::comgrIdentity()
        std::string                        rocmPath; // code_object::rocmPath()
        std::map<std::string, std::string> environment; // compilerEnvironment()

        // Captures the comgr identity, the ROCm path and the environment once per
        // process, because comgr latches some of them when it is first used.
        static CacheKey
                    make(const DeviceTarget& target, const BackendInfo& backend, int codeObjectVersion);
        std::string canonicalJson() const; // the content of cache-key.json
        std::string directoryName() const; // "<isa>-<FNV-1a of canonicalJson()>"
    };

    // The problem's ProblemType as canonical JSON. Throws like problemTypeFields.
    std::string problemTypeKey(const TensileLite::ContractionProblemGemm& problem);
    // problem.size(i) for every index; a published entry matches only these sizes.
    std::vector<size_t> problemSizes(const TensileLite::ContractionProblemGemm& problem);
    // "TensileLibrary_JIT_<ProblemType hash>_<kernel and sizes hash>[_<collision>]".
    // A function of the content, so concurrent publishers of one entry choose the
    // same name.
    std::string entryPrefix(const std::string&         problemTypeKey,
                            const std::string&         kernelName,
                            const std::vector<size_t>& sizes,
                            unsigned                   collision = 0);

    enum class PublishStep
    {
        Staged,
        Locked,
        Allocated,
        CodeObjects,
        Entries,
        Mapping,
        Master,
        Unlocked,
    };
#ifdef HIPBLASLT_JIT_LIBRARY_TESTING
    // Called after each publication step; crash tests exit the process there.
    extern void (*publishStepHook)(PublishStep);
#endif

    // A persistent library of generated solutions. Each cache key directory is a
    // TensileLite lazy-loading library that the stock loader reads: a master
    // with one row per entry, matching only the entry's ProblemType and exact
    // sizes, an index mapping, and one entry file and code object per solution.
    // Publishing holds the root's file lock and replaces files atomically, in an
    // order that never lets a reader see a reference to a missing file, so
    // readers take no lock. Thread-safe.
    class JitLibrary
    {
    public:
        struct View
        {
            std::shared_ptr<GemmMaster>        master; // nullptr when unavailable
            TensileLite::hip::SolutionAdapter* adapter = nullptr;
        };

        // Sets key to device's cache key with empty backend fields.
        using DeviceKey = std::function<Status(int device, CacheKey& key)>;

        // Touches no file until first use. Without deviceKey, resolve() finds
        // only indices in directories this instance has looked up or published to.
        explicit JitLibrary(std::filesystem::path root, DeviceKey deviceKey = nullptr);
        JitLibrary(const JitLibrary&)            = delete;
        JitLibrary& operator=(const JitLibrary&) = delete;
        ~JitLibrary();

        // HIPBLASLT_JIT_LIBRARY_PATH, or /tmp/hipblaslt-jit-<uid>
        // (%TEMP%\hipblaslt-jit-<user> on Windows).
        static std::filesystem::path defaultRoot();
        // Rooted at defaultRoot(). Never destroyed, so views and adapters stay
        // valid until exit.
        static JitLibrary& process();

        const std::filesystem::path& root() const noexcept;
        std::filesystem::path        directory(const CacheKey& key) const;

        // Every later call fails with reason.
        void disable(const std::string& reason);

        // Up to count solutions published for exactly problem's ProblemType and
        // sizes, in index order (the order they were first published) and without
        // excluded kernels. Reloads the library first when another process has
        // published.
        Status lookup(const CacheKey&                           key,
                      int                                       device,
                      const TensileLite::ContractionProblemGemm& problem,
                      const TensileLite::Hardware&              hardware,
                      size_t                                    count,
                      const std::vector<std::string>&           excludeKernels,
                      std::vector<int32_t>&                     indices);

        // Returns one index per solution, in order. A solution already published
        // for the same kernel and sizes keeps its index.
        Status publish(const CacheKey&                           key,
                       int                                       device,
                       const TensileLite::ContractionProblemGemm& problem,
                       const std::vector<BuiltSolution>&         solutions,
                       std::vector<int32_t>&                     indices);

        // The library that maps index on device, and the adapter that loads its
        // code objects from the key directory. An index is looked for in every
        // directory whose key matches device's for any backend, because the
        // caller already chose the solution; each directory is reloaded once
        // when the index is newer than this process has seen. Empty, with why
        // set, otherwise.
        View resolve(int device, int32_t index, Status& why);
        std::shared_ptr<TensileLite::ContractionSolution> solutionByIndex(
            int device, const TensileLite::Hardware& hardware, int32_t index, Status& why);

    private:
        struct Directory;

        std::filesystem::path schemaDirectory() const;
        // Callers hold m_mutex.
        Status     checkRoot();
        Directory* open(const std::string& name, const std::string& isa, const std::string& key, Status& why);
        Status     attach(const CacheKey& key, int device, Directory*& attached);
        void       discover(int device);

        std::filesystem::path                             m_root;
        DeviceKey                                         m_deviceKey;
        std::mutex                                        m_mutex;
        std::string                                       m_disabled;
        bool                                              m_checked = false;
        std::map<std::string, std::unique_ptr<Directory>> m_directories;
        std::map<int, std::vector<Directory*>>            m_routes;
    };

    // Stores GEMM solutions in library under the cache key of backend.
    std::shared_ptr<const SolutionStore>
        makeLibraryStore(JitLibrary& library, const BackendInfo& backend, int codeObjectVersion);
}
