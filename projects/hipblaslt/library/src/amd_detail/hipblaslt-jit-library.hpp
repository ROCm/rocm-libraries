// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/SolutionLibrary.hpp>
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
    // Pre-built TensileLite indices are dense from 0 and other encoded solution indices
    // are negative, so JIT solution library indices use the rest of the int32 range.
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
        // Those toolchain fields are recorded in cache-key.msgpack and are not
        // part of the directory name: a later ROCm still reads the library.
        static CacheKey
                    make(const DeviceTarget& target, const BackendInfo& backend, int codeObjectVersion);
        std::string directoryName() const; // filesystem-safe literal, not a hash
    };

    // The problem's ProblemType as canonical JSON. Throws like problemTypeFields.
    std::string problemTypeKey(const TensileLite::ContractionProblemGemm& problem);
    // problem.size(i) for every index; a published entry matches only these sizes.
    std::vector<size_t> problemSizes(const TensileLite::ContractionProblemGemm& problem);
    // One lazy slice: operation identifier, datatypes, the TensileLite scale
    // suffix (scalar or vector UseScaleAB, MX block and type) and the epilogue
    // flags that change the kernel family. Sizes stay inside the slice.
    std::string sliceKey(const TensileLite::ContractionProblemGemm& problem);
    // Filesystem-safe literal of the kernel, the problem type and the sizes.
    // The same identity always names the same entry, so a second publish reuses
    // that entry's index.
    std::string entryPrefix(const std::string&         problemTypeKey,
                            const std::string&         kernelName,
                            const std::vector<size_t>& sizes);

    // A persistent library of generated solutions. Each directory holds one
    // slice per kernel family. A query opens that slice only: a TensileLite
    // lazy master, its index mapping, and one entry and code object per
    // solution, matching the full ProblemType and the exact sizes. Shared HIP
    // helpers live beside the slices, one code object per backend id and name.
    // Library files are MessagePack, including cache-key.msgpack. A solution
    // whose code-object version or kernarg version this build cannot run is
    // skipped; the rest of the library still loads.
    // Publishing holds the root's file lock and replaces files atomically, in an
    // order that never lets a reader see a reference to a missing file, so
    // readers take no lock. Thread-safe.
    //
    // Consulted the same way as the Equality and Origami libraries.
    // findTopSolutions returns published matches and, on a miss, generates and
    // publishes when beginQuery allowed it. Grouped-GEMM queries return nothing.
    class JitLibrary : public TensileLite::SolutionLibrary<TensileLite::ContractionProblemGemm>
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

        // /tmp/hipblaslt-jit-<uid> (%TEMP%\hipblaslt-jit-<user> on Windows).
        // HIPBLASLT_JIT_LIBRARY_PATH is applied by process(), not here.
        static std::filesystem::path defaultRoot();
        // Rooted at HIPBLASLT_JIT_LIBRARY_PATH, or defaultRoot(). Never
        // destroyed, so views and adapters stay valid until exit.
        static JitLibrary& process();

        const std::filesystem::path& root() const noexcept;
        std::filesystem::path        directory(const CacheKey& key) const;

        // Publishes the shortfall. indices receives new library indices. failure
        // is set when nothing new was published.
        using GenerateMissing = std::function<void(size_t                          count,
                                                   const std::vector<std::string>& excludeKernels,
                                                   std::vector<int32_t>&           indices,
                                                   Status&                         failure)>;

        // What findTopSolutions may do. Set on the calling thread before the
        // library call. A capturing stream sets allowGeneration false so a hit
        // can return and a miss does not build. endQuery clears it.
        void beginQuery(bool                     allowGeneration,
                        const DeviceTarget&      target,
                        const BackendInfo&       backend,
                        size_t                   workspace,
                        std::vector<std::string> excludeKernels,
                        GenerateMissing          generate);
        void endQuery();
        // Failure from the query in progress, or success when there is none.
        Status queryStatus() const;

        // Every later call fails with reason.
        void disable(const std::string& reason);

        std::shared_ptr<TensileLite::ContractionSolution>
            getSolutionByIndex(TensileLite::ContractionProblemGemm const& problem,
                               TensileLite::Hardware const&               hardware,
                               const int                                  index) const override;
        std::shared_ptr<TensileLite::ContractionSolution>
            findBestSolution(TensileLite::ContractionProblemGemm const& problem,
                             TensileLite::Hardware const&               hardware,
                             double*                                    fitness = nullptr) const override;
        TensileLite::SolutionSet<TensileLite::ContractionSolution>
            findAllSolutions(TensileLite::ContractionProblemGemm const& problem,
                             TensileLite::Hardware const&               hardware,
                             TensileLite::SolutionLibrarySearchType     searchType
                             = TensileLite::SolutionLibrarySearchType::DEFAULT) const override;
        TensileLite::SolutionSet<TensileLite::ContractionSolution> findAllSolutionsGroupedGemm(
            std::vector<TensileLite::ContractionProblemGemm> const& problems,
            TensileLite::Hardware const&                            hardware,
            TensileLite::SolutionLibrarySearchType                  searchType
            = TensileLite::SolutionLibrarySearchType::DEFAULT) const override;
        std::string type() const override;
        std::string description() const override;
        TensileLite::SolutionVector<TensileLite::ContractionSolution>
            findTopSolutions(TensileLite::ContractionProblemGemm const& problem,
                             TensileLite::Hardware const&               hardware,
                             int                                         numSolutions) const override;
        TensileLite::SolutionVector<TensileLite::ContractionSolution> findTopSolutionsGroupedGemm(
            std::vector<TensileLite::ContractionProblemGemm> const& problems,
            TensileLite::Hardware const&                            hardware,
            int                                                     numSolutions) const override;

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

        // Returns one index per supported solution, in order. A solution already
        // published for the same kernel and sizes keeps its index.
        Status publish(const CacheKey&                           key,
                       int                                       device,
                       const TensileLite::ContractionProblemGemm& problem,
                       const SupportedSolutions&                 solutions,
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
        Directory* open(const std::string& name, const CacheKey& key, Status& why);
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
