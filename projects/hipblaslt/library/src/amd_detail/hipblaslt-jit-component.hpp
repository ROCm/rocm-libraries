// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace TensileLite
{
    class Hardware;
}

// Compiled-in stages of JIT solution generation. This is not a plugin ABI.
namespace hipblaslt_jit
{
    // True when kernel is one of names. Shared by the replay backend and the
    // heuristic exclude list.
    inline bool excludedKernel(const std::vector<std::string>& names, const std::string& kernel)
    {
        return std::find(names.begin(), names.end(), kernel) != names.end();
    }

    enum class Stage
    {
        Configure,
        Generate,
        Build,
        Support,
        Load,
        Lookup,
        Publish,
    };

    struct Status
    {
        enum class Code
        {
            Success,
            NotSupported, // outside the component's domain; not a defect
            TargetMismatch, // configured for another device target
            Failed,
        };
        Code        code  = Code::Success;
        Stage       stage = Stage::Configure;
        std::string message; // one line
        std::string logPath; // kept for diagnosis, or empty
        bool        ok() const noexcept
        {
            return code == Code::Success;
        }
    };

    // One file. Assembly and Hip are sources. CodeObject is a linked executable
    // or a helper code object stored beside a solution. Relocatables stay inside
    // the comgr builder.
    struct Artifact
    {
        std::string name;
        enum class Kind
        {
            Assembly,
            Hip,
            CodeObject,
        };
        Kind                 kind = Kind::Assembly;
        std::vector<uint8_t> bytes;
    };

    // The sources that become one solution in JitLibrary. Not a TensileLite
    // library entry. helpers are HIP sources compiled once per backend id and
    // helper name. includes are headers. hipFlags are appended after the HIP
    // defaults, so -std=c++20 overrides -std=c++17.
    struct SourceCodeBundle
    {
        Artifact                 mainKernel; // Assembly or Hip
        std::vector<Artifact>    helpers;
        std::vector<Artifact>    includes;
        std::vector<std::string> hipFlags;
    };

    constexpr int jitCodeObjectVersion = 4;

    // Published HIP helper code objects. JitLibrary owns the cache and keys it
    // by backend id and helper name.
    class HelperCache
    {
    public:
        virtual ~HelperCache() = default;
        // The published code object, or an empty Artifact when name is absent.
        virtual Artifact find(const std::string& helperName) const = 0;
        // Stores a code object. A name that is already published is left unchanged.
        virtual Status publish(const Artifact& helper)             = 0;
        // Asks the cache before compiling. A hit returns the published Artifact.
        // A miss calls compile once, publishes that code object, and returns it.
        virtual Status acquire(const std::string&                       helperName,
                               const std::function<Status(Artifact&)>& compile,
                               Artifact&                                out)
            = 0;
    };

    struct BuildRequest
    {
        std::string              targetId; // gcnArchName, e.g. "gfx950:sramecc+:xnack-"
        int                      codeObjectVersion = jitCodeObjectVersion;
        std::filesystem::path    scratch; // receives comgr.log after a failure, or empty
        std::vector<std::string> kernelNames; // main kernels the code object must define
        // When set, ComgrBuilder asks this cache before compiling each helper.
        // Null compiles every helper and does not publish it.
        std::shared_ptr<HelperCache> helpers;
    };

    // The linked code object and any helper code objects. The TensileLite
    // solution record is what JitLibrary writes into the library; it is not a
    // library object and not part of the generated source.
    struct CompiledCode
    {
        Artifact              codeObject; // Kind::CodeObject; name is the main kernel
        std::vector<Artifact> helpers; // Kind::CodeObject, named by the helper source
        std::vector<uint8_t>  record;
    };

    class CodeObjectBuilder
    {
    public:
        virtual ~CodeObjectBuilder() = default;
        virtual Status
            build(const SourceCodeBundle&, const BuildRequest&, CompiledCode&) const = 0;
    };

    // Compiles the SourceCodeBundle with comgr: each helper code object, then
    // the main kernel. The CompiledCode is ready to add to JitLibrary. A helper
    // cache on the request is asked before a helper is compiled.
    std::shared_ptr<const CodeObjectBuilder> makeComgrBuilder();

    using GemmRequest = hipblaslt_ext::experimental::jit::detail::GemmRequest;
    using KernelBundle     = hipblaslt_ext::experimental::jit::detail::KernelBundle;

    // "configure", "generate", ... as reports name stages.
    const char* toString(Stage stage) noexcept;

    struct DeviceTarget
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        std::string     targetId; // gcnArchName, e.g. "gfx950:sramecc+:xnack-"
        std::string     isa; // targetId up to the first ':'
        std::string     libraryArch; // GEMM library subtree, e.g. "gfx1250v0"
        int             wavefrontSize = 0;
        int             cuCount       = 0;
        std::shared_ptr<TensileLite::Hardware> hardware;

        static Status make(int device, DeviceTarget& target);
    };

    struct GenerationRequest
    {
        const GemmRequest&  request;
        const DeviceTarget&      target;
        size_t                   count      = 1;
        size_t                   workspaceLimit = 0;
        std::vector<std::string> excludeKernels; // kernels the caller already has
        std::filesystem::path    scratch; // private directory owned by this call
        int codeObjectVersion = jitCodeObjectVersion; // for generators and the builder
    };

    struct BackendInfo
    {
        std::string id;
        std::string name; // reported as Diagnostics::backend
        uint32_t    version = 0; // increment when the generated solutions can change
    };

    class Backend
    {
    public:
        virtual ~Backend()                               = default;
        virtual const BackendInfo& info() const noexcept = 0;
        // Returns up to request.count source bundles for the request, best
        // first, and loads nothing. NotSupported means the request is outside
        // the backend's domain.
        virtual Status generate(const GenerationRequest&, std::vector<SourceCodeBundle>&) const
            = 0;
        // Main kernels source must define. Empty when the code object may
        // define whatever kernels it contains.
        virtual std::vector<std::string> kernelNames(const SourceCodeBundle&) const
        {
            return {};
        }
        // The TensileLite solution record JitLibrary stores for source, or
        // empty. Not a library object. A predicate check loads it locally with
        // loadGemmLibrary inside the backend function; it is not a bundle field.
        virtual std::vector<uint8_t> solutionRecord(const SourceCodeBundle&) const
        {
            return {};
        }
    };

    class SolutionLoader
    {
    public:
        virtual ~SolutionLoader() = default;
        // Evaluates the predicates and workspace of the entry's solutions for the
        // request and returns the local indices of those that support it, best
        // first. Loads no code.
        virtual Status support(const CompiledCode&,
                               const GemmRequest&,
                               const DeviceTarget&,
                               size_t            workspaceLimit,
                               std::vector<int>& indices) const
            = 0;
        // Loads the code objects once and returns a process-local executable
        // bundle for each of the solutions at indices.
        virtual Status load(const CompiledCode&,
                            const GemmRequest&,
                            const DeviceTarget&,
                            size_t                                            workspaceLimit,
                            const std::vector<int>&                           indices,
                            std::vector<std::shared_ptr<const KernelBundle>>& bundles) const
            = 0;
    };

    // Built code objects, each with the local indices of the solutions in its
    // record that support a request, best first.
    using SupportedSolutions = std::vector<std::pair<CompiledCode, std::vector<int>>>;

    class SolutionStore
    {
    public:
        virtual ~SolutionStore() = default;
        // Up to count indices of stored solutions for exactly this request that
        // need at most maxWorkspaceBytes, best first, without excluded kernels.
        virtual Status lookup(const GemmRequest&,
                              const DeviceTarget&,
                              size_t                          count,
                              size_t                          maxWorkspaceBytes,
                              const std::vector<std::string>& excludeKernels,
                              std::vector<int32_t>&           indices) const
            = 0;
        // Returns one library index per supported solution, in order.
        virtual Status publish(const GemmRequest&,
                               const DeviceTarget&,
                               const SupportedSolutions&,
                               std::vector<int32_t>& indices) const
            = 0;
    };

    class Jit
    {
    public:
        struct Components
        {
            std::shared_ptr<const Backend>           backend;
            std::shared_ptr<const CodeObjectBuilder> builder;
            std::shared_ptr<const SolutionLoader>    loader;
            std::shared_ptr<const SolutionStore>     store; // optional
        };

        struct Outcome
        {
            std::vector<int32_t>                             indices; // published, best first
            std::vector<std::shared_ptr<const KernelBundle>> bundles; // loaded, best first
            std::vector<Status>                              failures; // in the order they happened
            std::string                                      summary; // the backend's success note
        };

        // Throws std::invalid_argument when a required component is missing.
        explicit Jit(Components components);

        // Generate, build and check support, then publish, or load when there
        // is no store or publishing failed. Returns at most count solutions
        // that support the request, best first. Thread-safe. Sets the stage of
        // every failure it reports.
        Outcome generate(const GemmRequest&         request,
                         const DeviceTarget&             target,
                         size_t                          count,
                         size_t                          workspaceLimit,
                         const std::vector<std::string>& excludeKernels) const;

        const Components& components() const noexcept
        {
            return m_components;
        }

    private:
        Components m_components;
    };

    // The comgr builder and the TensileLite loader around one backend.
    std::shared_ptr<const Jit> makeJit(std::shared_ptr<const Backend> backend);
}

namespace hipblaslt_ext::experimental::jit::detail
{
    struct CompiledSolution
    {
        hipblaslt_jit::DeviceTarget               target;
        std::shared_ptr<const GemmRequest>   request;
        std::shared_ptr<const hipblaslt_jit::Jit> jit;
        int32_t                                   libraryIndex   = 0;
        uint64_t                                  process        = 0;
        size_t                                    workspaceLimit = 0;
        size_t                                    workspaceBytes = 0;
    };
}
