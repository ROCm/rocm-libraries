// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include <cstdint>
#include <filesystem>
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

    // A header a HIP unit includes by name.
    struct IncludeFile
    {
        std::string          name;
        std::vector<uint8_t> bytes;
    };

    // A source file the builder assembles or compiles.
    struct BuildUnit
    {
        enum class Kind
        {
            Assembly,
            Hip,
        };
        std::string              name;
        std::vector<uint8_t>     bytes;
        Kind                     kind = Kind::Assembly;
        std::vector<IncludeFile> includes; // Hip only
    };

    // One CustomKernel library entry plus the code that defines its kernels.
    struct GeneratedSolution
    {
        // A TensileLite library holding local solutions 0 to N-1, best first;
        // each solution carries its customKernel record.
        std::vector<uint8_t>     entry;
        std::vector<std::string> kernelNames; // every main kernel the solutions name
        std::vector<BuildUnit>   units;
        // Appended after the HIP defaults, so -std=c++20 overrides -std=c++17.
        std::vector<std::string> hipFlags;
    };

    constexpr int jitCodeObjectVersion = 4;

    struct BuildRequest
    {
        std::string           targetId; // gcnArchName, e.g. "gfx950:sramecc+:xnack-"
        int                   codeObjectVersion = jitCodeObjectVersion;
        std::filesystem::path scratch; // receives comgr.log after a failure, or empty
    };

    struct CodeObject
    {
        std::vector<uint8_t> bytes;
    };

    struct BuiltSolution
    {
        GeneratedSolution generated;
        CodeObject        object;
    };

    class CodeObjectBuilder
    {
    public:
        virtual ~CodeObjectBuilder() = default;
        virtual Status
            build(const GeneratedSolution&, const BuildRequest&, BuiltSolution&) const = 0;
    };

    // Builds every unit of a solution with comgr and links them into one code
    // object for BuildRequest::targetId.
    std::shared_ptr<const CodeObjectBuilder> makeComgrBuilder();

    using OperationRequest = hipblaslt_ext::experimental::jit::detail::OperationRequest;
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
        const OperationRequest&  request;
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
        std::string version; // changes whenever the generated solutions can change
    };

    class Backend
    {
    public:
        virtual ~Backend()                               = default;
        virtual const BackendInfo& info() const noexcept = 0;
        // Returns entries holding up to request.count solutions for the request,
        // best first, and loads nothing. NotSupported means the request is
        // outside the backend's domain.
        virtual Status generate(const GenerationRequest&, std::vector<GeneratedSolution>&) const
            = 0;
    };

    class SolutionLoader
    {
    public:
        virtual ~SolutionLoader() = default;
        // Evaluates the predicates and workspace of the entry's solutions for the
        // request and returns the local indices of those that support it, best
        // first. Loads no code.
        virtual Status support(const BuiltSolution&,
                               const OperationRequest&,
                               const DeviceTarget&,
                               size_t            workspaceLimit,
                               std::vector<int>& indices) const
            = 0;
        // Loads the code objects once and returns a process-local executable
        // bundle for each of the solutions at indices.
        virtual Status load(const BuiltSolution&,
                            const OperationRequest&,
                            const DeviceTarget&,
                            size_t                                            workspaceLimit,
                            const std::vector<int>&                           indices,
                            std::vector<std::shared_ptr<const KernelBundle>>& bundles) const
            = 0;
    };

    // Built entries, each with the local indices of its solutions that support
    // a request, best first.
    using SupportedSolutions = std::vector<std::pair<BuiltSolution, std::vector<int>>>;

    class SolutionStore
    {
    public:
        virtual ~SolutionStore() = default;
        // Up to count indices of stored solutions for exactly this request that
        // need at most maxWorkspaceBytes, best first, without excluded kernels.
        virtual Status lookup(const OperationRequest&,
                              const DeviceTarget&,
                              size_t                          count,
                              size_t                          maxWorkspaceBytes,
                              const std::vector<std::string>& excludeKernels,
                              std::vector<int32_t>&           indices) const
            = 0;
        // Returns one library index per supported solution, in order.
        virtual Status publish(const OperationRequest&,
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
        Outcome generate(const OperationRequest&         request,
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
}

namespace hipblaslt_ext::experimental::jit::detail
{
    struct CompiledSolution
    {
        hipblaslt_jit::DeviceTarget               target;
        std::shared_ptr<const OperationRequest>   request;
        std::shared_ptr<const hipblaslt_jit::Jit> jit;
        std::shared_ptr<const KernelBundle>       bundle;
        uint64_t                                  process        = 0;
        size_t                                    workspaceLimit = 0;
        size_t                                    workspaceBytes = 0;
    };
}
