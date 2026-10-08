// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include <Tensile/AMDGPU.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace TensileLite
{
    class Hardware;
    class ContractionSolution;
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
        std::string log; // the full diagnostics behind message, such as the comgr log, or empty
        bool        ok() const noexcept
        {
            return code == Code::Success;
        }
    };

    using Processor = TensileLite::AMDGPU::Processor;

    // A target-ID feature. Any is the feature left out of the target ID.
    enum class TargetFeature
    {
        Any,
        On,
        Off,
    };

    // The processor a kernel is built for, its target-ID features, and the
    // ASIC revision that selects the GEMM library subtree. Strings exist only
    // where comgr, logs, files and directory names need them.
    struct Target
    {
        Processor     processor    = Processor::gfx000;
        TargetFeature sramecc      = TargetFeature::Any;
        TargetFeature xnack        = TargetFeature::Any;
        int           asicRevision = -1; // hipDeviceProp_t::asicRevision, or -1 when unknown

        // Parses a target ID, as in hipDeviceProp_t::gcnArchName or an
        // `.amdgcn_target` directive. Empty for an unknown processor, including
        // gfx000, or a feature other than one sramecc and one xnack setting.
        // asicRevision is left -1.
        static std::optional<Target> fromTargetId(const std::string& targetId);
        std::string                  processorName() const; // gfx950
        std::string                  targetId() const; // gfx950:sramecc+:xnack-
        std::string                  isaName() const; // amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-
        // The GEMM library subtree: gfx1250v0 for a gfx1250 A0 part, otherwise
        // processorName().
        std::string libraryArch() const;
    };

    // One file. Assembly and Hip are sources. Relocatable is compiled and not
    // linked, such as a helper from the helper library or an assembled main
    // kernel. Executable is a linked code object.
    struct Artifact
    {
        std::string name;
        enum class Kind
        {
            Assembly,
            Hip,
            Relocatable,
            Executable,
        };
        Kind                 kind = Kind::Assembly;
        std::vector<uint8_t> bytes;
    };

    constexpr int jitCodeObjectVersion = 4;

    // The sources that become one solution in JitLibrary, and how to build
    // them. Not a TensileLite library entry. helpers are HIP sources, compiled
    // with the main kernel, or Relocatables from the helper library, which keeps
    // one per backend id and helper name. includes are headers. hipFlags are
    // appended after the HIP defaults, so -std=c++20 overrides -std=c++17.
    struct SourceCodeBundle
    {
        // Assembly or Hip, or for an Executable build a Relocatable built from either
        Artifact                 mainKernel;
        std::vector<Artifact>    helpers;
        std::vector<Artifact>    includes;
        std::vector<std::string> hipFlags;
        Target                   target; // the device's
        int                      codeObjectVersion = jitCodeObjectVersion;
        // Kernels the executable must define; empty accepts what was built.
        std::vector<std::string> kernelNames;
    };

    enum class BuildKind
    {
        // The main kernel linked with every helper into one Executable.
        Executable,
        // The main kernel alone as one Relocatable. The helper library stores it.
        Relocatable,
    };

    class CodeObjectBuilder
    {
    public:
        virtual ~CodeObjectBuilder() = default;
        // Sets out to the one artifact built. Writes no file; a failure carries
        // the compiler log in Status::log.
        virtual Status build(const SourceCodeBundle&, BuildKind, Artifact& out) const = 0;
    };

    // Builds a SourceCodeBundle with comgr. An Executable build checks the
    // ISA, the code-object version and the bundle's kernel names. JitLibrary's
    // helper library is the helper cache. comgr's own file cache is not part of
    // the library.
    std::shared_ptr<const CodeObjectBuilder> makeComgrBuilder();

    // A TensileLite kernel's assembly and the solutions it implements. They
    // come from the solution state the kernel keeps in custom.config of its
    // .amdgpu_metadata, so the kernel name and the kernel-argument versions are
    // theirs. target is a field because a ContractionSolution states its
    // target only as a hardware predicate.
    struct TensileliteAssembly : Artifact
    {
        std::vector<std::shared_ptr<TensileLite::ContractionSolution>> solutions;
        Target                                                         target;

        // Assembles source.mainKernel, an Assembly, alone into relocatable and
        // reads assembly from its metadata note. An Executable build of source
        // with relocatable as its main kernel links it without assembling again.
        // Needs a MessagePack build of TensileLite and a HIP device, which
        // TensileLite reads while it builds some predicates.
        static Status make(const CodeObjectBuilder& builder,
                           const SourceCodeBundle&  source,
                           TensileliteAssembly&     assembly,
                           Artifact&                relocatable);
    };

    // Appends status.log to <scratch>/comgr.log, sets status.logPath and names
    // that file in the message. Does nothing without a log or a scratch directory.
    void keepBuildLog(Status& status, const std::filesystem::path& scratch);

    using GemmRequest = hipblaslt_ext::experimental::jit::detail::GemmRequest;
    using KernelBundle     = hipblaslt_ext::experimental::jit::detail::KernelBundle;

    // "configure", "generate", ... as reports name stages.
    const char* toString(Stage stage) noexcept;

    // The Target of a device: gcnArchName parsed, with its asicRevision.
    struct DeviceTarget : Target
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        int             wavefrontSize = 0;
        int             cuCount       = 0;
        std::shared_ptr<TensileLite::Hardware> hardware;

        // Fails when gcnArchName is not a known target ID.
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
        int codeObjectVersion = jitCodeObjectVersion; // for generators and each bundle
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
        // first, and loads nothing. Each bundle names the request's target, its
        // code-object version and the kernels it must define. NotSupported
        // means the request is outside the backend's domain.
        virtual Status generate(const GenerationRequest&, std::vector<SourceCodeBundle>&) const
            = 0;
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
        // Evaluates the predicates and workspace of the TensileLite record's
        // solutions for the request and returns the local indices of those that
        // support it, best first. Loads no code.
        virtual Status support(const std::vector<uint8_t>& record,
                               const GemmRequest&,
                               const DeviceTarget&,
                               size_t            workspaceLimit,
                               std::vector<int>& indices) const
            = 0;
        // Loads the executable once and returns a process-local executable
        // bundle for each of the record's solutions at indices.
        virtual Status load(const Artifact&             executable,
                            const std::vector<uint8_t>& record,
                            const GemmRequest&,
                            const DeviceTarget&,
                            size_t                                            workspaceLimit,
                            const std::vector<int>&                           indices,
                            std::vector<std::shared_ptr<const KernelBundle>>& bundles) const
            = 0;
    };

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
        // Publishes the executable with the solutions at the local indices of
        // its TensileLite record, best first. Returns one library index per
        // solution, in order.
        virtual Status publish(const GemmRequest&,
                               const DeviceTarget&,
                               const Artifact&             executable,
                               const std::vector<uint8_t>& record,
                               const std::vector<int>&     solutions,
                               std::vector<int32_t>&       indices) const
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
