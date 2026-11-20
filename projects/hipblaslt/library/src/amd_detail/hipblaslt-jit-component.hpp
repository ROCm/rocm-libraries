// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include <Tensile/AMDGPU.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace TensileLite
{
    class Hardware;
    class ContractionSolution;
}

// Compiled-in stages of JIT solution generation. This is not a plugin ABI.
namespace hipblaslt_jit
{
    // True when kernel is one of names.
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

    // The processor a kernel is built for and its target-ID features. Strings
    // exist only where comgr, logs, files and directory names need them.
    struct Target
    {
        Processor     processor = Processor::gfx000;
        TargetFeature sramecc   = TargetFeature::Any;
        TargetFeature xnack     = TargetFeature::Any;

        // Parses a target ID, as in hipDeviceProp_t::gcnArchName or an
        // `.amdgcn_target` directive. Empty for an unknown processor, including
        // gfx000, or a feature other than one sramecc and one xnack setting.
        static std::optional<Target> fromTargetId(const std::string& targetId);
        std::string                  processorName() const; // gfx950
        std::string                  targetId() const; // gfx950:sramecc+:xnack-
        std::string                  isaName() const; // amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-
    };

    // One file. Assembly and Hip are sources. Relocatable is compiled and not
    // linked, such as a helper JitLibrary stores or an assembled main kernel.
    // Executable is a linked code object.
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

    // The sources that become one executable in JitLibrary and the solutions it
    // runs. hipFlags follow the HIP defaults, so -std=c++20 overrides -std=c++17.
    struct SourceCodeBundle
    {
        // Assembly or Hip, or for an Executable build a Relocatable built from either
        Artifact                 mainKernel;
        std::vector<Artifact>    helpers;
        std::vector<Artifact>    includes;
        std::vector<std::string> hipFlags;
        Target                   target; // the device's
        int                      codeObjectVersion = jitCodeObjectVersion;
        // The solutions the executable runs, best first: a TensileLite kernel's
        // from TensileliteAssembly::make, or those a backend with a HIP main
        // kernel builds itself. An Executable build requires each one's kernel.
        std::vector<std::shared_ptr<TensileLite::ContractionSolution>> solutions;
        // Other kernels the executable must define, such as helpers'. Jit
        // stores the HIP helper helpers[i] in JitLibrary under kernelNames[i];
        // a helper without one is compiled with the executable.
        std::vector<std::string> kernelNames;
    };

    enum class BuildKind
    {
        // The main kernel linked with every helper into one Executable.
        Executable,
        // The main kernel alone as one Relocatable, such as a helper JitLibrary stores.
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

    // Builds with comgr. An Executable build checks the ISA, the code-object
    // version and that every kernel of the solutions and kernelNames is defined.
    std::shared_ptr<const CodeObjectBuilder> makeComgrBuilder();

    // A TensileLite kernel's assembly and the solutions built from the solution
    // state in custom.config of its .amdgpu_metadata.
    struct TensileliteAssembly : Artifact
    {
        std::vector<std::shared_ptr<TensileLite::ContractionSolution>> solutions;
        Target                                                         target;

        // Assembles source.mainKernel alone into relocatable and reads assembly
        // from its metadata note. Needs a MessagePack build of TensileLite and a
        // HIP device, which TensileLite reads while it builds some predicates.
        static Status make(const CodeObjectBuilder& builder,
                           const SourceCodeBundle&  source,
                           TensileliteAssembly&     assembly,
                           Artifact&                relocatable);
    };

    // source_bundle::readSourceBundle, then TensileliteAssembly::make: mainKernel
    // becomes the assembled Relocatable and solutions the kernel's. Fails unless
    // the main kernel is assembly.
    Status readTensileSourceBundle(const CodeObjectBuilder&     builder,
                                   const std::filesystem::path& bundle,
                                   const Target&                target,
                                   int                          codeObjectVersion,
                                   SourceCodeBundle&            source);

    // Appends status.log to <scratch>/comgr.log and names
    // that file in the message. Does nothing without a log or a scratch directory.
    void keepBuildLog(Status& status, const std::filesystem::path& scratch);

    using GemmRequest = hipblaslt_ext::experimental::jit::detail::GemmRequest;

    // The Target of a device: gcnArchName parsed.
    struct DeviceTarget : Target
    {
        int                                    device = -1;
        hipDeviceProp_t                        properties{};
        int                                    wavefrontSize = 0;
        int                                    cuCount       = 0;
        std::shared_ptr<TensileLite::Hardware> hardware;

        // Fails when gcnArchName is not a known target ID.
        static Status make(int device, DeviceTarget& target);
    };

    struct GenerationRequest
    {
        const GemmRequest&       request;
        const DeviceTarget&      target;
        size_t                   count          = 1;
        size_t                   workspaceLimit = 0;
        std::vector<std::string> excludeKernels; // kernels the caller already has
        std::filesystem::path    scratch; // private directory owned by this call
        int codeObjectVersion = jitCodeObjectVersion; // for generators and each bundle
    };

    struct BackendInfo
    {
        std::string id;
        uint32_t    version = 0; // increment when the generated solutions can change
    };

    class Backend
    {
    public:
        virtual ~Backend()                               = default;
        virtual const BackendInfo& info() const noexcept = 0;
        // Returns up to request.count source bundles for the request, best
        // first, and loads nothing. Each bundle names the request's target, its
        // code-object version and the solutions its executable runs. NotSupported
        // means the request is outside the backend's domain.
        virtual Status generate(const GenerationRequest&, std::vector<SourceCodeBundle>&) const = 0;
    };

    class SolutionStore
    {
    public:
        virtual ~SolutionStore() = default;
        // Publishes executable with solutions, those of its bundle that support
        // the request, best first, as JitLibrary::publish does. Returns one
        // library index per solution, in order.
        virtual Status
            publish(const GemmRequest&,
                    const DeviceTarget&,
                    const Artifact& executable,
                    const std::vector<std::shared_ptr<TensileLite::ContractionSolution>>& solutions,
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
            std::shared_ptr<const SolutionStore>     store;
        };

        struct Outcome
        {
            std::vector<int32_t> indices; // published, best first
            std::vector<Status>  failures; // in the order they happened
        };

        // Throws std::invalid_argument when a component is missing.
        explicit Jit(Components components);

        // Generate, check each bundle's solutions with supportingSolutions,
        // build, then publish to the store. Returns the library indices of at
        // most count published solutions that support the request, best first;
        // JitLibrary is the only place they live, and nothing is loaded in
        // process. A bundle whose support check, build or publish fails adds a
        // failure and contributes no solution. Thread-safe. Sets the stage of
        // every failure it reports.
        Outcome generate(const GemmRequest&              request,
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

    // The comgr builder and a store in the process JIT solution library around
    // one backend.
    std::shared_ptr<const Jit> makeJit(std::shared_ptr<const Backend> backend);
}
