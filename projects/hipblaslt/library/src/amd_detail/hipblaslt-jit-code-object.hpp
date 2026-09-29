// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// In-process code-object construction through AMD comgr. The build, link and
// metadata functions are noexcept: failures, including allocation failure, are
// reported through Status and a log carrying the comgr diagnostics of every stage
// that ran. Only the Target string helpers may throw (std::bad_alloc).
namespace hipblaslt_jit::code_object
{
    enum class Status
    {
        Success,
        InvalidArgument,
        ComgrError,
        NoOutput,
        MetadataError,
        InternalError,
    };

    const char* toString(Status status) noexcept;

    struct Target
    {
        // Processor name without features, for example "gfx950" or "gfx1250".
        std::string gfx;
        // Target-ID features, for example {"sramecc+", "xnack-"}. Empty means "any";
        // the assembly's .amdgcn_target directive must name the same features.
        std::vector<std::string> features;
        // 0 selects the processor default (64 for gfx9, otherwise 32).
        int wavefrontSize = 0;

        // Parses "gfx950:sramecc+:xnack-" (hipDeviceProp_t::gcnArchName form).
        static Target fromTargetId(const std::string& targetId, int wavefrontSize = 0);
        std::string   targetId() const; // gfx950:sramecc+:xnack-
        std::string   isaName() const; // amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-
        int           resolvedWavefrontSize() const;
    };

    struct AssemblySource
    {
        std::string name;
        std::string text;
    };

    struct HipSource
    {
        std::string name;
        std::string text;
    };

    // Header made available to #include "name" without touching the file system.
    struct IncludeFile
    {
        std::string name;
        std::string text;
    };

    struct Relocatable
    {
        std::string       name;
        std::vector<char> bytes;
    };

    enum class HipPipeline
    {
        // COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC -> CODEGEN_BC_TO_RELOCATABLE, with IR
        // optimization only in the first stage (as the clang driver does).
        Staged,
        // COMPILE_SOURCE_TO_RELOCATABLE in one action.
        SourceToRelocatable,
    };

    struct Options
    {
        // 4, 5 or 6. Applies to assembly and HIP sources.
        int codeObjectVersion = 5;
        // Adds per-architecture defaults: "-Xclangas -target-feature -Xclangas
        // +real-true16" for gfx11*/gfx12* assembly, matching TensileLite.
        bool architectureDefaults = true;
        // Rewrites the target ID in ".amdgcn_target" / "amdhsa.target" to
        // Target::targetId(), as TensileLite's assembler does before assembling.
        bool                     retargetAssembly = false;
        std::vector<std::string> assemblerFlags;
        HipPipeline              hipPipeline = HipPipeline::SourceToRelocatable;
        // HIP front-end flags, appended after the defaults (-O3 -std=c++17 ... -cuid=<id>),
        // so an explicit -cuid here overrides the content-derived one.
        std::vector<std::string> compilerFlags;
        // Staged codegen flags, appended after the defaults
        // (-O3 -Xclang -disable-llvm-optzns): re-running IR optimization on
        // already optimized bitcode changes the generated code.
        std::vector<std::string> codegenFlags;
        // Link runs through comgr's clang driver: pass "-Xlinker --build-id=sha1",
        // not bare lld options.
        std::vector<std::string> linkerFlags;
        // ROCm prefix providing hip/*.h for HIP sources ("--rocm-path="). Without it
        // comgr finds the headers only through the HIP_PATH environment variable
        // (not ROCM_PATH) for the TheRock layout. See rocmPath().
        std::string              rocmPath;
        std::vector<std::string> includeDirectories;
        std::vector<IncludeFile> includes;
        // Prepends "-include __clang_hip_runtime_wrapper.h"; comgr omits the HIP
        // runtime wrapper that the clang driver normally injects.
        bool includeHipRuntimeWrapper = true;
    };

    struct Result
    {
        Status            status = Status::InternalError;
        std::string       log;
        std::vector<char> bytes; // uncompressed, unbundled executable ELF
        bool              ok() const noexcept
        {
            return status == Status::Success;
        }
    };

    struct RelocatableResult
    {
        Status                   status = Status::InternalError;
        std::string              log;
        std::vector<Relocatable> objects;
        bool                     ok() const noexcept
        {
            return status == Status::Success;
        }
    };

    struct KernelArgument
    {
        std::string   name;
        std::string   valueKind;
        std::uint64_t offset = 0;
        std::uint64_t size   = 0;
    };

    struct KernelMetadata
    {
        std::string                 name;
        std::string                 symbol; // descriptor symbol, "<name>.kd"
        std::uint64_t               kernargSegmentSize      = 0;
        std::uint64_t               kernargSegmentAlign     = 0;
        std::uint64_t               groupSegmentFixedSize   = 0; // static LDS bytes
        std::uint64_t               privateSegmentFixedSize = 0;
        std::uint32_t               vgprCount               = 0;
        std::uint32_t               agprCount               = 0;
        std::uint32_t               sgprCount               = 0;
        std::uint32_t               vgprSpillCount          = 0;
        std::uint32_t               sgprSpillCount          = 0;
        std::uint32_t               wavefrontSize           = 0;
        std::uint32_t               maxFlatWorkgroupSize    = 0;
        std::vector<KernelArgument> arguments;
    };

    struct CodeObjectMetadata
    {
        std::string                 isaName; // from comgr
        std::string                 target; // amdhsa.target
        int                         codeObjectVersion = 0; // from the ELF ABI version
        std::vector<KernelMetadata> kernels;
    };

    struct MetadataResult
    {
        Status             status = Status::InternalError;
        std::string        log;
        CodeObjectMetadata metadata;
        bool               ok() const noexcept
        {
            return status == Status::Success;
        }
    };

    // Assembles each source to a relocatable object.
    RelocatableResult assembleRelocatables(const std::vector<AssemblySource>& sources,
                                           const Target&                      target,
                                           const Options&                     options) noexcept;

    // Compiles each HIP source as its own translation unit (Options::hipPipeline) with
    // a content-derived -cuid, so the resulting objects can be linked together.
    RelocatableResult compileHipRelocatables(const std::vector<HipSource>& sources,
                                             const Target&                 target,
                                             const Options&                options) noexcept;

    // Links relocatables from either producer, including assembled and compiled
    // objects together, into one executable code object.
    Result link(const std::vector<Relocatable>& objects,
                const Target&                   target,
                const Options&                  options) noexcept;

    // assembleRelocatables + link: all sources share one executable.
    Result assemble(const std::vector<AssemblySource>& sources,
                    const Target&                      target,
                    const Options&                     options) noexcept;

    // compileHipRelocatables + link: all sources share one executable.
    Result compileHip(const std::vector<HipSource>& sources,
                      const Target&                 target,
                      const Options&                options) noexcept;

    MetadataResult readMetadata(const void* elf, std::size_t size) noexcept;
    MetadataResult readMetadata(const std::vector<char>& elf) noexcept;

    // "major.minor" of the loaded comgr library, for diagnostics.
    std::string comgrVersion() noexcept;

    // "<major>.<minor>:<library path>:<size>:<mtime ns>" of the loaded comgr
    // library, or "major.minor" when the library file cannot be identified.
    std::string comgrIdentity() noexcept;

    // The ROCm prefix for Options::rocmPath: HIP_PATH when set, otherwise the
    // prefix of the loaded HIP runtime (<prefix>/lib/libamdhip64.so). Empty if
    // neither is known.
    std::string rocmPath() noexcept;

    // Who decides whether comgr uses its on-disk cache in this process.
    enum class ComgrCache
    {
        Default, // HIPBLASLT_JIT is not 1 or 2: comgr's own default applies
        UserValue, // the user set AMD_COMGR_CACHE, and it is kept
        Disabled, // hipBLASLt set AMD_COMGR_CACHE=0 when it was loaded
        DisabledLate, // hipBLASLt set AMD_COMGR_CACHE=0 after it was loaded
    };

    const char* toString(ComgrCache setting) noexcept;

    // The precedence rule alone: Disabled when jitMode (HIPBLASLT_JIT) is "1" or
    // "2" and comgrCache (AMD_COMGR_CACHE) is unset, UserValue when it is set,
    // otherwise Default.
    ComgrCache comgrCachePolicy(const char* jitMode, const char* comgrCache) noexcept;

    // Applies comgrCachePolicy to the environment, setting AMD_COMGR_CACHE=0
    // without overwriting a user value. comgr reads AMD_COMGR_CACHE and its cache
    // directory once, at the first cached action of any comgr user in the process
    // (hipRTC and rocRoller included), so this also runs when the library is
    // loaded, and the setting then applies to all of them. DisabledLate means a
    // comgr action before this call may already have fixed the previous value.
    ComgrCache prepareProcessEnvironment() noexcept;
}
