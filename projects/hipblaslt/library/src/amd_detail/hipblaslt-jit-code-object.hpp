// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"

#include <cstddef>
#include <string>
#include <vector>

// In-process code-object construction through AMD comgr. build and inspect are
// noexcept: failures, including allocation failure, are reported through Status
// and a log carrying the comgr diagnostics of every stage that ran. Assembly,
// HIP compilation and linking stay inside the implementation. Only the Target
// string helpers may throw (std::bad_alloc).
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
        // Extra headers offered to every HIP source as #include "name".
        // A bundle's includes are added as well.
        std::vector<hipblaslt_jit::Artifact> includes;
        // Prepends "-include __clang_hip_runtime_wrapper.h"; comgr omits the HIP
        // runtime wrapper that the clang driver normally injects.
        bool includeHipRuntimeWrapper = true;
    };

    struct Result
    {
        Status                  status = Status::InternalError;
        std::string             log;
        hipblaslt_jit::Artifact codeObject; // Kind::CodeObject when ok
        bool                    ok() const noexcept
        {
            return status == Status::Success;
        }
    };

    // Kernel names, ISA and ELF code-object version. The builder uses this to
    // reject an object that does not match the request. hipBLASLt loads the
    // bytes from Result.
    struct ObjectCheck
    {
        Status                   status = Status::InternalError;
        std::string              log;
        std::string              isaName;
        int                      codeObjectVersion = 0;
        std::vector<std::string> kernelNames;
        bool                     ok() const noexcept
        {
            return status == Status::Success;
        }
    };

    // Assembles or compiles bundle.mainKernel, with bundle.includes, and links
    // one executable code object. ComgrBuilder compiles helpers itself, asking
    // JitLibrary before it does, so they are not compiled here.
    // Target::wavefrontSize applies to assembly; HIP keeps the processor default.
    Result build(const hipblaslt_jit::SourceCodeBundle& bundle,
                 const Target&                          target,
                 const Options&                         options) noexcept;

    ObjectCheck inspect(const void* elf, std::size_t size) noexcept;
    ObjectCheck inspect(const std::vector<uint8_t>& elf) noexcept;

    // "major.minor" of the loaded comgr library, for diagnostics.
    std::string comgrVersion() noexcept;

    // "<major>.<minor>:<library path>:<size>:<mtime ns>" of the loaded comgr
    // library, or "major.minor" when the library file cannot be identified.
    std::string comgrIdentity() noexcept;

    // The ROCm prefix for Options::rocmPath: HIP_PATH when set, otherwise the
    // prefix of the loaded HIP runtime (<prefix>/lib/libamdhip64.so). Empty if
    // neither is known.
    std::string rocmPath() noexcept;
}
