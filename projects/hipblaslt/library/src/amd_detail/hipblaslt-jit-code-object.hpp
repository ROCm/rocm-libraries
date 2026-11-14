// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"

#include <cstddef>
#include <string>
#include <vector>

// In-process code-object construction through AMD comgr. Failures, including
// allocation failure, are reported through Status and a log carrying the comgr
// diagnostics of every stage that ran. The Target string functions may throw
// (std::bad_alloc).
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

    struct Options
    {
        // 4, 5 or 6.
        int codeObjectVersion = 5;
        // 32 or 64 for assembly, or 0 for the processor default (64 before
        // gfx10, otherwise 32). HIP always keeps the processor default.
        int wavefrontSize = 0;
    };

    struct Result
    {
        Status                  status = Status::InternalError;
        std::string             log;
        hipblaslt_jit::Artifact artifact; // Relocatable or Executable when ok
        bool                    ok() const noexcept
        {
            return status == Status::Success;
        }
    };

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

    // The Relocatable for bundle.mainKernel, which sees bundle.includes. Assembly
    // is retargeted to target.targetId(); HIP gets bundle.hipFlags after the
    // defaults (-O3 -std=c++17 ... -cuid=<id>), so an explicit -cuid overrides.
    Result compileRelocatable(const hipblaslt_jit::SourceCodeBundle& bundle,
                              const Target&                          target,
                              const Options&                         options) noexcept;

    // Links Relocatables into one Executable.
    Result linkRelocatables(const std::vector<hipblaslt_jit::Artifact>& objects,
                            const Target&                               target) noexcept;

    ObjectCheck inspect(const std::vector<uint8_t>& elf) noexcept;

    // The NT_AMDGPU_METADATA note of a relocatable or executable code object:
    // the MessagePack document the assembler encodes from .amdgpu_metadata,
    // such as the solution state a TensileLite kernel keeps in custom.config.
    struct MetadataNote
    {
        Status               status = Status::InternalError;
        std::string          log;
        std::vector<uint8_t> bytes;
        bool                 ok() const noexcept
        {
            return status == Status::Success;
        }
    };

    MetadataNote metadataNote(const std::vector<uint8_t>& elf) noexcept;

    // HIP_PATH when set, otherwise the prefix of the loaded HIP runtime
    // (<prefix>/lib/libamdhip64.so). Empty if neither is known.
    std::string rocmPath() noexcept;
}
