// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

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
}
