// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Builds the hand-written HIP kernel with comgr for each committed assembly
// target, as an executable and as a relocatable, then each assembly file alone,
// from its relocatable, and linked with that kernel from its source and from
// the relocatable, and checks that each executable defines every expected
// kernel. Needs no GPU.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    const char* const kernelName = "hipblaslt_jit_builder_test_scale";

    // The kernel and target an assembly file declares in its .amdhsa_kernel
    // and .amdgcn_target directives.
    struct Assembly
    {
        std::string kernel;
        hj::Target  target;
    };

    bool after(const std::string& line, const std::string& prefix, std::string& rest)
    {
        if(line.rfind(prefix, 0) != 0)
            return false;
        rest = line.substr(prefix.size());
        return true;
    }

    Assembly readAssembly(const std::string& text)
    {
        Assembly           result;
        std::string        targetId;
        std::istringstream lines(text);
        for(std::string line, rest; std::getline(lines, line);)
        {
            if(after(line, ".amdhsa_kernel ", rest) && result.kernel.empty())
                result.kernel = rest;
            else if(after(line, ".amdgcn_target \"amdgcn-amd-amdhsa--", rest))
                targetId = rest.substr(0, rest.find('"'));
        }
        const auto target = hj::Target::fromTargetId(targetId);
        {
            INFO(("The assembly names no kernel or known target"));
            REQUIRE((!result.kernel.empty() && target));
        }
        result.target = *target;
        return result;
    }

    hj::Artifact build(const hj::SourceCodeBundle& source, hj::BuildKind kind, const fs::path& scratch)
    {
        hj::Artifact built;
        auto         status = hj::makeComgrBuilder()->build(source, kind, built);
        hj::keepBuildLog(status, scratch);
        {
            INFO(("The build for " + source.target.targetId() + " failed: " + status.message));
            REQUIRE((status.ok()));
        }
        const auto expected = kind == hj::BuildKind::Executable ? hj::Artifact::Kind::Executable
                                                                : hj::Artifact::Kind::Relocatable;
        {
            INFO(("The build did not return one artifact of the kind requested"));
            REQUIRE((built.kind == expected && !built.bytes.empty()));
        }
        return built;
    }

    void buildAssembly(const fs::path&     file,
                       const Assembly&     assembly,
                       const hj::Artifact& kernel,
                       const hj::Artifact& relocatable,
                       const fs::path&     scratch)
    {
        const auto           text = hipblaslt_jit_test::readFile(file);
        const auto           name = file.filename().u8string();
        hj::SourceCodeBundle source;
        source.mainKernel  = {name,
                              hj::Artifact::Kind::Assembly,
                              std::vector<uint8_t>(text.begin(), text.end())};
        source.target      = assembly.target;
        source.kernelNames = {assembly.kernel};
        build(source, hj::BuildKind::Executable, scratch);
        std::cout << "PASS comgr built " << name << " for " << assembly.target.targetId() << '\n';

        // The note TensileliteAssembly reads its solution from.
        auto       assembled = source;
        assembled.mainKernel = build(source, hj::BuildKind::Relocatable, scratch);
        const auto note      = hj::code_object::metadataNote(assembled.mainKernel.bytes);
        {
            INFO(("The relocatable of " + name + " has no metadata note with custom.config: "
                  + note.log));
            const std::string document(note.bytes.begin(), note.bytes.end());
            REQUIRE((note.ok() && document.find("custom.config") != std::string::npos));
        }
        build(assembled, hj::BuildKind::Executable, scratch);
        hj::Artifact twice;
        const auto   again
            = hj::makeComgrBuilder()->build(assembled, hj::BuildKind::Relocatable, twice);
        {
            INFO(("A relocatable build accepted a relocatable main kernel"));
            REQUIRE((!again.ok() && again.stage == hj::Stage::Build));
        }
        std::cout << "PASS comgr linked " << name << " from its relocatable, whose metadata note "
                  << "holds custom.config\n";

        source.kernelNames = {assembly.kernel, kernelName};
        for(const auto& helper : {kernel, relocatable})
        {
            source.helpers = {helper};
            build(source, hj::BuildKind::Executable, scratch);
        }
        std::cout << "PASS comgr linked " << name
                  << " with the HIP kernel from its source and from its relocatable\n";

        source.helpers     = {};
        source.kernelNames = {assembly.kernel, "hipblaslt_jit_builder_test_missing"};
        hj::Artifact built;
        const auto   missing = hj::makeComgrBuilder()->build(source, hj::BuildKind::Executable, built);
        {
            INFO(("The build accepted a kernel name the executable does not define"));
            REQUIRE((!missing.ok() && missing.stage == hj::Stage::Build
                     && missing.message.find("hipblaslt_jit_builder_test_missing")
                            != std::string::npos));
        }
    }
}

TEST_CASE("target IDs parse into the typed target", "[jit-cpu]")
{
    const auto device = hj::Target::fromTargetId("gfx950:sramecc+:xnack-");
    {
        INFO(("gfx950:sramecc+:xnack- did not parse into its processor and features"));
        REQUIRE((device && device->processor == hj::Processor::gfx950
                 && device->sramecc == hj::TargetFeature::On
                 && device->xnack == hj::TargetFeature::Off && device->asicRevision == -1));
    }
    {
        INFO(("The target strings changed"));
        REQUIRE((device->processorName() == "gfx950"
                 && device->targetId() == "gfx950:sramecc+:xnack-"
                 && device->isaName() == "amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-"
                 && device->libraryArch() == "gfx950"));
    }
    auto gfx1250 = hj::Target::fromTargetId("gfx1250");
    REQUIRE((gfx1250 && gfx1250->targetId() == "gfx1250" && gfx1250->libraryArch() == "gfx1250"));
    gfx1250->asicRevision = 0;
    {
        INFO(("A gfx1250 A0 part does not load the gfx1250v0 library"));
        REQUIRE((gfx1250->targetId() == "gfx1250" && gfx1250->libraryArch() == "gfx1250v0"));
    }
    for(const char* id : {"",
                          "gfx000",
                          "gfx9500",
                          "gfx1250-strict",
                          "gfx950:",
                          "gfx950:xnack",
                          "gfx950:xnack-:xnack+",
                          "gfx950:tgsplit+"})
    {
        INFO((std::string("Parsed the unknown target ID '") + id + "'"));
        REQUIRE((!hj::Target::fromTargetId(id)));
    }
    std::cout << "PASS target IDs parse, and unknown processors and features are rejected\n";
}

TEST_CASE("comgr builds the HIP kernel and every committed assembly file", "[jit-cpu]")
{
    const auto source  = hipblaslt_jit_test::readFile(fs::u8path(HIPBLASLT_JIT_KERNEL));
    const auto scratch = fs::u8path(HIPBLASLT_JIT_SCRATCH);
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    const hj::Artifact kernel{"builder_test_kernel.hip",
                              hj::Artifact::Kind::Hip,
                              std::vector<uint8_t>(source.begin(), source.end())};

    std::map<std::string, std::vector<std::pair<fs::path, Assembly>>> byTarget;
    for(const auto& arch : fs::directory_iterator(fs::u8path(HIPBLASLT_JIT_DATA)))
    {
        if(!arch.is_directory())
            continue;
        for(const auto& sources : fs::directory_iterator(arch))
        {
            if(!sources.is_directory())
                continue;
            for(const auto& file : fs::directory_iterator(sources.path() / "sources"))
            {
                if(file.path().extension() != ".s")
                    continue;
                const auto assembly = readAssembly(hipblaslt_jit_test::readFile(file.path()));
                byTarget[assembly.target.targetId()].push_back({file.path(), assembly});
            }
        }
    }
    int built = 0;
    for(const auto& [targetId, files] : byTarget)
    {
        hj::SourceCodeBundle hip;
        hip.mainKernel  = kernel;
        hip.target      = files.front().second.target;
        hip.kernelNames = {kernelName};
        build(hip, hj::BuildKind::Executable, scratch);
        const auto relocatable = build(hip, hj::BuildKind::Relocatable, scratch);
        std::cout << "PASS comgr built the HIP kernel for " << targetId
                  << " as an executable and as a relocatable\n";
        for(const auto& [file, assembly] : files)
        {
            buildAssembly(file, assembly, kernel, relocatable, scratch);
            ++built;
        }
    }
    {
        INFO(("No committed assembly under the data directory"));
        REQUIRE((built > 0));
    }
    std::cout << "PASS the build of every assembly file fails when a kernel name is not defined\n";

    hj::SourceCodeBundle unknown;
    unknown.mainKernel = kernel;
    hj::Artifact out;
    const auto   noTarget = hj::makeComgrBuilder()->build(unknown, hj::BuildKind::Executable, out);
    {
        INFO(("A bundle without a target processor was built"));
        REQUIRE((!noTarget.ok() && noTarget.stage == hj::Stage::Build));
    }
    unknown.target  = byTarget.begin()->second.front().second.target;
    unknown.helpers = {kernel};
    const auto withHelpers
        = hj::makeComgrBuilder()->build(unknown, hj::BuildKind::Relocatable, out);
    {
        INFO(("A relocatable build accepted helpers"));
        REQUIRE((!withHelpers.ok() && withHelpers.stage == hj::Stage::Build));
    }
    std::cout << "PASS the builder rejects a bundle without a target and a relocatable with helpers\n";
}
