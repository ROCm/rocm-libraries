// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Builds the hand-written HIP kernel with comgr for each committed assembly
// target, as an executable and as a relocatable, then each assembly file alone,
// from its relocatable, and linked with that kernel from its source and from
// the relocatable, and checks that each executable defines every expected
// kernel. Also refuses each assembly file for a target it does not declare.
// Needs no GPU.
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
        JIT_REQUIRE("The assembly names no kernel or known target",
                    !result.kernel.empty() && target);
        result.target = *target;
        return result;
    }

    // Every committed assembly file under the data directory, with what it declares.
    std::vector<std::pair<fs::path, Assembly>> committedAssembly()
    {
        std::vector<std::pair<fs::path, Assembly>> files;
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
                    files.push_back(
                        {file.path(), readAssembly(hipblaslt_jit_test::readFile(file.path()))});
                }
            }
        }
        JIT_REQUIRE("No committed assembly under the data directory", !files.empty());
        return files;
    }

    hj::SourceCodeBundle assemblyBundle(const std::string& name,
                                        const std::string& text,
                                        const std::string& kernel,
                                        const hj::Target&  target)
    {
        hj::SourceCodeBundle source;
        source.mainKernel
            = {name, hj::Artifact::Kind::Assembly, std::vector<uint8_t>(text.begin(), text.end())};
        source.target      = target;
        source.kernelNames = {kernel};
        return source;
    }

    hj::Target targetOf(const std::string& id)
    {
        const auto target = hj::Target::fromTargetId(id);
        JIT_REQUIRE("Cannot parse " + id, target.has_value());
        return *target;
    }

    hj::Artifact build(const hj::SourceCodeBundle& source, hj::BuildKind kind, const fs::path& scratch)
    {
        hj::Artifact built;
        auto         status = hj::makeComgrBuilder()->build(source, kind, built);
        hj::keepBuildLog(status, scratch);
        JIT_REQUIRE("The build for " + source.target.targetId() + " failed: " + status.message,
                    status.ok());
        const auto expected = kind == hj::BuildKind::Executable ? hj::Artifact::Kind::Executable
                                                                : hj::Artifact::Kind::Relocatable;
        JIT_REQUIRE("The build did not return one artifact of the kind requested",
                    built.kind == expected && !built.bytes.empty());
        return built;
    }

    void buildAssembly(const fs::path&     file,
                       const Assembly&     assembly,
                       const hj::Artifact& kernel,
                       const hj::Artifact& relocatable,
                       const fs::path&     scratch)
    {
        const auto           text = hipblaslt_jit_test::readFile(file);
        const auto           name = hj::files::utf8String(file.filename());
        hj::SourceCodeBundle source;
        source.mainKernel  = {name,
                              hj::Artifact::Kind::Assembly,
                              std::vector<uint8_t>(text.begin(), text.end())};
        source.target      = assembly.target;
        source.kernelNames = {assembly.kernel};
        build(source, hj::BuildKind::Executable, scratch);

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
        JIT_REQUIRE("A relocatable build accepted a relocatable main kernel",
                    !again.ok() && again.stage == hj::Stage::Build);

        source.kernelNames = {assembly.kernel, kernelName};
        for(const auto& helper : {kernel, relocatable})
        {
            source.helpers = {helper};
            build(source, hj::BuildKind::Executable, scratch);
        }

        source.helpers     = {};
        source.kernelNames = {assembly.kernel, "hipblaslt_jit_builder_test_missing"};
        hj::Artifact built;
        const auto   missing = hj::makeComgrBuilder()->build(source, hj::BuildKind::Executable, built);
        JIT_REQUIRE("The build accepted a kernel name the executable does not define",
                    !missing.ok() && missing.stage == hj::Stage::Build
                        && missing.message.find("hipblaslt_jit_builder_test_missing")
                               != std::string::npos);
    }
}

TEST_CASE("target IDs parse into the typed target", "[jit-cpu]")
{
    const auto device = hj::Target::fromTargetId("gfx950:sramecc+:xnack-");
    JIT_REQUIRE("gfx950:sramecc+:xnack- did not parse into its processor and features",
                device && device->processor == hj::Processor::gfx950
                    && device->sramecc == hj::TargetFeature::On
                    && device->xnack == hj::TargetFeature::Off);
    JIT_REQUIRE("The target strings changed",
                device->processorName() == "gfx950"
                    && device->targetId() == "gfx950:sramecc+:xnack-"
                    && device->isaName() == "amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-");
    auto gfx1250 = hj::Target::fromTargetId("gfx1250");
    REQUIRE((gfx1250 && gfx1250->targetId() == "gfx1250"));
    for(const char* id : {"",
                          "gfx000",
                          "gfx9500",
                          "gfx1250-strict",
                          "gfx950:",
                          "gfx950:xnack",
                          "gfx950:xnack-:xnack+",
                          "gfx950:tgsplit+"})
    {
        JIT_REQUIRE(std::string("Parsed the unknown target ID '") + id + "'",
                    !hj::Target::fromTargetId(id));
    }
}

TEST_CASE("HIP files whose names differ by a hyphen and an underscore link", "[jit-cpu]")
{
    const auto assemblies = committedAssembly();
    JIT_REQUIRE("No committed assembly under the data directory", !assemblies.empty());
    const auto hip = [](const char* symbol) {
        const std::string text = std::string("#include <hip/hip_runtime.h>\n"
                                             "extern \"C\" __global__ void ")
                                 + symbol + "(float* x) { if(x) *x = 1.f; }\n";
        return std::vector<uint8_t>(text.begin(), text.end());
    };
    hj::SourceCodeBundle bundle;
    bundle.mainKernel = {"main-a.hip", hj::Artifact::Kind::Hip, hip("hipblaslt_jit_cuid_hyphen")};
    bundle.helpers
        = {{"main_a.hip", hj::Artifact::Kind::Hip, hip("hipblaslt_jit_cuid_underscore")}};
    bundle.target      = assemblies.front().second.target;
    bundle.kernelNames = {"hipblaslt_jit_cuid_hyphen", "hipblaslt_jit_cuid_underscore"};
    hj::Artifact built;
    const auto   status = hj::makeComgrBuilder()->build(bundle, hj::BuildKind::Executable, built);
    JIT_REQUIRE("main-a.hip and main_a.hip did not link: " + status.message,
                status.ok() && built.kind == hj::Artifact::Kind::Executable);
}

// Restores HIP_PATH. rocmPath reads it before it looks up the HIP runtime.
struct HipPathGuard
{
    std::string saved;
    bool        had = false;

    HipPathGuard()
    {
        if(const char* previous = std::getenv("HIP_PATH"))
        {
            saved = previous;
            had   = true;
        }
    }

    void set(const char* value) const
    {
#ifdef _WIN32
        _putenv_s("HIP_PATH", value);
#else
        setenv("HIP_PATH", value, 1);
#endif
    }

    ~HipPathGuard()
    {
        if(had)
            set(saved.c_str());
#ifndef _WIN32
        else
            unsetenv("HIP_PATH");
#else
        else
            _putenv_s("HIP_PATH", "");
#endif
    }
};

TEST_CASE("the ROCm prefix is the HIP runtime when HIP_PATH is unset", "[jit-cpu]")
{
    // unsetenv does not exist on Windows, and rocmPath does not discover the
    // runtime prefix there. HIP_PATH is the ROCm prefix when it is set covers
    // that platform.
#ifndef _WIN32
    HipPathGuard guard;
    unsetenv("HIP_PATH");
    const auto prefix = hj::code_object::rocmPath();
    const auto lib    = fs::path(prefix) / "lib";
    JIT_REQUIRE("ROCm prefix '" + prefix + "' is not the HIP runtime",
                !prefix.empty() && prefix.find("clients") == std::string::npos
                    && fs::is_directory(lib));
#endif
}

TEST_CASE("HIP_PATH is the ROCm prefix when it is set", "[jit-cpu]")
{
    HipPathGuard guard;
    guard.set("hip-prefix-sentinel");
    const auto prefix = hj::code_object::rocmPath();
    JIT_REQUIRE("HIP_PATH was ignored: '" + prefix + "'", prefix == "hip-prefix-sentinel");
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
    for(const auto& [file, assembly] : committedAssembly())
        byTarget[assembly.target.targetId()].push_back({file, assembly});
    int built = 0;
    for(const auto& [targetId, files] : byTarget)
    {
        hj::SourceCodeBundle hip;
        hip.mainKernel  = kernel;
        hip.target      = files.front().second.target;
        hip.kernelNames = {kernelName};
        build(hip, hj::BuildKind::Executable, scratch);
        const auto relocatable = build(hip, hj::BuildKind::Relocatable, scratch);
        for(const auto& [file, assembly] : files)
        {
            buildAssembly(file, assembly, kernel, relocatable, scratch);
            ++built;
        }
    }
    JIT_REQUIRE("No committed assembly under the data directory", built > 0);

    hj::SourceCodeBundle unknown;
    unknown.mainKernel = kernel;
    hj::Artifact out;
    const auto   noTarget = hj::makeComgrBuilder()->build(unknown, hj::BuildKind::Executable, out);
    JIT_REQUIRE("A bundle without a target processor was built",
                !noTarget.ok() && noTarget.stage == hj::Stage::Build);
    unknown.target  = byTarget.begin()->second.front().second.target;
    unknown.helpers = {kernel};
    const auto withHelpers
        = hj::makeComgrBuilder()->build(unknown, hj::BuildKind::Relocatable, out);
    JIT_REQUIRE("A relocatable build accepted helpers",
                !withHelpers.ok() && withHelpers.stage == hj::Stage::Build);
}

// The builder assembles a file only for the processor its .amdgcn_target names
// and for a device that has every feature it declares. A bare declaration
// builds for a device whose target ID has features.
TEST_CASE("comgr refuses assembly declared for another target", "[jit-cpu]")
{
    const auto scratch = fs::u8path(HIPBLASLT_JIT_SCRATCH) / "targets";
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    for(const auto& committed : committedAssembly())
    {
        const auto& file      = committed.first;
        const auto& assembly  = committed.second;
        const auto  text      = hipblaslt_jit_test::readFile(file);
        const auto  name      = hj::files::utf8String(file.filename());
        const auto  processor = assembly.target.processorName();
        const auto  declared  = "\"amdgcn-amd-amdhsa--" + assembly.target.targetId() + "\"";
        JIT_REQUIRE(name + " does not declare a bare target",
                    assembly.target.targetId() == processor
                        && text.find(declared) != std::string::npos);
        const auto declaring = [&](const std::string& targetId) {
            auto rewritten = text;
            rewritten.replace(rewritten.find(declared),
                              declared.size(),
                              "\"amdgcn-amd-amdhsa--" + targetId + "\"");
            return rewritten;
        };
        const auto refuses = [&](const std::string& source,
                                 const std::string& declaredId,
                                 const std::string& deviceId) {
            const auto   device = targetOf(deviceId);
            hj::Artifact out;
            const auto   status = hj::makeComgrBuilder()->build(
                assemblyBundle(name, source, assembly.kernel, device),
                hj::BuildKind::Executable,
                out);
            JIT_REQUIRE(
                name + " declaring " + declaredId + " was built for " + deviceId + ": "
                    + status.message,
                !status.ok() && status.stage == hj::Stage::Build
                    && status.message.find("targets " + declaredId + ", not " + device.targetId())
                           != std::string::npos);
        };

        const auto other = processor == "gfx950" ? "gfx942" : "gfx950";
        refuses(text, processor, std::string(other) + ":sramecc+:xnack-");
        refuses(declaring(processor + ":xnack+"),
                processor + ":xnack+",
                processor + ":sramecc+:xnack-");
        refuses(declaring(processor + ":sramecc-"),
                processor + ":sramecc-",
                processor + ":sramecc+:xnack-");

        hj::Artifact built;
        auto         status = hj::makeComgrBuilder()->build(
            assemblyBundle(name, text, assembly.kernel, targetOf(processor + ":sramecc+:xnack-")),
            hj::BuildKind::Executable,
            built);
        hj::keepBuildLog(status, scratch);
        JIT_REQUIRE(name + " was not built for " + processor
                        + ":sramecc+:xnack-: " + status.message,
                    status.ok() && built.kind == hj::Artifact::Kind::Executable);
    }
}
