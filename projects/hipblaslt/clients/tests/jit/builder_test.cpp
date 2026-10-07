// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-code-object.hpp"
#include "solution_entry.hpp"
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Builds the hand-written HIP kernel with comgr for each committed assembly
// target, then that assembly linked with the kernel, and checks that the code
// object defines every expected kernel. Needs no GPU.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    const char* const kernelName = "hipblaslt_jit_builder_test_scale";

    void build(hj::GeneratedSolution& solution,
               const std::string&     targetId,
               const fs::path&        scratch,
               hj::BuiltSolution&     built)
    {
        const hj::BuildRequest request{targetId, hj::jitCodeObjectVersion, scratch};
        const auto             status = hj::makeComgrBuilder()->build(solution, request, built);
        {
            INFO(("The build for " + targetId + " failed: " + status.message));
            REQUIRE((status.ok()));
        }

        const auto metadata
            = hj::code_object::readMetadata(built.object.bytes.data(), built.object.bytes.size());
        {
            INFO(("Cannot read the built code object: " + metadata.log));
            REQUIRE((metadata.ok()));
        }
        const auto& kernels  = metadata.metadata.kernelNames;
        auto        expected = solution.kernelNames;
        expected.push_back(kernelName);
        for(const auto& name : expected)
        {
            INFO(("The code object does not define " + name));
            REQUIRE((std::find(kernels.begin(), kernels.end(), name) != kernels.end()));
        }
        {
            INFO(("The code object is not version " + std::to_string(hj::jitCodeObjectVersion)));
            REQUIRE((metadata.metadata.codeObjectVersion == hj::jitCodeObjectVersion));
        }
    }

    void buildKernel(const std::string&   targetId,
                     const hj::BuildUnit& kernel,
                     const fs::path&      scratch)
    {
        hj::GeneratedSolution solution;
        solution.kernelNames = {kernelName};
        solution.units       = {kernel};
        hj::BuiltSolution built;
        build(solution, targetId, scratch, built);
        std::cout << "PASS comgr built the HIP kernel for " << targetId << '\n';
    }

    void buildAssembly(const fs::path&        file,
                       const hj::BuildUnit&   kernel,
                       const fs::path&        scratch,
                       const std::string&     kernelNameFromSource,
                       const std::string&     targetId)
    {
        const auto text = hipblaslt_jit_test::readFile(file);
        hj::GeneratedSolution solution;
        solution.kernelNames = {kernelNameFromSource};
        solution.units.push_back({file.filename().u8string(),
                                  std::vector<uint8_t>(text.begin(), text.end()),
                                  hj::BuildUnit::Kind::Assembly,
                                  {}});
        solution.units.push_back(kernel);
        hj::BuiltSolution built;
        build(solution, targetId, scratch, built);
        std::cout << "PASS comgr built " << file.filename().u8string() << " and the HIP kernel for "
                  << targetId << '\n';

        solution.kernelNames.push_back("hipblaslt_jit_builder_test_missing");
        const hj::BuildRequest request{targetId, hj::jitCodeObjectVersion, scratch};
        const auto             missing = hj::makeComgrBuilder()->build(solution, request, built);
        {
            INFO(("The build accepted a kernel name the code object does not define"));
            REQUIRE((!missing.ok() && missing.stage == hj::Stage::Build
                && missing.message.find("hipblaslt_jit_builder_test_missing")
                       != std::string::npos));
        }
    }
}

TEST_CASE("comgr builds the HIP kernel and every committed assembly file", "[jit-cpu]")
{
    const auto source  = hipblaslt_jit_test::readFile(fs::u8path(HIPBLASLT_JIT_KERNEL));
    const auto scratch = fs::u8path(HIPBLASLT_JIT_SCRATCH);
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    const hj::BuildUnit kernel{"builder_test_kernel.hip",
                               std::vector<uint8_t>(source.begin(), source.end()),
                               hj::BuildUnit::Kind::Hip,
                               {}};

    std::map<std::string, std::vector<std::pair<fs::path, std::string>>> byTarget;
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
                const auto assembly
                    = hipblaslt_jit_test::readAssembly(hipblaslt_jit_test::readFile(file.path()));
                byTarget[assembly.target].push_back({file.path(), assembly.kernel});
            }
        }
    }
    int built = 0;
    for(const auto& [targetId, files] : byTarget)
    {
        buildKernel(targetId, kernel, scratch);
        for(const auto& [file, kernelFromSource] : files)
        {
            buildAssembly(file, kernel, scratch, kernelFromSource, targetId);
            ++built;
        }
    }
    {
        INFO(("No committed assembly under the data directory"));
        REQUIRE((built > 0));
    }
    std::cout << "PASS the build of every assembly file fails when a kernel name is not defined\n";
}
