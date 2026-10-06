// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

// Builds the hand-written HIP kernel with comgr for each written bundle's
// target, then every written bundle with that kernel linked into its code
// object, and checks that the code object defines every expected kernel. Needs
// no GPU.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;
    using hipblaslt_jit_test::require;

    const char* const kernelName = "hipblaslt_jit_builder_test_scale";

    // The string value of key in the manifest's object named owner.
    std::string field(const std::string& manifest, const std::string& owner, const std::string& key)
    {
        std::smatch      match;
        const std::regex pattern("\"" + owner + "\": *\\{[^}]*\"" + key + "\": *\"([^\"]*)\"");
        require(std::regex_search(manifest, match, pattern), "The manifest has no " + owner + "." + key);
        return match[1].str();
    }

    // The strings of the manifest's array named key.
    std::vector<std::string> strings(const std::string& manifest, const std::string& key)
    {
        std::smatch match;
        require(std::regex_search(manifest, match, std::regex("\"" + key + "\": *\\[([^\\]]*)\\]")),
                "The manifest has no " + key);
        const auto               list = match[1].str();
        std::vector<std::string> result;
        const std::regex         item("\"([^\"]*)\"");
        for(auto at = std::sregex_iterator(list.begin(), list.end(), item);
            at != std::sregex_iterator();
            ++at)
            result.push_back((*at)[1].str());
        require(!result.empty(), "The manifest's " + key + " is empty");
        return result;
    }

    // Builds solution for targetId and checks that the code object defines its
    // kernels and kernelName.
    void build(hj::GeneratedSolution& solution,
               const std::string&     targetId,
               const fs::path&        scratch,
               hj::BuiltSolution&     built)
    {
        const hj::BuildRequest request{targetId, hj::jitCodeObjectVersion, scratch};
        const auto             status = hj::makeComgrBuilder()->build(solution, request, built);
        require(status.ok(), "The build for " + targetId + " failed: " + status.message);

        const auto metadata
            = hj::code_object::readMetadata(built.object.bytes.data(), built.object.bytes.size());
        require(metadata.ok(), "Cannot read the built code object: " + metadata.log);
        const auto& kernels  = metadata.metadata.kernelNames;
        auto        expected = solution.kernelNames;
        expected.push_back(kernelName);
        for(const auto& name : expected)
            require(std::find(kernels.begin(), kernels.end(), name) != kernels.end(),
                    "The code object does not define " + name);
        require(metadata.metadata.codeObjectVersion == hj::jitCodeObjectVersion,
                "The code object is not version " + std::to_string(hj::jitCodeObjectVersion));
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

    void buildBundle(const fs::path& bundle, const hj::BuildUnit& kernel, const fs::path& scratch)
    {
        const auto bytes    = hj::source_bundle::readArtifact(bundle / "manifest.json");
        const auto manifest = std::string(bytes.begin(), bytes.end());
        auto       sources  = hj::source_bundle::readSourceBundle(bundle);
        require(sources.hip.empty(), "Expected only assembly in " + bundle.u8string());

        hj::GeneratedSolution solution;
        solution.entry       = std::move(sources.library);
        solution.kernelNames = strings(manifest, "main_kernels");
        for(auto& file : sources.assembly)
            solution.units.push_back(
                {std::move(file.name), std::move(file.bytes), hj::BuildUnit::Kind::Assembly, {}});
        solution.units.push_back(kernel);

        const auto        targetId = field(manifest, "architecture", "compiler_target");
        hj::BuiltSolution built;
        build(solution, targetId, scratch, built);
        std::cout << "PASS comgr built " << bundle.filename().u8string() << ", "
                  << solution.kernelNames.size() << " main kernel(s) and the HIP kernel, for "
                  << targetId << '\n';

        solution.kernelNames.push_back("hipblaslt_jit_builder_test_missing");
        const hj::BuildRequest request{targetId, hj::jitCodeObjectVersion, scratch};
        const auto             missing = hj::makeComgrBuilder()->build(solution, request, built);
        require(!missing.ok() && missing.stage == hj::Stage::Build
                    && missing.message.find("hipblaslt_jit_builder_test_missing")
                           != std::string::npos,
                "The build accepted a kernel name the code object does not define");
    }
}

int main(int argc, char** argv)
try
{
    require(argc == 4, "Usage: hipblaslt-jit-builder-test BUNDLES KERNEL SCRATCH");
    const auto bundles = fs::u8path(argv[1]);
    const auto source  = hipblaslt_jit_test::readFile(fs::u8path(argv[2]));
    const auto scratch = fs::u8path(argv[3]);
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    const hj::BuildUnit kernel{"builder_test_kernel.hip",
                               std::vector<uint8_t>(source.begin(), source.end()),
                               hj::BuildUnit::Kind::Hip,
                               {}};
    int built = 0;
    for(const auto& target : fs::directory_iterator(bundles))
    {
        buildKernel(target.path().filename().u8string(), kernel, scratch);
        for(const auto& bundle : fs::directory_iterator(target))
        {
            buildBundle(bundle.path(), kernel, scratch);
            ++built;
        }
    }
    require(built > 0, "No bundle in " + bundles.u8string());
    std::cout << "PASS the build of every bundle fails when a kernel name is not defined\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
