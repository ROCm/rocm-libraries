// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Checks that the committed JIT test bundles still match this tree: each
// manifest's kernel-argument and persistent-loop argument layout versions equal
// those in GlobalParameters.py, its code-object version equals the builder's,
// the host library reads its library entry, and comgr builds its sources into a
// code object that defines its main kernel. Needs no GPU.
//
// usage: hipblaslt-jit-bundle-freshness-test DATA_DIR SCRATCH_DIR

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    namespace fs = std::filesystem;
    using namespace hipblaslt_jit;
    using Master = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

    struct ProbeRequest final : OperationRequest
    {
        std::string_view kind() const noexcept override
        {
            return "test.bundle-freshness.v1";
        }
    };

    std::string text(const fs::path& path)
    {
        const auto bytes = source_bundle::readArtifact(path);
        return {bytes.begin(), bytes.end()};
    }

    // The value of key in the manifest's object named owner, without quotes;
    // empty when absent.
    std::string field(const std::string& manifest, const std::string& owner, const std::string& key)
    {
        const auto object = manifest.find("\"" + owner + "\": {");
        if(object == std::string::npos)
            return {};
        const auto  end = manifest.find('}', object);
        std::smatch match;
        const std::regex pattern("\"" + key + "\": *\"?([^\",}\\s]*)");
        const auto       body = manifest.substr(object, end - object);
        return std::regex_search(body, match, pattern) ? match[1].str() : std::string();
    }

    // What makes the bundle stale, or nothing.
    std::vector<std::string> check(const fs::path& bundle, const fs::path& scratch)
    {
        std::vector<std::string> problems;
        try
        {
            const auto manifest = text(source_bundle::artifact(bundle, "manifest.json"));
            const std::pair<const char*, std::string> versions[] = {
                {"kernargs_version", std::to_string(HIPBLASLT_JIT_KERNARGS_VERSION)},
                {"persistent_loop_args_version",
                 std::to_string(HIPBLASLT_JIT_PERSISTENT_LOOP_ARGS_VERSION)},
                {"code_object_version", std::to_string(jitCodeObjectVersion)},
            };
            for(const auto& [key, expected] : versions)
            {
                const auto recorded = field(manifest, "provenance", key);
                if(recorded != expected)
                    problems.push_back(std::string(key) + " is "
                                       + (recorded.empty() ? "missing" : recorded) + ", not "
                                       + expected);
            }

            auto              sources = source_bundle::readSourceBundle(bundle);
            GeneratedSolution solution;
            solution.entry     = std::move(sources.library);
            const auto library = std::dynamic_pointer_cast<Master>(
                TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(solution.entry));
            if(!library || library->solutions.size() != 1 || !library->solutions.count(0))
                throw std::runtime_error("the library entry is not one local solution");
            solution.kernelName = library->solutions.at(0)->kernelName;
            if(solution.kernelName != field(manifest, "main_kernel", "name"))
                problems.push_back("the library entry's kernel is not the manifest's main kernel");
            for(auto& file : sources.assembly)
                solution.units.push_back({BuildUnit::Role::Main,
                                          std::move(file.name),
                                          std::move(file.bytes),
                                          BuildUnit::Kind::Assembly,
                                          {}});
            std::vector<IncludeFile> includes;
            for(auto& header : sources.headers)
                includes.push_back({std::move(header.name), std::move(header.bytes)});
            for(auto& file : sources.helpers)
                solution.units.push_back({BuildUnit::Role::Helper,
                                          std::move(file.name),
                                          std::move(file.bytes),
                                          BuildUnit::Kind::Hip,
                                          includes});

            DeviceTarget target;
            target.targetId = field(manifest, "architecture", "compiler_target");
            target.isa      = target.targetId.substr(0, target.targetId.find(':'));
            fs::create_directories(scratch);
            const ProbeRequest      probe;
            const GenerationRequest request{probe, target, nullptr, 1, 0, {}, scratch};
            BuiltSolution           built;
            const auto status = makeComgrBuilder()->build(solution, request, built);
            if(!status.ok())
                problems.push_back("the build for " + target.targetId + " failed: "
                                   + status.message);
        }
        catch(const std::exception& error)
        {
            problems.push_back(error.what());
        }
        return problems;
    }

    // Copies bundle and adds one to the provenance value of key.
    fs::path doctor(const fs::path& bundle, const fs::path& copy, const std::string& key)
    {
        fs::remove_all(copy);
        fs::create_directories(copy.parent_path());
        fs::copy(bundle, copy, fs::copy_options::recursive);
        auto        manifest = text(copy / "manifest.json");
        std::smatch match;
        const std::regex pattern("(\"" + key + "\": *)([0-9]+)");
        if(!std::regex_search(manifest, match, pattern))
            throw std::runtime_error("No " + key + " to doctor in " + bundle.u8string());
        const auto value = std::to_string(std::stoi(match[2].str()) + 1);
        manifest = match.prefix().str() + match[1].str() + value + match.suffix().str();
        std::ofstream(copy / "manifest.json", std::ios::binary | std::ios::trunc) << manifest;
        return copy;
    }
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "usage: " << argv[0] << " DATA_DIR SCRATCH_DIR\n";
        return 2;
    }
    try
    {
        const fs::path data = fs::u8path(argv[1]), scratch = fs::u8path(argv[2]);
        fs::create_directories(scratch);
        std::vector<fs::path> bundles;
        for(const auto& architecture : fs::directory_iterator(data))
            if(architecture.is_directory())
                for(const auto& bundle : fs::directory_iterator(architecture.path()))
                    if(bundle.is_directory())
                        bundles.push_back(bundle.path());
        std::sort(bundles.begin(), bundles.end());
        if(bundles.empty())
            throw std::runtime_error("No bundles under " + data.u8string());

        size_t stale = 0;
        for(const auto& bundle : bundles)
        {
            const auto name
                = bundle.parent_path().filename().u8string() + "/" + bundle.filename().u8string();
            const auto problems = check(bundle, scratch / "build" / name);
            stale += !problems.empty();
            for(const auto& problem : problems)
                std::cout << "STALE " << name << ": " << problem << '\n';
            if(problems.empty())
                std::cout << "PASS " << name << " is current\n";
        }

        for(const char* key : {"kernargs_version", "persistent_loop_args_version"})
        {
            const auto copy     = doctor(bundles.front(), scratch / "doctored" / key, key);
            const auto problems = check(copy, scratch / "build-doctored" / key);
            if(problems.size() != 1 || problems[0].find(key) != 0)
                throw std::runtime_error(std::string("A bundle with another ") + key
                                         + " was not reported stale");
        }
        std::cout << "PASS a manifest with another kernargs_version or "
                     "persistent_loop_args_version is reported stale\n";

        if(stale)
        {
            std::cout << "FAIL: " << stale << " of " << bundles.size()
                      << " committed JIT test bundles are stale and must be regenerated; see "
                         "clients/tests/jit/data/README.md\n";
            return 1;
        }
        std::cout << "PASS: all " << bundles.size() << " committed JIT test bundles are current\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
