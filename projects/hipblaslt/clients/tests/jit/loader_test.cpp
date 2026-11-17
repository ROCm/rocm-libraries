// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-fs.hpp"
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <map>
#include <string>
#include <vector>

// Loads the JIT solution library written for this device and finds the
// pre-generated kernel's solutions, including both solutions of one published
// executable in one entry, and loads each code object once. Launches nothing.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;
}

TEST_CASE("the loader reads the JIT solution library", "[jit-gpu]")
{
    const auto     target   = hipblaslt_jit_test::currentDeviceTarget();
    const auto     device   = target.device;
    const auto&    hardware = *target.hardware;
    const auto     key      = hipblaslt_jit_test::pregeneratedKey(target.properties);
    hj::JitLibrary library(fs::u8path(HIPBLASLT_JIT_LIBRARY));

    const auto prepared
        = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA), target);
    const hipblaslt_jit_test::Prepared* pair = nullptr;
    bool                                paired = false;
    for(const auto& item : prepared)
    {
        if(item.name == "plain-pair")
            pair = &item;
        for(const auto k : item.ks)
        {
            const auto problem  = hipblaslt_jit_test::fp16Gemm(false, k);
            const auto expected = hj::supportingSolutions(
                item.source.solutions, problem, hardware, hipblaslt_jit_test::workspaceLimit);
            const auto           label = item.name + " K=" + std::to_string(k);
            std::vector<int32_t> indices;
            const auto status = library.lookup(key, device, problem, hardware, 4, {}, indices);
            JIT_REQUIRE("lookup " + label + ": " + status.message, status.ok());
            JIT_REQUIRE(label + " did not find one JIT solution index per published solution",
                        !expected.empty() && indices.size() == expected.size());
            for(size_t i = 0; i < indices.size(); ++i)
            {
                hj::Status why;
                const auto solution = library.solutionByIndex(device, hardware, indices[i], why);
                JIT_REQUIRE("Cannot load solution: " + why.message,
                            TensileLite::isJitKernelIndex(indices[i]) && solution != nullptr);
                JIT_REQUIRE("Found solution " + solution->solutionName + ", expected "
                                + expected[i]->solutionName,
                            solution->kernelName == expected[i]->kernelName
                                && solution->solutionName == expected[i]->solutionName);
                JIT_REQUIRE("The loaded solution has another index", solution->index == indices[i]);
            }
            if(indices.size() < 2)
                continue;

            // The solutions published from one executable share its entry.
            hj::Status why;
            const auto view = library.resolve(device, indices.front(), why);
            JIT_REQUIRE("resolve: " + why.message, view.master != nullptr);
            const auto& mapping = view.master->libraryMapping;
            const auto  prefix  = mapping.at(indices.front());
            for(const auto index : indices)
            {
                JIT_REQUIRE(label + " published its solutions into more than one entry",
                            mapping.count(index) && mapping.at(index) == prefix);
            }
            const auto file = library.directory(key) / "slices" / hj::sliceKey(problem)
                              / fs::u8path(prefix + ".dat");
            const auto entry = std::dynamic_pointer_cast<hj::GemmMaster>(
                TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(file.string()));
            {
                INFO(("The entry " + hj::files::utf8String(file)
                      + " does not hold every solution of " + label));
                REQUIRE((entry && entry->solutions.size() == indices.size()));
                for(const auto index : indices)
                    REQUIRE((entry->solutions.count(index)));
            }
            paired = true;
        }
    }
    JIT_REQUIRE("plain-pair was not published as one entry", pair && paired);

    std::vector<int32_t> transposed;
    const auto           status = library.lookup(
        key, device, hipblaslt_jit_test::fp16Gemm(true, 512), hardware, 4, {}, transposed);
    JIT_REQUIRE("A transposed A found a published solution", status.ok() && transposed.empty());
}

// plain-pair's two solutions at K=1024 share one code object. In a copy of the
// published library, resolve fails with the path while the code objects are
// truncated and loads them once they are restored. Truncated again, they are
// not read when either solution is resolved again.
TEST_CASE("resolve loads each code object once", "[jit-gpu]")
{
    const auto     target   = hipblaslt_jit_test::currentDeviceTarget();
    const auto     device   = target.device;
    const auto&    hardware = *target.hardware;
    const auto     key      = hipblaslt_jit_test::pregeneratedKey(target.properties);
    const auto     problem  = hipblaslt_jit_test::fp16Gemm(false, 1024);
    const fs::path scratch  = fs::u8path(HIPBLASLT_JIT_SCRATCH);
    const fs::path root     = scratch / "library";
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    fs::copy(fs::u8path(HIPBLASLT_JIT_LIBRARY), root, fs::copy_options::recursive);

    hj::JitLibrary       library(root);
    std::vector<int32_t> indices;
    const auto           status = library.lookup(key, device, problem, hardware, 4, {}, indices);
    JIT_REQUIRE("lookup: " + status.message, status.ok() && indices.size() == 2);

    const auto slice = library.directory(key) / "slices" / hj::sliceKey(problem);
    std::map<fs::path, std::string> codeObjects;
    for(const auto& file : fs::recursive_directory_iterator(slice))
        if(file.path().extension() == ".co")
            codeObjects[file.path()] = hipblaslt_jit_test::readFile(file.path());
    JIT_REQUIRE("No code object under " + hj::files::utf8String(slice), !codeObjects.empty());
    for(const auto& [path, bytes] : codeObjects)
        hipblaslt_jit_test::writeFile(path, bytes.substr(0, bytes.size() / 2));

    hj::Status why;
    JIT_REQUIRE("resolve accepted a truncated code object",
                library.resolve(device, indices[0], why).master == nullptr);
    JIT_REQUIRE("The failure does not name the code object: " + why.message,
                why.message.find("Cannot load JIT code object " + slice.string()) == 0
                    && why.message.find(".co: ") != std::string::npos);

    hj::BackendInfo backend;
    backend.id      = "pregenerated";
    backend.version = 1;
    library.beginQuery(target, backend, hipblaslt_jit_test::workspaceLimit, {});
    const auto found = library.findTopSolutions(problem, hardware, 2);
    const auto query = library.queryStatus();
    library.endQuery();
    JIT_REQUIRE("findTopSolutions treated a truncated code object as no match: " + query.message,
                found.empty() && !query.ok()
                    && query.message.find("Cannot load JIT code object ") == 0
                    && query.message.find(".co: ") != std::string::npos);

    // A fresh query, so a failure left by findTopSolutions cannot satisfy this.
    library.beginQuery(target, backend, hipblaslt_jit_test::workspaceLimit, {});
    JIT_REQUIRE("the fresh query inherited the earlier failure", library.queryStatus().ok());
    const auto byIndex
        = library.getSolutionByIndex(problem, hardware, static_cast<int>(indices[0]));
    const auto afterIndex = library.queryStatus();
    library.endQuery();
    JIT_REQUIRE("getSolutionByIndex left a truncated code object successful: " + afterIndex.message,
                byIndex == nullptr && !afterIndex.ok()
                    && afterIndex.message.find(".co: ") != std::string::npos);

    for(const auto& [path, bytes] : codeObjects)
        hipblaslt_jit_test::writeFile(path, bytes);
    const auto first = library.resolve(device, indices[0], why);
    JIT_REQUIRE("resolve after the code object was restored: " + why.message,
                first.master != nullptr && first.adapter != nullptr);
    const auto loadedName = hj::files::utf8String(
        fs::u8path(first.master->libraryMapping.at(indices[0]) + ".co").filename());
    JIT_REQUIRE("The adapter did not load " + loadedName,
                first.adapter->FindCodeObject(loadedName));
    for(const auto& [path, bytes] : codeObjects)
        hipblaslt_jit_test::writeFile(path, bytes.substr(0, bytes.size() / 2));
    for(const auto index : {indices[1], indices[0]})
    {
        const auto again = library.resolve(device, index, why);
        JIT_REQUIRE("Resolving index " + std::to_string(index)
                        + " again read its code object: " + why.message,
                    again.master != nullptr && again.adapter == first.adapter);
    }
}

// A published entry whose .dat is not a library makes lookup fail and name
// that file. The load used to dereference the null library and terminate.
TEST_CASE("lookup reports a damaged solution entry", "[jit-gpu]")
{
    const auto     target   = hipblaslt_jit_test::currentDeviceTarget();
    const auto     device   = target.device;
    const auto&    hardware = *target.hardware;
    const auto     key      = hipblaslt_jit_test::pregeneratedKey(target.properties);
    const auto     problem  = hipblaslt_jit_test::fp16Gemm(false, 1024);
    const fs::path scratch  = fs::u8path(HIPBLASLT_JIT_SCRATCH);
    const fs::path root     = scratch / "library";
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    fs::copy(fs::u8path(HIPBLASLT_JIT_LIBRARY), root, fs::copy_options::recursive);

    hj::JitLibrary       library(root);
    std::vector<int32_t> indices;
    const auto           status = library.lookup(key, device, problem, hardware, 4, {}, indices);
    JIT_REQUIRE("lookup: " + status.message, status.ok() && indices.size() == 2);

    hj::Status why;
    const auto view = library.resolve(device, indices.front(), why);
    JIT_REQUIRE("resolve: " + why.message, view.master != nullptr);
    const auto prefix = view.master->libraryMapping.at(indices.front());
    const auto entry
        = library.directory(key) / "slices" / hj::sliceKey(problem) / fs::u8path(prefix + ".dat");
    JIT_REQUIRE("No entry " + hj::files::utf8String(entry), fs::is_regular_file(entry));

    const auto staging = library.directory(key) / "staging";
    fs::create_directories(staging);
    const auto replaced
        = hj::files::writeAtomically(staging, entry, std::vector<uint8_t>{0x01, 0x02, 0x03});
    JIT_REQUIRE("Cannot replace " + hj::files::utf8String(entry) + ": " + replaced.message,
                replaced.ok());

    hj::JitLibrary       fresh(root);
    std::vector<int32_t> damaged;
    const auto           failed = fresh.lookup(key, device, problem, hardware, 4, {}, damaged);
    JIT_REQUIRE("A damaged entry was accepted: " + failed.message,
                !failed.ok() && damaged.empty()
                    && failed.message.find(hj::files::utf8String(entry.filename()))
                           != std::string::npos);
}
