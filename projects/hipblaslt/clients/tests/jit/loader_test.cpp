// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-loader.hpp"
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

// Loads the JIT solution library written for this device and finds the
// pre-generated kernel. Also checks that an entry whose solutions are not
// 0 to N-1 is rejected. Launches nothing.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;
    using hipblaslt_jit_test::require;

    std::vector<uint8_t> skipIndexOne(std::vector<uint8_t> entry)
    {
        const std::vector<uint8_t> from{0xa5, 'i', 'n', 'd', 'e', 'x', 1};
        auto                       at = entry.begin();
        while((at = std::search(at, entry.end(), from.begin(), from.end())) != entry.end())
            *(at += from.size() - 1) = 2;
        return entry;
    }
}

TEST_CASE("the loader reads the JIT solution library", "[jit-gpu]")
{
    int             device = 0;
    hipDeviceProp_t properties{};
    require(hipGetDevice(&device) == hipSuccess
                && hipGetDeviceProperties(&properties, device) == hipSuccess,
            "Cannot query the current HIP device");
    const std::string target(properties.gcnArchName);
    const auto        arch     = target.substr(0, target.find(':'));
    const auto        hardware = TensileLite::hip::GetDevice(properties, device);
    const auto        key      = hipblaslt_jit_test::pregeneratedKey(properties);
    hj::JitLibrary    library(fs::u8path(HIPBLASLT_JIT_LIBRARY));

    const auto prepared = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA));
    const hipblaslt_jit_test::Prepared* plain = nullptr;
    const hipblaslt_jit_test::Prepared* pair  = nullptr;
    bool                                loaded = false;
    for(const auto& item : prepared)
    {
        if(item.arch != arch)
            continue;
        if(item.name == "plain")
            plain = &item;
        if(item.name == "plain-pair")
            pair = &item;
        for(const auto& publication : item.publications)
        {
            std::vector<int32_t> indices;
            const auto           problem = hipblaslt_jit_test::fp16Gemm(false, publication.k);
            const auto           status
                = library.lookup(key, device, problem, *hardware, 4, {}, indices);
            require(status.ok(), "lookup K=" + std::to_string(publication.k) + ": " + status.message);
            require(indices.size() == 1 && hj::isJitIndex(indices[0]),
                    item.name + " K=" + std::to_string(publication.k)
                        + " is not one JIT solution index");
            hj::Status why;
            const auto solution = library.solutionByIndex(device, *hardware, indices[0], why);
            require(solution != nullptr, "Cannot load solution: " + why.message);
            require(solution->kernelName == publication.kernel,
                    "Found kernel " + solution->kernelName + ", expected " + publication.kernel);
            require(solution->solutionName == publication.solutionName,
                    "Found solution " + solution->solutionName + ", expected "
                        + publication.solutionName);
            require(solution->index == indices[0], "The loaded solution has another index");
            if(!loaded)
            {
                const auto view = library.resolve(device, indices[0], why);
                require(view.master && view.adapter, "resolve: " + why.message);
                const auto object = library.directory(key)
                                    / fs::u8path(std::string(solution->codeObjectFilename.load()));
                require(view.adapter->loadCodeObjectFile(object.string()) == hipSuccess,
                        "Cannot load " + object.u8string());
                require(view.adapter->initKernel(solution->kernelName) == hipSuccess,
                        "Cannot resolve " + solution->kernelName);
                loaded = true;
            }
        }
    }
    require(plain && pair && loaded, "This device has no committed plain kernel");

    std::vector<int32_t> transposed;
    const auto           status = library.lookup(
        key, device, hipblaslt_jit_test::fp16Gemm(true, 512), *hardware, 4, {}, transposed);
    require(status.ok() && transposed.empty(), "A transposed A found a published solution");

    hj::BuiltSolution damaged;
    damaged.generated            = pair->solution;
    damaged.generated.entry      = skipIndexOne(std::move(damaged.generated.entry));
    hipblaslt_jit_test::reject([&] { hj::parseTensileBundle(damaged, hardware); },
                               "The loader accepted solutions 0 and 2");
}
