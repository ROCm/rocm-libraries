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
    {
        INFO(("Cannot query the current HIP device"));
        REQUIRE((hipGetDevice(&device) == hipSuccess
            && hipGetDeviceProperties(&properties, device) == hipSuccess));
    }
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
            {
                INFO(("lookup K=" + std::to_string(publication.k) + ": " + status.message));
                REQUIRE((status.ok()));
            }
            {
                INFO((item.name + " K=" + std::to_string(publication.k)
                    + " is not one JIT solution index"));
                REQUIRE((indices.size() == 1 && hj::isJitIndex(indices[0])));
            }
            hj::Status why;
            const auto solution = library.solutionByIndex(device, *hardware, indices[0], why);
            {
                INFO(("Cannot load solution: " + why.message));
                REQUIRE((solution != nullptr));
            }
            {
                INFO(("Found kernel " + solution->kernelName + ", expected " + publication.kernel));
                REQUIRE((solution->kernelName == publication.kernel));
            }
            {
                INFO(("Found solution " + solution->solutionName + ", expected "
                    + publication.solutionName));
                REQUIRE((solution->solutionName == publication.solutionName));
            }
            {
                INFO(("The loaded solution has another index"));
                REQUIRE((solution->index == indices[0]));
            }
            if(!loaded)
            {
                const auto view = library.resolve(device, indices[0], why);
                {
                    INFO(("resolve: " + why.message));
                    REQUIRE((view.master && view.adapter));
                }
                const auto object = library.directory(key)
                                    / fs::u8path(std::string(solution->codeObjectFilename.load()));
                {
                    INFO(("Cannot load " + object.u8string()));
                    REQUIRE((view.adapter->loadCodeObjectFile(object.string()) == hipSuccess));
                }
                {
                    INFO(("Cannot resolve " + solution->kernelName));
                    REQUIRE((view.adapter->initKernel(solution->kernelName) == hipSuccess));
                }
                loaded = true;
            }
        }
    }
    {
        INFO(("This device has no committed plain kernel"));
        REQUIRE((plain && pair && loaded));
    }

    std::vector<int32_t> transposed;
    const auto           status = library.lookup(
        key, device, hipblaslt_jit_test::fp16Gemm(true, 512), *hardware, 4, {}, transposed);
    {
        INFO(("A transposed A found a published solution"));
        REQUIRE((status.ok() && transposed.empty()));
    }

    hj::BuiltSolution damaged;
    damaged.generated            = pair->solution;
    damaged.generated.entry      = skipIndexOne(std::move(damaged.generated.entry));
    hipblaslt_jit_test::reject([&] { hj::parseTensileBundle(damaged, hardware); },
                               "The loader accepted solutions 0 and 2");
}
