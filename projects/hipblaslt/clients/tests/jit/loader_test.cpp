// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-loader.hpp"
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/msgpack/Writing.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

// Loads the JIT solution library written for this device and finds the
// pre-generated kernel's solutions, including both solutions of one published
// executable in one entry. Also checks that a library whose solutions are not
// 0 to N-1 is rejected. Launches nothing.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    // A library of copies of solutions under indices, written by TensileLite.
    std::vector<uint8_t> numbered(const std::vector<hipblaslt_jit_test::Solution>& solutions,
                                  const std::vector<int>&                         indices)
    {
        using Gemm = TensileLite::ContractionProblemGemm;
        using Rows = TensileLite::ProblemSelectionLibrary<Gemm, TensileLite::ContractionSolution>;
        auto rows  = std::make_shared<Rows>();
        hj::GemmMaster library;
        for(size_t i = 0; i < solutions.size(); ++i)
        {
            auto copy   = hj::copySolution(solutions[i]);
            copy->index = indices[i];
            library.solutions.emplace(copy->index, copy);
            rows->rows.emplace_back(
                copy->problemPredicate,
                std::make_shared<
                    TensileLite::SingleSolutionLibrary<Gemm, TensileLite::ContractionSolution>>(
                    copy));
        }
        library.library = rows;
        return TensileLite::MessagePackWriteLibraryData(library);
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
    const auto     target   = hipblaslt_jit_test::deviceTarget(properties);
    const auto     hardware = TensileLite::hip::GetDevice(properties, device);
    const auto     key      = hipblaslt_jit_test::pregeneratedKey(properties);
    hj::JitLibrary library(fs::u8path(HIPBLASLT_JIT_LIBRARY));

    const auto prepared
        = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA), target);
    const hipblaslt_jit_test::Prepared* pair = nullptr;
    bool                                loaded = false, paired = false;
    for(const auto& item : prepared)
    {
        if(item.name == "plain-pair")
            pair = &item;
        for(const auto k : item.ks)
        {
            const auto problem = hipblaslt_jit_test::fp16Gemm(false, k);
            const auto expected
                = hipblaslt_jit_test::supported(item.assembly.solutions, problem, *hardware);
            const auto           label = item.name + " K=" + std::to_string(k);
            std::vector<int32_t> indices;
            const auto status = library.lookup(key, device, problem, *hardware, 4, {}, indices);
            {
                INFO(("lookup " + label + ": " + status.message));
                REQUIRE((status.ok()));
            }
            {
                INFO((label + " did not find one JIT solution index per published solution"));
                REQUIRE((!expected.empty() && indices.size() == expected.size()));
            }
            for(size_t i = 0; i < indices.size(); ++i)
            {
                hj::Status why;
                const auto solution = library.solutionByIndex(device, *hardware, indices[i], why);
                {
                    INFO(("Cannot load solution: " + why.message));
                    REQUIRE((hj::isJitIndex(indices[i]) && solution != nullptr));
                }
                {
                    INFO(("Found solution " + solution->solutionName + ", expected "
                          + expected[i]->solutionName));
                    REQUIRE((solution->kernelName == expected[i]->kernelName
                             && solution->solutionName == expected[i]->solutionName));
                }
                {
                    INFO(("The loaded solution has another index"));
                    REQUIRE((solution->index == indices[i]));
                }
                if(loaded)
                    continue;
                const auto view = library.resolve(device, indices[i], why);
                {
                    INFO(("resolve: " + why.message));
                    REQUIRE((view.master && view.adapter));
                }
                const auto object = library.directory(key) / "slices" / hj::sliceKey(problem)
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
            if(indices.size() < 2)
                continue;

            // The solutions published from one executable share its entry.
            hj::Status why;
            const auto view = library.resolve(device, indices.front(), why);
            {
                INFO(("resolve: " + why.message));
                REQUIRE((view.master != nullptr));
            }
            const auto& mapping = view.master->libraryMapping;
            const auto  prefix  = mapping.at(indices.front());
            for(const auto index : indices)
            {
                INFO((label + " published its solutions into more than one entry"));
                REQUIRE((mapping.count(index) && mapping.at(index) == prefix));
            }
            const auto file = library.directory(key) / "slices" / hj::sliceKey(problem)
                              / fs::u8path(prefix + ".dat");
            const auto entry = std::dynamic_pointer_cast<hj::GemmMaster>(
                TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(file.string()));
            {
                INFO(("The entry " + file.u8string() + " does not hold every solution of "
                      + label));
                REQUIRE((entry && entry->solutions.size() == indices.size()));
                for(const auto index : indices)
                    REQUIRE((entry->solutions.count(index)));
            }
            paired = true;
        }
    }
    {
        INFO(("plain-pair was not published as one entry, or nothing was loaded"));
        REQUIRE((pair && loaded && paired));
    }

    std::vector<int32_t> transposed;
    const auto           status = library.lookup(
        key, device, hipblaslt_jit_test::fp16Gemm(true, 512), *hardware, 4, {}, transposed);
    {
        INFO(("A transposed A found a published solution"));
        REQUIRE((status.ok() && transposed.empty()));
    }

    const auto& solutions = pair->assembly.solutions;
    {
        INFO(("The loader rejected solutions 0 and 1"));
        REQUIRE((hj::loadGemmLibrary(numbered(solutions, {0, 1}))->solutions.size() == 2));
    }
    hipblaslt_jit_test::reject([&] { hj::loadGemmLibrary(numbered(solutions, {0, 2})); },
                               "The loader accepted solutions 0 and 2");
}
