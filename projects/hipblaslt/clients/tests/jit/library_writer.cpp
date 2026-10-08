// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime_api.h>

#include <filesystem>
#include <set>
#include <string>

// Builds the committed assembly for this device and publishes, at each K the
// tests use, every solution that supports it into the JIT solution library.
// CTest runs this before the loader.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;
}

TEST_CASE("pre-generated kernels publish into the JIT solution library", "[jit-gpu]")
{
    int             device = 0;
    hipDeviceProp_t properties{};
    {
        INFO(("Cannot query the current HIP device"));
        REQUIRE((hipGetDevice(&device) == hipSuccess
            && hipGetDeviceProperties(&properties, device) == hipSuccess));
    }
    const auto target      = hipblaslt_jit_test::deviceTarget(properties);
    const auto hardware    = TensileLite::hip::GetDevice(properties, device);
    const auto libraryRoot = fs::u8path(HIPBLASLT_JIT_LIBRARY);
    const auto scratch     = fs::u8path(HIPBLASLT_JIT_SCRATCH);
    fs::remove_all(libraryRoot);
    fs::remove_all(scratch);
    fs::create_directories(scratch);

    hj::JitLibrary    library(libraryRoot);
    const auto        key = hipblaslt_jit_test::pregeneratedKey(properties);
    const auto        prepared
        = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA), target, scratch);
    std::set<int32_t> published;
    bool              paired = false;
    for(const auto& item : prepared)
    {
        hj::Artifact executable;
        auto builtStatus
            = hj::makeComgrBuilder()->build(item.source, hj::BuildKind::Executable, executable);
        hj::keepBuildLog(builtStatus, scratch);
        {
            INFO(("The build for " + target.targetId() + " failed: " + builtStatus.message));
            REQUIRE((builtStatus.ok()));
        }
        for(const auto k : item.ks)
        {
            const auto problem   = hipblaslt_jit_test::fp16Gemm(false, k);
            const auto solutions
                = hipblaslt_jit_test::supported(item.assembly.solutions, problem, *hardware);
            const auto label = item.name + " K=" + std::to_string(k);
            std::vector<int32_t> indices;
            const auto status
                = library.publish(key, device, problem, executable, solutions, indices);
            {
                INFO((label + ": " + status.message));
                REQUIRE((status.ok() && !solutions.empty()));
            }
            {
                INFO((label + " did not receive one new JIT solution index per solution"));
                REQUIRE((indices.size() == solutions.size()));
                for(const auto index : indices)
                    REQUIRE((hj::isJitIndex(index) && published.insert(index).second));
            }

            std::vector<int32_t> again;
            const auto repeated
                = library.publish(key, device, problem, executable, solutions, again);
            {
                INFO((label + " published again did not keep its indices: " + repeated.message));
                REQUIRE((repeated.ok() && again == indices));
            }
            paired |= solutions.size() > 1;
        }
    }
    {
        INFO(("Expected plain at K=512, both plain-pair solutions at K=1024, and one at K=256"));
        REQUIRE((published.size() == 4 && paired));
    }
}
