// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime_api.h>

#include <filesystem>
#include <string>

// Builds the committed assembly for this device and publishes each described
// solution into the JIT solution library. CTest runs this before the loader.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;
    using hipblaslt_jit_test::require;
}

TEST_CASE("pre-generated kernels publish into the JIT solution library", "[jit-gpu]")
{
    int             device = 0;
    hipDeviceProp_t properties{};
    require(hipGetDevice(&device) == hipSuccess
                && hipGetDeviceProperties(&properties, device) == hipSuccess,
            "Cannot query the current HIP device");
    const std::string target(properties.gcnArchName);
    const auto        arch    = target.substr(0, target.find(':'));
    const auto        libraryRoot = fs::u8path(HIPBLASLT_JIT_LIBRARY);
    const auto        scratch = fs::u8path(HIPBLASLT_JIT_SCRATCH);
    fs::remove_all(libraryRoot);
    fs::remove_all(scratch);
    fs::create_directories(scratch);

    hj::JitLibrary library(libraryRoot);
    const auto     key       = hipblaslt_jit_test::pregeneratedKey(properties);
    const auto     prepared  = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA));
    int            published = 0;
    for(const auto& item : prepared)
    {
        if(item.arch != arch)
            continue;
        hj::BuiltSolution built;
        const auto        builtStatus = hj::makeComgrBuilder()->build(
            item.solution, {target, hj::jitCodeObjectVersion, scratch}, built);
        require(builtStatus.ok(), "The build for " + target + " failed: " + builtStatus.message);
        for(const auto& publication : item.publications)
        {
            std::vector<int32_t>  indices;
            hj::SupportedSolutions solutions{{built, std::vector<int>{publication.local}}};
            const auto            status = library.publish(key,
                                                device,
                                                hipblaslt_jit_test::fp16Gemm(false, publication.k),
                                                solutions,
                                                indices);
            require(status.ok(),
                    item.name + " K=" + std::to_string(publication.k) + ": " + status.message);
            require(indices.size() == 1 && hj::isJitIndex(indices[0]),
                    item.name + " did not receive a JIT solution index");
            ++published;
        }
    }
    require(published == 3, "Expected the plain solution and both plain-pair solutions");
}
