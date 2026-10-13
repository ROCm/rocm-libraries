// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include "hipblaslt-jit-fs.hpp"
#include "hipblaslt-jit-msgpack.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

// Builds the committed assembly for this device and publishes, at each K the
// tests use, every solution that supports it into the JIT solution library.
// CTest runs this before the loader. Also checks the support check on its own
// and that publication allocates no index at or above jitIndexEnd.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    std::vector<std::string> names(const std::vector<hipblaslt_jit_test::Solution>& solutions)
    {
        std::vector<std::string> result;
        for(const auto& solution : solutions)
            result.push_back(solution->solutionName);
        return result;
    }
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
            const auto solutions = hj::supportingSolutions(
                item.source.solutions, problem, *hardware, hipblaslt_jit_test::workspaceLimit);
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

TEST_CASE("the support check applies each solution's predicates and workspace", "[jit-gpu]")
{
    int             device = 0;
    hipDeviceProp_t properties{};
    {
        INFO(("Cannot query the current HIP device"));
        REQUIRE((hipGetDevice(&device) == hipSuccess
            && hipGetDeviceProperties(&properties, device) == hipSuccess));
    }
    const auto target   = hipblaslt_jit_test::deviceTarget(properties);
    const auto hardware = TensileLite::hip::GetDevice(properties, device);
    const auto prepared
        = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA), target);
    const auto& pair = prepared.back().source.solutions;
    REQUIRE((prepared.back().name == "plain-pair" && pair.size() == 2));

    constexpr auto limit = hipblaslt_jit_test::workspaceLimit;
    const auto     k1024 = hipblaslt_jit_test::fp16Gemm(false, 1024);
    const auto     k256  = hipblaslt_jit_test::fp16Gemm(false, 256);
    {
        INFO(("K=1024 did not keep both plain-pair solutions, best first"));
        REQUIRE((names(hj::supportingSolutions(pair, k1024, *hardware, limit)) == names(pair)));
    }
    {
        INFO(("K=256 did not keep only the solution for any K"));
        REQUIRE((names(hj::supportingSolutions(pair, k256, *hardware, limit))
                 == names({pair.back()})));
    }
    {
        INFO(("A transposed A kept a solution"));
        const auto transposed = hipblaslt_jit_test::fp16Gemm(true, 1024);
        REQUIRE((hj::supportingSolutions(pair, transposed, *hardware, limit).empty()));
    }
    {
        INFO(("Another processor kept a solution"));
        const auto other = target.processor == hj::Processor::gfx90a ? hj::Processor::gfx942
                                                                     : hj::Processor::gfx90a;
        const TensileLite::AMDGPU elsewhere(other, properties.multiProcessorCount, "other");
        REQUIRE((hj::supportingSolutions(pair, k1024, elsewhere, limit).empty()));
    }

    // A split-K copy needs workspace for its partial tiles.
    auto split                               = hj::copySolution(pair.back());
    split->sizeMapping.globalSplitU          = 4;
    split->sizeMapping.workspaceSizePerElemC = 4;
    const auto required                      = split->requiredWorkspaceSize(k256, *hardware);
    {
        INFO(("The split-K copy needs no workspace"));
        REQUIRE((required > 0));
    }
    {
        INFO(("A workspace limit below the requirement kept the split-K copy"));
        REQUIRE((hj::supportingSolutions({split}, k256, *hardware, required - 1).empty()));
    }
    {
        INFO(("A workspace limit equal to the requirement dropped the split-K copy"));
        REQUIRE((hj::supportingSolutions({split}, k256, *hardware, required).size() == 1));
    }
}

TEST_CASE("publication allocates no index at or above jitIndexEnd", "[jit-gpu]")
{
    {
        INFO(("The JIT range is not [jitIndexBase, jitIndexEnd)"));
        REQUIRE((hj::isJitIndex(hj::jitIndexBase) && hj::isJitIndex(hj::jitIndexEnd - 1)
                 && !hj::isJitIndex(hj::jitIndexBase - 1) && !hj::isJitIndex(hj::jitIndexEnd)
                 && !hj::isJitIndex(INT32_MAX)));
    }
    int             device = 0;
    hipDeviceProp_t properties{};
    {
        INFO(("Cannot query the current HIP device"));
        REQUIRE((hipGetDevice(&device) == hipSuccess
            && hipGetDeviceProperties(&properties, device) == hipSuccess));
    }
    const auto target  = hipblaslt_jit_test::deviceTarget(properties);
    const auto scratch = fs::u8path(HIPBLASLT_JIT_SCRATCH) / "index-end";
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    const auto plain
        = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA), target, scratch)
              .front();
    hj::Artifact executable;
    auto         built
        = hj::makeComgrBuilder()->build(plain.source, hj::BuildKind::Executable, executable);
    hj::keepBuildLog(built, scratch);
    {
        INFO(("The build for " + target.targetId() + " failed: " + built.message));
        REQUIRE((built.ok()));
    }

    // Each K is its own entry, so publishing at a new K allocates an index.
    hj::JitLibrary library(scratch / "library");
    const auto     key     = hipblaslt_jit_test::pregeneratedKey(properties);
    const auto     publish = [&](size_t k, std::vector<int32_t>& indices) {
        return library.publish(key,
                               device,
                               hipblaslt_jit_test::fp16Gemm(false, k),
                               executable,
                               plain.source.solutions,
                               indices);
    };
    const auto allocator = library.buildDirectory() / "allocator.dat";
    const auto setNext   = [&](int64_t next) {
        std::vector<uint8_t> bytes;
        INFO(("Cannot write " + allocator.u8string()));
        REQUIRE((hj::msgpack_io::writeAllocator(next, bytes).ok()
                 && hj::files::writeAtomically(library.directory(key) / "staging", allocator, bytes)
                        .ok()));
    };
    const auto storedNext = [&] {
        std::vector<uint8_t> bytes;
        int64_t              next = 0;
        INFO(("Cannot read " + allocator.u8string()));
        REQUIRE((hj::files::readFile(allocator, 1 << 20, bytes).ok()
                 && hj::msgpack_io::readAllocator(bytes, next).ok()));
        return next;
    };

    std::vector<int32_t> first, last, past, again;
    {
        const auto status = publish(512, first);
        INFO(("publish: " + status.message));
        REQUIRE((status.ok() && first == std::vector<int32_t>{hj::jitIndexBase}));
    }
    setNext(hj::jitIndexEnd - 1);
    {
        const auto status = publish(1024, last);
        INFO(("The last JIT index was not allocated: " + status.message));
        REQUIRE((status.ok() && last == std::vector<int32_t>{hj::jitIndexEnd - 1}));
    }
    {
        const auto status = publish(1536, past);
        INFO(("An exhausted range allocated an index: " + status.message));
        REQUIRE((!status.ok() && status.message.find("exhausted") != std::string::npos
                 && past.empty()));
    }
    {
        const auto next = storedNext();
        INFO(("The allocator holds " + std::to_string(next) + ", not jitIndexEnd"));
        REQUIRE((next == hj::jitIndexEnd));
    }
    {
        const auto status = publish(1024, again);
        INFO(("A published solution lost its index once the range was exhausted: "
              + status.message));
        REQUIRE((status.ok() && again == last));
    }

    hj::Status why;
    const auto lastView = library.resolve(device, hj::jitIndexEnd - 1, why);
    {
        INFO(("The last JIT index does not resolve: " + why.message));
        REQUIRE((lastView.master != nullptr));
    }
    const auto endView = library.resolve(device, hj::jitIndexEnd, why);
    {
        INFO(("jitIndexEnd resolved as a JIT index"));
        REQUIRE((endView.master == nullptr
                 && why.message.find("outside the reserved JIT range") != std::string::npos));
    }

    setNext(hj::jitIndexEnd + 1);
    {
        const auto status = publish(1024, again);
        INFO(("An allocator above jitIndexEnd was used"));
        REQUIRE((!status.ok()
                 && status.message.find("outside the JIT range") != std::string::npos));
    }
}
