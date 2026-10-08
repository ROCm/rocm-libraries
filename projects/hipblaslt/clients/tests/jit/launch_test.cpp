// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime_api.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

// Compiles this device's committed assembly, publishes the code object into
// the JIT solution library, retrieves it through the solution library, and
// launches it the way the TensileLite client does.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    struct DeviceMemory
    {
        void* pointer = nullptr;

        explicit DeviceMemory(size_t bytes)
        {
            if(bytes != 0 && hipMalloc(&pointer, bytes) != hipSuccess)
                pointer = nullptr;
        }

        ~DeviceMemory()
        {
            if(pointer)
                static_cast<void>(hipFree(pointer));
        }

        DeviceMemory(const DeviceMemory&)            = delete;
        DeviceMemory& operator=(const DeviceMemory&) = delete;
    };
}

TEST_CASE("a published JIT solution launches", "[jit-gpu]")
{
    int device = 0;
    {
        INFO(("Cannot select the current HIP device"));
        REQUIRE((hipGetDevice(&device) == hipSuccess));
    }

    hipDeviceProp_t properties{};
    {
        INFO(("Cannot query the current HIP device"));
        REQUIRE((hipGetDeviceProperties(&properties, device) == hipSuccess));
    }
    hj::DeviceTarget target;
    target.device        = device;
    target.properties    = properties;
    target.targetId      = properties.gcnArchName;
    target.isa           = target.targetId.substr(0, target.targetId.find(':'));
    target.libraryArch   = target.isa;
    target.wavefrontSize = properties.warpSize;
    target.cuCount       = properties.multiProcessorCount;
    target.hardware      = TensileLite::hip::GetDevice(properties, device);
    REQUIRE((target.hardware != nullptr));

    const auto prepared = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA));
    const hipblaslt_jit_test::Prepared* plain = nullptr;
    for(const auto& item : prepared)
        if(item.arch == target.isa && item.name == "plain")
            plain = &item;
    {
        INFO(("No committed plain assembly for " + target.isa));
        REQUIRE((plain != nullptr && !plain->publications.empty()));
    }
    const auto& publication = plain->publications.front();

    const auto libraryRoot = fs::u8path(HIPBLASLT_JIT_LAUNCH_LIBRARY);
    const auto scratch     = fs::u8path(HIPBLASLT_JIT_LAUNCH_SCRATCH);
    fs::remove_all(libraryRoot);
    fs::remove_all(scratch);
    fs::create_directories(scratch);

    hj::BackendInfo backend;
    backend.id      = "pregenerated";
    backend.version = 1;
    const auto key  = hj::CacheKey::make(target, backend, hj::jitCodeObjectVersion);

    hj::CompiledCode built;
    hj::BuildRequest request{target.targetId, hj::jitCodeObjectVersion, scratch, plain->kernelNames};
    const auto       builtStatus = hj::makeComgrBuilder()->build(plain->source, request, built);
    {
        INFO(("The build for " + target.targetId + " failed: " + builtStatus.message));
        REQUIRE((builtStatus.ok()));
    }
    built.record = plain->record;

    hj::JitLibrary             library(libraryRoot);
    const auto                 problem = hipblaslt_jit_test::fp16Gemm(false, publication.k);
    std::vector<int32_t>       published;
    hj::SupportedSolutions     solutions{{built, std::vector<int>{publication.local}}};
    const auto                 publishedStatus
        = library.publish(key, device, problem, solutions, published);
    {
        INFO(("publish: " + publishedStatus.message));
        REQUIRE((publishedStatus.ok()));
    }
    {
        INFO(("publish did not return one JIT solution index"));
        REQUIRE((published.size() == 1 && hj::isJitIndex(published.front())));
    }

    constexpr size_t workspace = 64 * 1024 * 1024;
    library.beginQuery(false, target, backend, workspace, {}, nullptr);
    const auto found   = library.findTopSolutions(problem, *target.hardware, 1);
    const auto byIndex = library.getSolutionByIndex(problem, *target.hardware, published.front());
    const auto query   = library.queryStatus();
    library.endQuery();
    {
        INFO(("retrieval: " + query.message));
        REQUIRE((query.ok()));
    }
    {
        INFO(("findTopSolutions did not return the published kernel"));
        REQUIRE((found.size() == 1 && found.front() && found.front()->KernelName() == publication.kernel));
    }
    {
        INFO(("getSolutionByIndex did not return the published kernel"));
        REQUIRE((byIndex && byIndex->KernelName() == publication.kernel));
    }

    hj::Status why;
    const auto view = library.resolve(device, published.front(), why);
    {
        INFO(("resolve: " + why.message));
        REQUIRE((view.adapter != nullptr));
    }

    constexpr size_t M = 256;
    constexpr size_t N = 128;
    const size_t     K = publication.k;
    DeviceMemory     A(M * K * sizeof(uint16_t));
    DeviceMemory     B(K * N * sizeof(uint16_t));
    DeviceMemory     C(M * N * sizeof(uint16_t));
    DeviceMemory     D(M * N * sizeof(uint16_t));
    {
        INFO(("Cannot allocate the GEMM tensors"));
        REQUIRE((A.pointer && B.pointer && C.pointer && D.pointer));
    }
    const auto   workspaceBytes = found.front()->requiredWorkspaceSize(problem, *target.hardware);
    DeviceMemory work(workspaceBytes);
    if(workspaceBytes != 0)
    {
        INFO(("Cannot allocate the kernel workspace"));
        REQUIRE((work.pointer != nullptr));
    }

    TensileLite::ContractionInputs inputs;
    inputs.a     = A.pointer;
    inputs.b     = B.pointer;
    inputs.c     = C.pointer;
    inputs.d     = D.pointer;
    inputs.alpha = 1.25f;
    inputs.beta  = 0.5f;
    inputs.ws    = work.pointer;

    std::vector<TensileLite::KernelInvocation> kernels;
    try
    {
        kernels = found.front()->solve(problem, inputs, *target.hardware);
    }
    catch(const std::exception& error)
    {
        INFO((std::string("solve: ") + error.what()));
        REQUIRE((false));
    }
    {
        INFO(("solve returned no kernel invocation"));
        REQUIRE((!kernels.empty()));
    }
    {
        INFO(("launchKernels failed"));
        REQUIRE((view.adapter->launchKernels(kernels) == hipSuccess));
    }
    {
        INFO(("the launched kernel did not complete"));
        REQUIRE((hipDeviceSynchronize() == hipSuccess));
    }
}
