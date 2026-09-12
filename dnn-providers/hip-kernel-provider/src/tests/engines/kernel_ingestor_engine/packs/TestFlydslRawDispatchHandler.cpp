// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// THROWAWAY Phase-A proof. Not for release, not for a PR.
//
// Proves the "native escape hatch": a UDD author's own IKernelDispatchHandler<Handle>
// can bypass the ingestor's source-kind resolution entirely and raw-load a build-time
// flyDSL->HSACO code object with hipModuleLoadData + hipModuleLaunchKernel, then run it
// through the *real* dispatch-handler contract (prepare/launch) driven by the engine's
// pointwise fixture. This is the same three HIP calls run_hsaco.cpp already proves on
// this gfx1151 box, but sitting inside the hipDNN handler interface.

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fstream>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::findDeviceBuffer;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::PreparedDispatch;

#define FLY_CK(x)                                                                             \
    do                                                                                        \
    {                                                                                         \
        const hipError_t err = (x);                                                           \
        if(err != hipSuccess)                                                                 \
        {                                                                                     \
            ADD_FAILURE() << "HIP error " << err << " (" << hipGetErrorString(err) << ") at " \
                          << __FILE__ << ":" << __LINE__;                                     \
        }                                                                                     \
    } while(0)

// Build-time flyDSL->HSACO artifact from flydsl_poc_scratch. Absolute path is fine
// for a throwaway. Symbol vadd_0. gfx950 flyDSL 0.3.x kernarg ABI: each Tensor arg is
// (global_buffer ptr, by_value i32 size) => 44 bytes:
//   out_ptr@0(8) out_N@8(4)  a_ptr@16(8) a_N@24(4)  b_ptr@32(8) b_N@40(4).
constexpr const char* FLYDSL_HSACO_PATH
    = "/home/AMD/brpepers/wt/hipdnn-flydsl-poc/"
      "flydsl_poc_scratch/vadd_gfx950.hsaco";
constexpr const char* FLYDSL_SYMBOL = "vadd_0";

/// Author-owned launch state: a loaded HIP module + resolved function. Holds nothing
/// tied to the MatchContext it was prepared from, per the PreparedDispatch contract.
class PreparedFlydsl : public PreparedDispatch
{
public:
    PreparedFlydsl(hipModule_t mod, hipFunction_t fn)
        : _mod(mod)
        , _fn(fn)
    {
    }

    ~PreparedFlydsl() override
    {
        if(_mod != nullptr)
        {
            static_cast<void>(hipModuleUnload(_mod));
        }
    }

    PreparedFlydsl(const PreparedFlydsl&) = delete;
    PreparedFlydsl& operator=(const PreparedFlydsl&) = delete;

    hipFunction_t function() const
    {
        return _fn;
    }

private:
    hipModule_t _mod = nullptr;
    hipFunction_t _fn = nullptr;
};

/// The escape hatch: ignores the ingestor's KernelDefinition/source-kind machinery and
/// raw-loads a flyDSL HSACO. prepare() = hipModuleLoadData + hipModuleGetFunction;
/// launch() = uid->pointer resolution + hipModuleLaunchKernel.
class FlydslRawDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& /*context*/,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        std::vector<char> blob;
        {
            // std::fopen is deprecated in the Windows CRT; a stream reads the same bytes.
            std::ifstream file(FLYDSL_HSACO_PATH, std::ios::binary | std::ios::ate);
            if(!file)
            {
                throw std::runtime_error(std::string("cannot open HSACO: ") + FLYDSL_HSACO_PATH);
            }
            const auto n = static_cast<std::streamsize>(file.tellg());
            file.seekg(0);
            blob.resize(static_cast<size_t>(n));
            if(!file.read(blob.data(), n))
            {
                throw std::runtime_error("short read on HSACO");
            }
        }

        hipModule_t mod = nullptr;
        if(hipModuleLoadData(&mod, blob.data()) != hipSuccess)
        {
            throw std::runtime_error("hipModuleLoadData failed");
        }
        hipFunction_t fn = nullptr;
        if(hipModuleGetFunction(&fn, mod, FLYDSL_SYMBOL) != hipSuccess)
        {
            static_cast<void>(hipModuleUnload(mod));
            throw std::runtime_error("hipModuleGetFunction failed");
        }
        return std::make_unique<PreparedFlydsl>(mod, fn);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& fly = static_cast<const PreparedFlydsl&>(prepared);

        const auto a = findDeviceBuffer(INPUT_A_UID, deviceBuffers, numDeviceBuffers);
        const auto b = findDeviceBuffer(INPUT_B_UID, deviceBuffers, numDeviceBuffers);
        const auto out = findDeviceBuffer(OUTPUT_UID, deviceBuffers, numDeviceBuffers);

        // flyDSL 0.3.x ABI: (ptr, i32 size) per Tensor, 44 bytes total. This is a
        // 1-element add, so every runtime size is 1 (thread 0 does out[0]=a[0]+b[0]).
        std::array<unsigned char, 44> args{};
        const int32_t elems = 1;
        void* out_ptr = out.ptr;
        void* a_ptr = a.ptr;
        void* b_ptr = b.ptr;
        std::memcpy(args.data() + 0, &out_ptr, sizeof(void*));
        std::memcpy(args.data() + 8, &elems, sizeof(int32_t));
        std::memcpy(args.data() + 16, &a_ptr, sizeof(void*));
        std::memcpy(args.data() + 24, &elems, sizeof(int32_t));
        std::memcpy(args.data() + 32, &b_ptr, sizeof(void*));
        std::memcpy(args.data() + 40, &elems, sizeof(int32_t));
        size_t argsz = args.size();

        std::array<void*, 5> config{HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                    args.data(),
                                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                    &argsz,
                                    HIP_LAUNCH_PARAM_END};

        // One element: grid(1,1,1) block(1,1,1); thread 0 does out[0]=a[0]+b[0].
        FLY_CK(hipModuleLaunchKernel(
            fly.function(), 1, 1, 1, 1, 1, 1, 0, handle.getStream(), nullptr, config.data()));
    }
};

/// 1-element add buffers keyed by the pointwise fixture's uids.
class AddBuffers
{
public:
    AddBuffers(float a, float b)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_a, sizeof(float)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_b, sizeof(float)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_c, sizeof(float)));
        EXPECT_EQ(hipSuccess, hipMemcpy(_a, &a, sizeof(float), hipMemcpyHostToDevice));
        EXPECT_EQ(hipSuccess, hipMemcpy(_b, &b, sizeof(float), hipMemcpyHostToDevice));
    }

    ~AddBuffers()
    {
        static_cast<void>(hipFree(_a));
        static_cast<void>(hipFree(_b));
        static_cast<void>(hipFree(_c));
    }

    AddBuffers(const AddBuffers&) = delete;
    AddBuffers& operator=(const AddBuffers&) = delete;

    std::array<hipdnnPluginDeviceBuffer_t, 3> descriptors() const
    {
        return {hipdnnPluginDeviceBuffer_t{INPUT_A_UID, _a},
                hipdnnPluginDeviceBuffer_t{INPUT_B_UID, _b},
                hipdnnPluginDeviceBuffer_t{OUTPUT_UID, _c}};
    }

    float readResult() const
    {
        float result{};
        EXPECT_EQ(hipSuccess, hipMemcpy(&result, _c, sizeof(float), hipMemcpyDeviceToHost));
        return result;
    }

private:
    void* _a = nullptr;
    void* _b = nullptr;
    void* _c = nullptr;
};

TEST(TestFlydslRawDispatch, LaunchesAFlydslHsacoThroughTheEscapeHatch)
{
    SKIP_IF_NO_DEVICES();

    // FLYDSL_HSACO_PATH is baked at configure time and points at a scratch artifact the POC
    // built outside the tree, so it exists on the author's machine and nowhere else. Absent
    // means "this POC's artifact was not staged here", which is the same not-installed
    // condition the attention pack suites skip on -- not a failure of the dispatch path.
    if(!std::ifstream(FLYDSL_HSACO_PATH).good())
    {
        GTEST_SKIP() << "flyDSL HSACO not staged: " << FLYDSL_HSACO_PATH;
    }

    // Real fixture-built context/bindings/kernel, exactly what the engine would hand a
    // handler — our handler ignores them and raw-loads the flyDSL HSACO instead.
    const GraphFixture fixture(buildPointwiseGraph(), currentDeviceProperties());
    auto bound = matchesGraph(POINTWISE_ADD, fixture.context());
    ASSERT_TRUE(bound.has_value());

    const FlydslRawDispatchHandler handler;

    const auto prepared = handler.prepare(fixture.context(), *bound, makeKernel(64, "FLOAT"));
    ASSERT_NE(prepared, nullptr);

    const Handle handle;
    const AddBuffers buffers(3.0f, 4.0f);
    const auto descriptors = buffers.descriptors();

    handler.launch(
        handle, *prepared, descriptors.data(), static_cast<uint32_t>(descriptors.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    EXPECT_FLOAT_EQ(buffers.readResult(), 7.0f);
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
