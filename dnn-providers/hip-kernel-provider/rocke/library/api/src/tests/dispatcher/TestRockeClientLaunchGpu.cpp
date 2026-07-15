// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU-tier end-to-end test for the rocKE-client engine.
//
// It drives the SHIPPING engine path against the real per-arch AOT kpack the
// build packs into <plugin>/arch_content/rocke/<arch>/:
//
//   dispatcher::loadManifestsFromDirectory  (manifest + args_signature parse)
//   RockeClientEngine::isApplicable          (dispatcher selection for the arch)
//   RockeClientEngine::initializeExecutionContext
//                                            (SdpaGraphAdapter graph->bindings,
//                                             RockeClientPlan kpack->hipModule)
//   RockeClientContext::plan().execute       (bindArgs/packArgs/evalGrid + launch)
//
// A real single-node SDPA-forward op-graph (the SdpaGraphFixture the unit tests
// use) is fed in, the plan executes the actual kernel on device buffers keyed by
// the graph's tensor uids, and the output is checked against the shared SDPA
// golden reference (CpuFpReferenceSdpa, the same fp32 reference the AITER SDPA
// tests validate against) within a principled tolerance (calculateSdpaFwdTolerance).
// Nothing about the launch or the reference is reimplemented here.
//
// Device-gated: skips (never fails) without a HIP device, without a bundle for
// the running arch (build the rocke_client_aot_kpack target to produce it), or
// when the engine declines the graph (no matching AOT instance for the arch).

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesSdpa.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "RockeClientContext.hpp"
#include "RockeClientHandle.hpp"
#include "dispatcher/AotCatalog.hpp"
#include "engines/RockeClientEngine.hpp"
#include "tests/dispatcher/SdpaGraphFixture.hpp"

namespace rocke_client
{
namespace
{
using hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;

// Tensor uids the SdpaGraphFixture assigns, in creation order (q, k, v, o).
constexpr std::int64_t Q_UID = 1;
constexpr std::int64_t K_UID = 2;
constexpr std::int64_t V_UID = 3;
constexpr std::int64_t O_UID = 4;

// Random inputs are drawn from [-VAL_BOUND, VAL_BOUND]; the same bounds feed the
// tolerance estimate (which takes double).
constexpr double VAL_BOUND = 0.3;

// Bare gfx arch of device 0 ("gfx1151"), or "" on failure. Mirrors deviceArch()
// in RockeClientDispatcher.cpp.
std::string deviceArch(int deviceId)
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, deviceId) != hipSuccess)
    {
        return {};
    }
    std::string arch(static_cast<const char*>(props.gcnArchName));
    const auto colon = arch.find(':');
    if(colon != std::string::npos)
    {
        arch.resize(colon);
    }
    return arch;
}

// BSHD-contiguous strides for logical dims [B, H, S, D] -- matches the layout the
// SdpaGraphFixture emits (physical B, S, H, D), so a tensor's packed host buffer
// is exactly what the kernel reads once copied to the device.
std::vector<std::int64_t> bshdStrides(std::int64_t h, std::int64_t s, std::int64_t d)
{
    return {s * h * d, d, h * d, 1};
}

// Allocate a device copy of a host tensor's packed buffer and return the pointer.
void* toDevice(const Tensor<half>& tensor)
{
    void* devicePtr = nullptr;
    const std::size_t bytes = tensor.elementCount() * sizeof(half);
    EXPECT_EQ(hipMalloc(&devicePtr, bytes), hipSuccess);
    EXPECT_EQ(hipMemcpy(devicePtr,
                        const_cast<Tensor<half>&>(tensor).rawHostData(),
                        bytes,
                        hipMemcpyHostToDevice),
              hipSuccess);
    return devicePtr;
}

TEST(TestRockeClientLaunchGpu, EngineLaunchesRealSdpaKernelAndMatchesGoldenRef)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "No HIP device available";
    }
    const std::string arch = deviceArch(0);
    if(arch.empty())
    {
        GTEST_SKIP() << "Could not determine device arch";
    }

    const std::filesystem::path root(ROCKE_AOT_BUNDLE_ROOT);
    if(!std::filesystem::is_directory(root / arch))
    {
        GTEST_SKIP() << "No rocKE AOT bundle for arch '" << arch << "' under " << root
                     << "; build the rocke_client_aot_kpack target to generate it";
    }

    auto instances = dispatcher::loadManifestsFromDirectory(root);
    if(instances.empty())
    {
        GTEST_SKIP() << "No AOT instances parsed from the bundle under " << root;
    }

    // Real engine over the build-packed catalog, and a real device stream so the
    // dispatcher resolves the running arch from the handle.
    const RockeClientEngine engine{dispatcher::AotCatalog{std::move(instances)}};
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    RockeClientHandle handle;
    handle.setStream(stream);

    // Real single-node fp16/BSHD SDPA graph; defaults match the gfx1151 instance
    // (batch 2, 4 heads, seqlen 64, head dim 64, mask none, default scale).
    const dispatcher::test::SdpaGraphConfig cfg{};
    const auto fixture = dispatcher::test::buildSdpaGraph(cfg);
    const auto graph = fixture.graphWrapper();

    if(!engine.isApplicable(handle, graph))
    {
        static_cast<void>(hipStreamDestroy(stream));
        GTEST_SKIP() << "engine declined the graph on arch '" << arch
                     << "' (no matching AOT instance)";
    }

    // Build the plan through the engine -- this loads the real kernel from kpack.
    RockeClientContext context;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper engineConfig(nullptr,
                                                                                         0);
    ASSERT_NO_THROW(engine.initializeExecutionContext(handle, graph, engineConfig, context));
    ASSERT_TRUE(context.hasValidPlan());

    // Host tensors laid out [B, H, S, D] with the fixture's BSHD strides, so the
    // reference reads them logically while their packed buffers match the kernel.
    const auto b = cfg.batch;
    const auto hq = cfg.numQueryHeads;
    const auto hkv = cfg.numKvHeads;
    const auto sq = cfg.seqlenQ;
    const auto sk = cfg.seqlenK;
    const auto d = cfg.headSizeQK;
    const auto dv = cfg.headSizeV;

    Tensor<half> q({b, hq, sq, d}, bshdStrides(hq, sq, d));
    Tensor<half> k({b, hkv, sk, d}, bshdStrides(hkv, sk, d));
    Tensor<half> v({b, hkv, sk, dv}, bshdStrides(hkv, sk, dv));
    Tensor<half> oRef({b, hq, sq, dv}, bshdStrides(hq, sq, dv));
    Tensor<half> oDev({b, hq, sq, dv}, bshdStrides(hq, sq, dv));
    q.fillWithRandomValues(
        half{static_cast<float>(-VAL_BOUND)}, half{static_cast<float>(VAL_BOUND)}, 1U);
    k.fillWithRandomValues(
        half{static_cast<float>(-VAL_BOUND)}, half{static_cast<float>(VAL_BOUND)}, 2U);
    v.fillWithRandomValues(
        half{static_cast<float>(-VAL_BOUND)}, half{static_cast<float>(VAL_BOUND)}, 3U);

    void* qd = toDevice(q);
    void* kd = toDevice(k);
    void* vd = toDevice(v);
    void* od = nullptr;
    ASSERT_EQ(hipMalloc(&od, oDev.elementCount() * sizeof(half)), hipSuccess);
    ASSERT_EQ(hipMemset(od, 0, oDev.elementCount() * sizeof(half)), hipSuccess);

    const std::array<hipdnnPluginDeviceBuffer_t, 4> buffers{{
        {.uid = Q_UID, .ptr = qd},
        {.uid = K_UID, .ptr = kd},
        {.uid = V_UID, .ptr = vd},
        {.uid = O_UID, .ptr = od},
    }};

    ASSERT_NO_THROW(
        context.plan().execute(handle, buffers.data(), static_cast<std::uint32_t>(buffers.size())));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(
        hipMemcpy(
            oDev.rawHostData(), od, oDev.elementCount() * sizeof(half), hipMemcpyDeviceToHost),
        hipSuccess);

    EXPECT_EQ(hipFree(qd), hipSuccess);
    EXPECT_EQ(hipFree(kd), hipSuccess);
    EXPECT_EQ(hipFree(vd), hipSuccess);
    EXPECT_EQ(hipFree(od), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

    // Golden reference (fp32 compute) and its principled tolerance -- the same
    // utilities the AITER SDPA tests use. Default scale, no mask, matching cfg.
    CpuFpReferenceSdpa::forward(q, k, v, oRef);
    const float tolerance = hipdnn_test_sdk::utilities::sdpa::calculateSdpaFwdTolerance<half, half>(
        -VAL_BOUND, VAL_BOUND, -VAL_BOUND, VAL_BOUND, -VAL_BOUND, VAL_BOUND, d, sk);

    const auto* refValues = static_cast<const half*>(const_cast<Tensor<half>&>(oRef).rawHostData());
    const auto* devValues = static_cast<const half*>(oDev.rawHostData());
    float maxAbs = 0.0F;
    for(std::size_t i = 0; i < oDev.elementCount(); ++i)
    {
        maxAbs = std::max(
            maxAbs, std::abs(static_cast<float>(devValues[i]) - static_cast<float>(refValues[i])));
    }
    EXPECT_LE(maxAbs, tolerance) << "device SDPA output diverged from the golden reference";
}

} // namespace
} // namespace rocke_client
