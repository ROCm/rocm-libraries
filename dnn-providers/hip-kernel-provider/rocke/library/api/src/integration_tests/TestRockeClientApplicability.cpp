// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ABI-level integration tests for the rocke-client engine through the loaded
// plugin C ABI. The installed rocKE AOT bundles cover three SDPA forward shapes:
// the baseline [B=2, H=4, S=64, D=64], a larger head dim [D=128], and a
// grouped-query [Hq=8, Hkv=2] instance. On a device whose plugin-relative catalog
// ships a matching per-arch kpack each graph must be supported through
// graph.is_supported_ext(handle). Other devices still decline cleanly because
// no matching per-arch kpack bundle is installed.
//
// GPU dependency: hipdnnBackendFinalize reads the device from the handle's
// stream; SKIP_IF_NO_DEVICES() mirrors the established pattern used by
// IntegrationEngineHeuristicApi and IntegrationGraphSupportCheck.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend.hpp>
#endif // HIPDNN_ENABLE_SDPA

namespace
{

// Minimal RAII wrapper for hipdnnHandle_t (mirrors TestRockeClientLoad.cpp).
class HipdnnHandle
{
public:
    HipdnnHandle()
    {
        EXPECT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    ~HipdnnHandle()
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    hipdnnHandle_t get() const
    {
        return _handle;
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

// Minimal RAII wrapper for hipStream_t.
class HipStream
{
public:
    HipStream()
    {
        EXPECT_EQ(hipStreamCreate(&_stream), hipSuccess);
    }

    ~HipStream()
    {
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    hipStream_t get() const
    {
        return _stream;
    }

private:
    hipStream_t _stream = nullptr;
};

#ifdef HIPDNN_ENABLE_SDPA
std::string currentDeviceArch()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
    {
        return {};
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
    {
        return {};
    }
    std::string arch = props.gcnArchName;
    const auto colon = arch.find(':');
    if(colon != std::string::npos)
    {
        arch.resize(colon);
    }
    return arch;
}

// True iff an installed rocKE AOT bundle exists for the running device, using
// the same plugin-relative layout the dispatcher resolves at runtime
// (aotBundleDir() -> "arch_content/rocke/<arch>"). Keyed on the concrete
// artifact the plugin will load, so no arch-name list can drift.
bool rockeHasAotBundleForCurrentDevice(const std::filesystem::path& pluginDir)
{
    const auto arch = currentDeviceArch();
    if(arch.empty())
    {
        return false;
    }
    const auto kpack
        = pluginDir / "arch_content" / "rocke" / arch / ("rocke_client_" + arch + ".kpack");
    return std::filesystem::exists(kpack);
}
#endif // HIPDNN_ENABLE_SDPA

#ifdef HIPDNN_ENABLE_SDPA

// Shipped AOT SDPA forward shapes the checked-in bundles cover: the baseline plus
// the two added instances (larger head dim; grouped-query attention). Q/O carry
// numQueryHeads, K/V carry numKvHeads; fp16 I/O with BSHD physical strides to
// match the rocKE bundle catalog.
struct SdpaShape
{
    std::string name;
    int batch;
    int numQueryHeads;
    int numKvHeads;
    int seqLen;
    int headDim;
};

const SdpaShape BASELINE_SHAPE{.name = "baseline_d64_hq4",
                               .batch = 2,
                               .numQueryHeads = 4,
                               .numKvHeads = 4,
                               .seqLen = 64,
                               .headDim = 64};
const SdpaShape D128_SHAPE{
    .name = "d128", .batch = 2, .numQueryHeads = 4, .numKvHeads = 4, .seqLen = 64, .headDim = 128};
const SdpaShape GQA_SHAPE{.name = "gqa_hq8_hkv2",
                          .batch = 2,
                          .numQueryHeads = 8,
                          .numKvHeads = 2,
                          .seqLen = 64,
                          .headDim = 64};

// BSHD-contiguous strides for logical dims [B, H, S, D] (physical order B, S, H, D).
std::vector<int64_t> bshdStrides(int heads, int seqLen, int headDim)
{
    return {static_cast<int64_t>(seqLen) * heads * headDim,
            headDim,
            static_cast<int64_t>(heads) * headDim,
            1};
}

hipdnn_frontend::graph::Graph buildSdpaForwardGraph(const SdpaShape& shape)
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;

    const std::vector<int64_t> qDims
        = {shape.batch, shape.numQueryHeads, shape.seqLen, shape.headDim};
    const std::vector<int64_t> kvDims
        = {shape.batch, shape.numKvHeads, shape.seqLen, shape.headDim};
    const auto qStrides = bshdStrides(shape.numQueryHeads, shape.seqLen, shape.headDim);
    const auto kvStrides = bshdStrides(shape.numKvHeads, shape.seqLen, shape.headDim);

    Graph graph;
    graph.set_name("RockeClientApplicabilityTest_SdpaFwd_" + shape.name)
        .set_io_data_type(DataType::HALF)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto qAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("Q", qDims, qStrides));
    auto kAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("K", kvDims, kvStrides));
    auto vAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("V", kvDims, kvStrides));
    qAttr->set_uid(1);
    kAttr->set_uid(2);
    vAttr->set_uid(3);

    const SdpaAttributes sdpaAttrs;
    auto [oAttr, statsAttr] = graph.sdpa(qAttr, kAttr, vAttr, sdpaAttrs);
    oAttr->set_dim(qDims).set_stride(qStrides).set_output(true).set_uid(4);

    return graph;
}

// Baseline graph used by the applicability / override-shape ABI checks.
hipdnn_frontend::graph::Graph buildMinimalSdpaForwardGraph()
{
    return buildSdpaForwardGraph(BASELINE_SHAPE);
}

// Physical offset of element (b, h, s, d) in a BSHD-contiguous [B, H, S, D]
// tensor: memory order is B, S, H, D, so offset = ((b*S + s)*H + h)*D + d.
std::size_t bshdOffset(int b, int h, int s, int d, int numHeads, int seqLen, int headDim)
{
    return ((static_cast<std::size_t>(b) * static_cast<std::size_t>(seqLen)
             + static_cast<std::size_t>(s))
                * static_cast<std::size_t>(numHeads)
            + static_cast<std::size_t>(h))
               * static_cast<std::size_t>(headDim)
           + static_cast<std::size_t>(d);
}

// Unmasked SDPA forward reference in fp32 (scale = 1/sqrt(D), softmax over keys),
// mirroring the numeric oracle aot/tests/sdpa_aot_numeric.py. Supports GQA: query
// head h reads KV head h / (numQueryHeads / numKvHeads).
void referenceSdpaForward(const std::vector<_Float16>& q,
                          const std::vector<_Float16>& k,
                          const std::vector<_Float16>& v,
                          std::vector<float>& out,
                          int batch,
                          int numQueryHeads,
                          int numKvHeads,
                          int seqLen,
                          int headDim)
{
    const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));
    const int groupSize = numQueryHeads / numKvHeads;
    std::vector<float> scores(static_cast<std::size_t>(seqLen));
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < numQueryHeads; ++h)
        {
            const int hkv = h / groupSize;
            for(int sq = 0; sq < seqLen; ++sq)
            {
                float maxScore = -std::numeric_limits<float>::infinity();
                for(int sk = 0; sk < seqLen; ++sk)
                {
                    float dot = 0.0F;
                    for(int d = 0; d < headDim; ++d)
                    {
                        dot += static_cast<float>(
                                   q[bshdOffset(b, h, sq, d, numQueryHeads, seqLen, headDim)])
                               * static_cast<float>(
                                   k[bshdOffset(b, hkv, sk, d, numKvHeads, seqLen, headDim)]);
                    }
                    scores[static_cast<std::size_t>(sk)] = dot * scale;
                    maxScore = std::max(maxScore, scores[static_cast<std::size_t>(sk)]);
                }
                float sum = 0.0F;
                for(int sk = 0; sk < seqLen; ++sk)
                {
                    const float e = std::exp(scores[static_cast<std::size_t>(sk)] - maxScore);
                    scores[static_cast<std::size_t>(sk)] = e;
                    sum += e;
                }
                for(int d = 0; d < headDim; ++d)
                {
                    float acc = 0.0F;
                    for(int sk = 0; sk < seqLen; ++sk)
                    {
                        acc += scores[static_cast<std::size_t>(sk)]
                               * static_cast<float>(
                                   v[bshdOffset(b, hkv, sk, d, numKvHeads, seqLen, headDim)]);
                    }
                    out[bshdOffset(b, h, sq, d, numQueryHeads, seqLen, headDim)] = acc / sum;
                }
            }
        }
    }
}

#endif // HIPDNN_ENABLE_SDPA

} // namespace

#ifdef HIPDNN_ENABLE_SDPA

// Asserts that the rocke-client engine's installed kpack catalog is visible
// through the full public C ABI. On devices with a checked-in AOT bundle, a
// structurally matching SDPA graph is supported. Other devices still decline
// cleanly rather than accepting an unlaunchable graph.
TEST(TestRockeClientApplicability, RockeClientSupportsCheckedInSdpaAotShapeThroughAbi)
{
    // hipdnnBackendFinalize reads the HIP device from the handle's stream.
    SKIP_IF_NO_DEVICES();

    // Load ONLY the rocke-client plugin; no other engine can interfere.
    const auto pluginPath = std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
    const std::string pluginPathStr = pluginPath.string();
    const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    const HipdnnHandle handle;
    ASSERT_NE(handle.get(), nullptr);

    // build_operation_graph / hipdnnBackendFinalize reads the HIP device from
    // the stream associated with the handle.
    const HipStream stream;
    ASSERT_NE(stream.get(), nullptr);
    ASSERT_EQ(hipdnnSetStream(handle.get(), stream.get()), HIPDNN_STATUS_SUCCESS);

    for(const auto& shape : {BASELINE_SHAPE, D128_SHAPE, GQA_SHAPE})
    {
        hipdnn_frontend::graph::Graph graph = buildSdpaForwardGraph(shape);
        const auto result = graph.is_supported_ext(handle.get());
        if(rockeHasAotBundleForCurrentDevice(pluginPath.parent_path()))
        {
            EXPECT_TRUE(result.is_good())
                << "rocke-client should support checked-in SDPA AOT shape '" << shape.name
                << "' via kpack. Error: " << result.get_message();
        }
        else
        {
            EXPECT_FALSE(result.is_good())
                << "rocke-client should decline devices without a checked-in kpack bundle (shape '"
                << shape.name << "')";
        }
    }
}

// Companion to the above, exercising the full 1.1 override path. With the engine
// plugin API reported at 1.1.0 the host now routes override-shape graphs to this
// plugin (pre-1.1 plugins are filtered out before applicability). The
// rocke-client adapter declines any graph that opts into execute-time override
// shapes, so is_supported_ext must still return !good end-to-end: version gate ->
// applicability query -> adapter decline.
TEST(TestRockeClientApplicability, RockeClientDeclinesOverrideShapeGraphThroughAbi)
{
    SKIP_IF_NO_DEVICES();

    const auto pluginPath = std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
    const std::string pluginPathStr = pluginPath.string();
    const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    const HipdnnHandle handle;
    ASSERT_NE(handle.get(), nullptr);
    const HipStream stream;
    ASSERT_NE(stream.get(), nullptr);
    ASSERT_EQ(hipdnnSetStream(handle.get(), stream.get()), HIPDNN_STATUS_SUCCESS);

    hipdnn_frontend::graph::Graph graph = buildMinimalSdpaForwardGraph();
    graph.set_override_shape_enabled(true);

    const auto result = graph.is_supported_ext(handle.get());
    EXPECT_FALSE(result.is_good())
        << "rocke-client must decline override-shape graphs; is_supported_ext returned success "
           "unexpectedly. Error: "
        << result.get_message();
}

// The E2E acceptance gate: a config-driven AOT compile produces an installed
// .kpack + manifest, a matching SDPA graph selects that instance, loads its HSACO
// via RocKEKernelStore, builds a RockeClientPlan, launches it, and the device
// output matches an fp32 reference. GPU-gated: needs a real device whose
// installed plugin-relative catalog contains a matching bundle. Exercises the
// full runtime path the branch exists for, not just applicability.
class RockeClientNumericParity : public ::testing::TestWithParam<SdpaShape>
{
};

TEST_P(RockeClientNumericParity, ExecutesCheckedInSdpaAotShapeWithNumericParity)
{
    SKIP_IF_NO_DEVICES();
    const SdpaShape& shape = GetParam();

    const auto pluginPath = std::filesystem::weakly_canonical(
        hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
    if(!rockeHasAotBundleForCurrentDevice(pluginPath.parent_path()))
    {
        GTEST_SKIP() << "no installed rocKE AOT bundle for this device";
    }

    const std::string pluginPathStr = pluginPath.string();
    const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);

    const HipdnnHandle handle;
    ASSERT_NE(handle.get(), nullptr);
    const HipStream stream;
    ASSERT_NE(stream.get(), nullptr);
    ASSERT_EQ(hipdnnSetStream(handle.get(), stream.get()), HIPDNN_STATUS_SUCCESS);

    hipdnn_frontend::graph::Graph graph = buildSdpaForwardGraph(shape);

    // A present kpack ships every checked-in instance (asserted by
    // RockeClientSupportsCheckedInSdpaAotShapeThroughAbi), so a shipped shape
    // that this device declines is a real regression, not a skip.
    const auto supported = graph.is_supported_ext(handle.get());
    ASSERT_TRUE(supported.is_good())
        << "shipped SDPA AOT shape '" << shape.name
        << "' declined on a device with an installed kpack: " << supported.get_message();

    const auto buildResult = graph.build(handle.get());
    ASSERT_TRUE(buildResult.is_good()) << "graph.build failed: " << buildResult.get_message();

    int64_t workspaceSize = 0;
    ASSERT_TRUE(graph.get_workspace_size(workspaceSize).is_good());
    void* workspace = nullptr;
    if(workspaceSize > 0)
    {
        ASSERT_EQ(hipMalloc(&workspace, static_cast<std::size_t>(workspaceSize)), hipSuccess);
    }

    const std::size_t qElems
        = static_cast<std::size_t>(shape.batch) * static_cast<std::size_t>(shape.numQueryHeads)
          * static_cast<std::size_t>(shape.seqLen) * static_cast<std::size_t>(shape.headDim);
    const std::size_t kvElems
        = static_cast<std::size_t>(shape.batch) * static_cast<std::size_t>(shape.numKvHeads)
          * static_cast<std::size_t>(shape.seqLen) * static_cast<std::size_t>(shape.headDim);
    std::vector<_Float16> hQ(qElems);
    std::vector<_Float16> hK(kvElems);
    std::vector<_Float16> hV(kvElems);
    std::vector<_Float16> hO(qElems, static_cast<_Float16>(0.0F));
    std::mt19937 rng(0xA07F00DU);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for(std::size_t i = 0; i < qElems; ++i)
    {
        hQ[i] = static_cast<_Float16>(nd(rng) * 0.3F);
    }
    for(std::size_t i = 0; i < kvElems; ++i)
    {
        hK[i] = static_cast<_Float16>(nd(rng) * 0.3F);
        hV[i] = static_cast<_Float16>(nd(rng) * 0.3F);
    }

    const std::size_t qBytes = qElems * sizeof(_Float16);
    const std::size_t kvBytes = kvElems * sizeof(_Float16);
    void* dQ = nullptr;
    void* dK = nullptr;
    void* dV = nullptr;
    void* dO = nullptr;
    ASSERT_EQ(hipMalloc(&dQ, qBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dK, kvBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dV, kvBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&dO, qBytes), hipSuccess);
    ASSERT_EQ(hipMemcpy(dQ, hQ.data(), qBytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dK, hK.data(), kvBytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dV, hV.data(), kvBytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dO, 0, qBytes), hipSuccess);

    std::unordered_map<int64_t, void*> variantPack = {{1, dQ}, {2, dK}, {3, dV}, {4, dO}};
    const auto execResult = graph.execute(handle.get(), variantPack, workspace);
    ASSERT_TRUE(execResult.is_good()) << "graph.execute failed: " << execResult.get_message();
    ASSERT_EQ(hipStreamSynchronize(stream.get()), hipSuccess);
    ASSERT_EQ(hipMemcpy(hO.data(), dO, qBytes, hipMemcpyDeviceToHost), hipSuccess);

    std::vector<float> reference(qElems, 0.0F);
    referenceSdpaForward(hQ,
                         hK,
                         hV,
                         reference,
                         shape.batch,
                         shape.numQueryHeads,
                         shape.numKvHeads,
                         shape.seqLen,
                         shape.headDim);

    float maxAbsDiff = 0.0F;
    std::size_t bad = 0;
    constexpr float ATOL = 2e-2F;
    for(std::size_t i = 0; i < qElems; ++i)
    {
        const float diff = std::abs(static_cast<float>(hO[i]) - reference[i]);
        maxAbsDiff = std::max(maxAbsDiff, diff);
        if(diff > ATOL)
        {
            ++bad;
        }
    }
    EXPECT_EQ(bad, 0U) << "rocke SDPA output diverged from reference for shape '" << shape.name
                       << "'; max_abs=" << maxAbsDiff << " bad=" << bad << "/" << qElems
                       << " atol=" << ATOL;

    ASSERT_EQ(hipFree(dQ), hipSuccess);
    ASSERT_EQ(hipFree(dK), hipSuccess);
    ASSERT_EQ(hipFree(dV), hipSuccess);
    ASSERT_EQ(hipFree(dO), hipSuccess);
    if(workspace != nullptr)
    {
        ASSERT_EQ(hipFree(workspace), hipSuccess);
    }
}

INSTANTIATE_TEST_SUITE_P(ShippedInstances,
                         RockeClientNumericParity,
                         ::testing::Values(BASELINE_SHAPE, D128_SHAPE, GQA_SHAPE),
                         [](const ::testing::TestParamInfo<SdpaShape>& info) {
                             return info.param.name;
                         });

#endif // HIPDNN_ENABLE_SDPA
