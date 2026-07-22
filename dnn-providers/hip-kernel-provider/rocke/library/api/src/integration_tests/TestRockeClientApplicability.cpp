// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ABI-level integration test: proves the rocke-client engine's Phase-1 no-op
// holds end-to-end through the loaded plugin's C ABI.
//
// The rocke-client plugin in Phase 1 carries an EMPTY AOT catalog. Its
// RockeClientEngine::isApplicable therefore returns false for EVERY graph,
// including a structurally valid SDPA forward graph. This test crosses the
// plugin DSO boundary (unlike the in-process unit test in TestRockeClientEngine)
// by driving the full public hipDNN C ABI call sequence:
//
//   hipdnnSetEnginePluginPaths_ext     -- load only rocke-client (no other engine)
//   hipdnnCreate + hipStreamCreate     -- handle + stream for device detection
//   graph.is_supported_ext(handle)     -- triggers:
//       build_operation_graph()
//         -> hipdnnBackendFinalize (graph descriptor; reads device from stream)
//       engine heuristic finalize (FALLBACK mode)
//         -> calls into the loaded rocke-client DSO via the plugin ABI
//         -> RockeClientEngine::isApplicable (empty catalog -> false)
//       count applicable engine configs (0)
//       -> returns Error (not good)
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
#include <filesystem>
#include <string>

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

// Builds a minimal fp16, 4D SDPA forward graph:
//   Q/K/V/O: [batch=2, heads=8, seq_len=32, head_dim=64], fp16 I/O.
// Uses the same shape/stride convention as FrontendGraphFactory::createSdpaForwardGraph().
hipdnn_frontend::graph::Graph buildMinimalSdpaForwardGraph()
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;

    const std::vector<int64_t> qkvDims = {2, 8, 32, 64};
    const std::vector<int64_t> qkvStrides = hipdnn_data_sdk::utilities::generateStrides(qkvDims);

    Graph graph;
    graph.set_name("RockeClientApplicabilityTest_SdpaFwd")
        .set_io_data_type(DataType::HALF)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto qAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("Q", qkvDims, qkvStrides));
    auto kAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("K", qkvDims, qkvStrides));
    auto vAttr = std::make_shared<TensorAttributes>(makeTensorAttributes("V", qkvDims, qkvStrides));

    const SdpaAttributes sdpaAttrs;
    auto [oAttr, statsAttr] = graph.sdpa(qAttr, kAttr, vAttr, sdpaAttrs);
    oAttr->set_output(true);

    return graph;
}

#endif // HIPDNN_ENABLE_SDPA

} // namespace

#ifdef HIPDNN_ENABLE_SDPA

// Asserts that the rocke-client engine declines a structurally valid SDPA
// forward graph through the full public C ABI — i.e. the Phase-1 empty-catalog
// no-op crosses the plugin DSO boundary, not just the in-process engine object.
//
// The test loads ONLY the rocke-client DSO so no other engine can mask the
// result.  is_supported_ext uses HeuristicMode::FALLBACK (built-in; no
// separate heuristic plugin required), which asks each loaded engine for
// applicable configs.  With an empty AOT catalog the rocke-client engine
// reports zero configs, making is_supported_ext return !good.
TEST(TestRockeClientApplicability, RockeClientDeclinesValidSdpaForwardGraphThroughAbi)
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

    // Build a structurally valid SDPA forward graph (fp16 I/O, 4D tensors).
    // This graph would be a candidate for the rocke-client engine once the
    // Phase-2 AOT catalog lands; in Phase 1 the catalog is empty.
    hipdnn_frontend::graph::Graph graph = buildMinimalSdpaForwardGraph();

    // is_supported_ext crosses the plugin DSO boundary:
    //   1. build_operation_graph() -> hipdnnBackendFinalize (graph descriptor)
    //   2. Engine heuristic (FALLBACK mode) finalize:
    //        -> loaded rocke-client DSO: RockeClientEngine::isApplicable()
    //        -> returns false (empty AOT catalog, Phase 1)
    //   3. Zero applicable engine configs -> is_supported_ext returns !good.
    //
    // This is the end-to-end ABI assertion: the no-op propagates through the
    // DSO boundary, not just in-process object wiring.
    const auto result = graph.is_supported_ext(handle.get());
    EXPECT_FALSE(result.is_good())
        << "rocke-client must decline all SDPA graphs in Phase 1 (empty AOT catalog); "
           "is_supported_ext returned success unexpectedly. "
           "Error: "
        << result.get_message();
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
        << "rocke-client must decline override-shape graphs (unsupported in Phase 1); "
           "is_supported_ext returned success unexpectedly. Error: "
        << result.get_message();
}

#endif // HIPDNN_ENABLE_SDPA
