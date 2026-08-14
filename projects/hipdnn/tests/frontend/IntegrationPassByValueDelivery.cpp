// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Frontend integration coverage for RFC-0016 runtime pass-by-value host-scalar
// DELIVERY. The version-filter tests (IntegrationPassByValueVersionFilter) prove
// a 1.2.0 plugin is admitted for a runtime-pbv graph; they stop at
// create_execution_plans and never execute, so they cannot show what value
// actually reaches the plugin. This suite closes that gap by executing the graph
// against a recorder plugin that, using the shared plugin SDK helpers
// makeScalarOperand()/resolveScalarOperand() (the exact path MIOpen and
// hip-kernel use), records the scalar it resolves from the caller's
// device_buffers. The test then asserts the EXACT host scalar delivered in the
// variant pack reached the plugin boundary at the correct tensor uid.
//
// This is a direct delivery assertion, complementary to the provider integration
// suites (which infer delivery from numeric kernel output on GPU): here the value
// is read back byte-exact, independent of any real kernel, so it also validates
// the SDK helpers inside a standalone plugin.
//
// GPU-less environments skip through the common test utility. A GPU is required
// only to allocate the real device buffers for the graph's ordinary tensors; the
// recorder plugin runs no kernel, so this is provider/arch independent.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/GraphExecuteTestKit.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <test_plugins/TestPassByValueRecorder.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

class IntegrationPassByValueDelivery : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);

        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testPassByValueRecorderPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

        // Resolve the exact loaded plugin path so the recorder reads the same
        // library instance the backend executed.
        std::vector<std::filesystem::path> loadedPaths;
        const auto pathResult = getLoadedEnginePluginPaths(_handle, loadedPaths);
        ASSERT_EQ(pathResult.code, ErrorCode::OK) << pathResult.err_msg;
        ASSERT_FALSE(loadedPaths.empty());
        _recorder = std::make_unique<hipdnn_tests::TestPassByValueRecorder>(loadedPaths.front());
        _recorder->reset();
    }

    void TearDown() override
    {
        _recorder.reset();
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }

    // Build + execute an RMSNorm (inference) graph whose epsilon is a runtime
    // pass-by-value parameter delivered as a HOST scalar at execute, and return
    // the epsilon tensor uid so the caller can assert on what the recorder saw.
    int64_t runWithRuntimeEpsilon(float hostEpsilonValue)
    {
        const std::vector<int64_t> dims = {2, 3, 8, 8};
        std::vector<int64_t> scaleDims = dims;
        scaleDims[0] = 1;

        Graph graph;
        graph.set_name("PassByValueDelivery")
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        Tensor<float> xTensor(dims);
        Tensor<float> scaleTensor(scaleDims);
        xTensor.fillWithValue(1.0f);
        scaleTensor.fillWithValue(1.0f);

        auto x = Graph::tensor(makeTensorAttributes("X", DataType::FLOAT, xTensor));
        auto scale = Graph::tensor(makeTensorAttributes("scale", DataType::FLOAT, scaleTensor));

        // Pure runtime-user-supplied epsilon: runtime flag set, no baked value,
        // so the provider MUST read it from device_buffers at execute.
        auto epsilon = std::make_shared<TensorAttributes>();
        epsilon->set_name("epsilon")
            .set_dim({1})
            .set_stride({1})
            .set_data_type(DataType::FLOAT)
            .set_as_runtime_parameter();

        RMSNormAttributes attrs;
        attrs.set_name("rmsnorm");
        attrs.set_epsilon(epsilon);
        attrs.set_forward_phase(NormFwdPhase::INFERENCE);

        auto outputs = graph.rmsnorm(x, scale, std::move(attrs));
        const auto& y = outputs[0];
        y->set_output(true).set_data_type(DataType::FLOAT);

        EXPECT_TRUE(epsilon->get_is_runtime_pass_by_value());

        auto result = graph.validate();
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.build(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // Exercise the shared frontend harness: ordinary tensors use device
        // storage while the runtime pass-by-value epsilon uses host storage.
        const hipdnn_test_sdk::utilities::GraphTensorBundle tensorBundle(graph, hostEpsilonValue);
        auto variantPack = tensorBundle.variantPack();

        result = graph.execute(_handle, variantPack, nullptr);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        EXPECT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

        return epsilon->get_uid();
    }

    hipdnnHandle_t _handle = nullptr;
    std::unique_ptr<hipdnn_tests::TestPassByValueRecorder> _recorder;
};

} // namespace

// The exact host scalar delivered in the variant pack must reach the plugin at
// the runtime pass-by-value tensor's uid. Proves end-to-end host->plugin
// delivery through the SDK resolveScalarOperand() path, not just admission.
TEST_F(IntegrationPassByValueDelivery, HostScalarReachesPluginAtCorrectUid)
{
    const float epsilonValue = 0.0009765625f; // exactly representable in float
    const int64_t epsilonUid = runWithRuntimeEpsilon(epsilonValue);

    ASSERT_EQ(_recorder->count(), 1u)
        << "Recorder should have resolved exactly one runtime pass-by-value scalar";
    EXPECT_EQ(_recorder->uidAt(0), epsilonUid)
        << "Recorded scalar uid must match the epsilon tensor uid";

    const auto delivered = _recorder->valueForUid(epsilonUid);
    ASSERT_TRUE(delivered.has_value()) << "No scalar recorded for the epsilon uid";
    EXPECT_DOUBLE_EQ(*delivered, static_cast<double>(epsilonValue))
        << "Plugin received a different scalar than was delivered in the variant pack";
}

// A second, materially different value must be seen exactly, proving the plugin
// reads the delivered pointer each execute rather than caching a first value.
TEST_F(IntegrationPassByValueDelivery, DifferentHostScalarIsDeliveredExactly)
{
    const float epsilonValue = 0.5f;
    const int64_t epsilonUid = runWithRuntimeEpsilon(epsilonValue);

    ASSERT_EQ(_recorder->count(), 1u);
    const auto delivered = _recorder->valueForUid(epsilonUid);
    ASSERT_TRUE(delivered.has_value());
    EXPECT_DOUBLE_EQ(*delivered, static_cast<double>(epsilonValue));
}

// The frontend contract requires the caller to deliver a host scalar for a
// pure runtime-user-supplied tensor via its uid in the variant pack. Omitting
// it must not silently execute with a stale/zero value: resolveScalarOperand()
// throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) inside the
// plugin (proven at the SDK level by
// TestRuntimePassByValue.ThrowsIfPureRuntimeUserSuppliedBufferMissing); this
// test closes the remaining seam by proving that failure actually propagates
// out through Graph::execute() to the caller as an Error, rather than being
// swallowed or crashing.
TEST_F(IntegrationPassByValueDelivery, MissingRuntimeScalarFailsExecute)
{
    const std::vector<int64_t> dims = {2, 3, 8, 8};
    std::vector<int64_t> scaleDims = dims;
    scaleDims[0] = 1;

    Graph graph;
    graph.set_name("PassByValueMissingScalar")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    Tensor<float> xTensor(dims);
    Tensor<float> scaleTensor(scaleDims);
    Tensor<float> yTensor(dims);
    xTensor.fillWithValue(1.0f);
    scaleTensor.fillWithValue(1.0f);

    auto x = Graph::tensor(makeTensorAttributes("X", DataType::FLOAT, xTensor));
    auto scale = Graph::tensor(makeTensorAttributes("scale", DataType::FLOAT, scaleTensor));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("epsilon")
        .set_dim({1})
        .set_stride({1})
        .set_data_type(DataType::FLOAT)
        .set_as_runtime_parameter();

    RMSNormAttributes attrs;
    attrs.set_name("rmsnorm");
    attrs.set_epsilon(epsilon);
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);

    auto outputs = graph.rmsnorm(x, scale, std::move(attrs));
    const auto& y = outputs[0];
    y->set_output(true).set_data_type(DataType::FLOAT);

    ASSERT_TRUE(epsilon->get_is_runtime_pass_by_value());

    auto result = graph.validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph.build(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Deliberately omit epsilon->get_uid() from the variant pack.
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();

    result = graph.execute(_handle, variantPack, nullptr);
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.err_msg.find(std::to_string(epsilon->get_uid())), std::string::npos)
        << "Expected the missing uid in the surfaced error message, got: " << result.err_msg;

    EXPECT_EQ(_recorder->count(), 0u)
        << "No scalar should have been recorded once resolution failed";
}

// ============================================================================
// Frontend lift round-trip: proves a runtime pass-by-value tensor's
// classification survives the REAL serialize/deserialize path through the
// frontend's own reconstruction API, not just the backend C-API descriptor
// accessors (see IntegrationGraphDescriptorApi.
// DeserializedGraphRestoresTensorRuntimePassByValueFlag for that layer) and
// not just the stub-backend unit tests in TestDescriptorHelpers.cpp. Uses
// liftGraphWithoutFinalization(), which serializes the graph to binary via
// Graph::to_binary(), builds a real backend descriptor from those bytes, and
// reconstructs a new frontend Graph via fromBackendDescriptor() -- the same
// path a caller exercises with from_compiled_plan_binary(). No handle,
// plugin, or device is needed since this never executes.
// ============================================================================

namespace
{

// Feature-named fixture (not the bare base IntegrationTestFixture) per
// hipDNN's GTest naming convention -- suite names embedding "Test" in the
// middle of the name are rejected.
class IntegrationPassByValueLiftRoundTrip : public hipdnn_tests::IntegrationTestFixture
{
};

// Builds an RMSNorm graph whose epsilon is a pure runtime-user-supplied
// pass-by-value tensor (set_as_runtime_parameter(): no baked value).
std::shared_ptr<hipdnn_tests::TestableGraphLifting> buildRuntimeEpsilonGraph()
{
    const std::vector<int64_t> dims = {2, 3, 8, 8};
    std::vector<int64_t> scaleDims = dims;
    scaleDims[0] = 1;

    auto graph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
    graph->set_name("PassByValueLiftRoundTrip")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim(dims)
        .set_stride({192, 64, 8, 1});
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2)
        .set_name("scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(scaleDims)
        .set_stride({64, 64, 8, 1});

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(3)
        .set_name("epsilon")
        .set_dim({1})
        .set_stride({1})
        .set_data_type(DataType::FLOAT)
        .set_as_runtime_parameter();

    RMSNormAttributes attrs;
    attrs.set_name("rmsnorm");
    attrs.set_epsilon(epsilon);
    attrs.set_forward_phase(NormFwdPhase::INFERENCE);

    auto outputs = graph->rmsnorm(x, scale, std::move(attrs));
    outputs[0]->set_uid(4).set_name("Y").set_output(true).set_data_type(DataType::FLOAT);

    return graph;
}

} // namespace

// The runtime pass-by-value classification set on the ORIGINAL graph's
// epsilon tensor (set_as_runtime_parameter(): runtime flag true, no baked
// value) must be readable, unchanged, off the epsilon tensor in the LIFTED
// (reconstructed-from-serialized-bytes) graph. This is the frontend-facing
// counterpart to the backend descriptor test: it proves a real caller
// reconstructing a Graph via from_compiled_plan_binary() sees the correct
// getters, not just that the backend's internal flatbuffer round-trips.
TEST_F(IntegrationPassByValueLiftRoundTrip, PreservesRuntimeClassification)
{
    auto originalGraph = buildRuntimeEpsilonGraph();

    auto liftedGraph = hipdnn_tests::liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_NE(tensorMap.count(3), 0u);
    const auto& liftedEpsilon = tensorMap[3];

    EXPECT_TRUE(liftedEpsilon->get_is_runtime_pass_by_value());
    EXPECT_FALSE(liftedEpsilon->get_pass_by_value().has_value());
    EXPECT_FALSE(liftedEpsilon->get_compile_time_constant<float>().has_value());
}

// Complementary case: a COMPILE-TIME CONSTANT epsilon (the default state,
// baked value, runtime flag clear) must also survive the lift round-trip
// with the opposite classification, proving the lift path distinguishes the
// two states rather than always reporting one.
TEST_F(IntegrationPassByValueLiftRoundTrip, PreservesCompileTimeConstant)
{
    auto graph = buildRuntimeEpsilonGraph();
    // Overwrite epsilon with a baked compile-time constant instead.
    auto tensorMap = graph->getTensorsByUid();
    ASSERT_NE(tensorMap.count(3), 0u);
    tensorMap[3]->set_value(1e-5f);

    auto liftedGraph = hipdnn_tests::liftGraphWithoutFinalization(*graph);
    ASSERT_NE(liftedGraph, nullptr);

    auto liftedTensorMap = liftedGraph->getTensorsByUid();
    ASSERT_NE(liftedTensorMap.count(3), 0u);
    const auto& liftedEpsilon = liftedTensorMap[3];

    EXPECT_FALSE(liftedEpsilon->get_is_runtime_pass_by_value());
    ASSERT_TRUE(liftedEpsilon->get_compile_time_constant<float>().has_value());
    EXPECT_FLOAT_EQ(*liftedEpsilon->get_compile_time_constant<float>(), 1e-5f);
}
