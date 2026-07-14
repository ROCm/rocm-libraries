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
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
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
        Tensor<float> yTensor(dims);
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

        // Deliver the host scalar as a HOST pointer in the variant pack, matching
        // the frontend contract for a pure runtime pass-by-value tensor.
        float hostEpsilon = hostEpsilonValue;
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[x->get_uid()] = xTensor.memory().deviceData();
        variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
        variantPack[y->get_uid()] = yTensor.memory().deviceData();
        variantPack[epsilon->get_uid()] = &hostEpsilon;

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
