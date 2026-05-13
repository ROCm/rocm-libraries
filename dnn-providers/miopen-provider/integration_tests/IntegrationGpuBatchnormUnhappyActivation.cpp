// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <hip/hip_runtime.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

/// Describes a single invalid BatchNorm + Activation configuration.
struct UnhappyBnActivationCase
{
    PointwiseMode pointwiseMode; 
    bool isBackward;             
    const char* name;           
};

/// Builds a minimal BN - Pointwise graph that should fail at build time
/// due to unsupported activation fusion.
Graph makeGraph(const UnhappyBnActivationCase& tc)
{
    Graph g;

    g.set_name("IntegrationGpuBatchnormUnhappyActivation");
    g.set_compute_data_type(DataType::FLOAT)
     .set_intermediate_data_type(DataType::FLOAT)
     .set_io_data_type(DataType::FLOAT);

    // Valid 4D tensor (NCHW)
    const std::vector<int64_t> dims  = {2, 4, 8, 8};
    const std::vector<int64_t> cDims = getDerivedShape(dims);

    // Input
    auto X = std::make_shared<TensorAttributes>(
        makeTensorAttributes("X", DataType::FLOAT, dims, generateStrides(dims)));

    // BN parameters
    auto mean   = std::make_shared<TensorAttributes>(
        makeTensorAttributes("mean", DataType::FLOAT, cDims, generateStrides(cDims)));
    auto invVar = std::make_shared<TensorAttributes>(
        makeTensorAttributes("inv_variance", DataType::FLOAT, cDims, generateStrides(cDims)));
    auto scale  = std::make_shared<TensorAttributes>(
        makeTensorAttributes("scale", DataType::FLOAT, cDims, generateStrides(cDims)));
    auto bias   = std::make_shared<TensorAttributes>(
        makeTensorAttributes("bias", DataType::FLOAT, cDims, generateStrides(cDims)));

    std::shared_ptr<TensorAttributes> bnOut;

    if (!tc.isBackward)
    {
        // Forward BatchNorm
        BatchnormInferenceAttributes bn;
        bnOut = g.batchnorm_inference(X, mean, invVar, scale, bias, bn);
    }
    else
    {
        // Backward BatchNorm
        auto dY = std::make_shared<TensorAttributes>(
            makeTensorAttributes("dY", DataType::FLOAT, dims, generateStrides(dims)));

        BatchnormBackwardAttributes bn;

        auto bnOutputs = g.batchnorm_backward(dY, X, scale, bn);
        bnOut = bnOutputs[0]; // Select dX
    }

    // Add unsupported activation via pointwise op
    PointwiseAttributes pw;
    pw.set_mode(tc.pointwiseMode);

    auto Z = g.pointwise(bnOut, pw);
    Z->set_output(true);

    return g;
}


// Test fixture
class IntegrationGpuBatchnormUnhappyActivation
    : public ::testing::TestWithParam<UnhappyBnActivationCase>
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream    = nullptr;
    int _deviceId          = 0;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);

        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};

        ASSERT_EQ(
            hipdnnSetEnginePluginPaths_ext(
                paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
            HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if (_handle != nullptr)
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);

        if (_stream != nullptr)
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
    }
};

} // anonymous namespace

// Test body
TEST_P(IntegrationGpuBatchnormUnhappyActivation, RejectsUnsupportedActivations)
{
    const auto& tc = GetParam();

    auto graph = makeGraph(tc);
    auto result = graph.build(_handle);

    EXPECT_NE(result.code, ErrorCode::OK)
        << "Unexpected success. err_msg: " << result.err_msg;

    EXPECT_FALSE(result.err_msg.empty());

    // Error should indicate a valid failure reason
    EXPECT_TRUE(
        result.err_msg.find("activation") != std::string::npos ||
        result.err_msg.find("engine") != std::string::npos ||
        result.err_msg.find("Pointwise") != std::string::npos
    ) << "Unexpected error message: " << result.err_msg;
}

// Test cases
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormUnhappyActivation,
    ::testing::Values(
        // Forward unsupported
        UnhappyBnActivationCase{PointwiseMode::SIGMOID_FWD, false, "SigmoidFwd"},
        UnhappyBnActivationCase{PointwiseMode::TANH_FWD, false, "TanhFwd"},

        // Backward unsupported
        UnhappyBnActivationCase{PointwiseMode::SIGMOID_BWD, true, "SigmoidBwd"},
        UnhappyBnActivationCase{PointwiseMode::RELU_FWD, true, "ReluFwdInBackward"}
    ),
    [](const ::testing::TestParamInfo<UnhappyBnActivationCase>& info) {
        return std::string(info.param.name);
    });