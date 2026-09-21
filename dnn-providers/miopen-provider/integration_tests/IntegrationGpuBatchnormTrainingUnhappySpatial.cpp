// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
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

/// Describes a single invalid BatchNorm training spatial configuration.
struct UnhappyBnTrainingSpatialCase
{
    std::vector<int64_t> dims;
    const char* name;
};

/// Builds a minimal BatchNorm training graph that should fail when
/// batch × spatial ≤ 1.
Graph makeUnhappySpatialGraph(const UnhappyBnTrainingSpatialCase& tc)
{
    Graph g;

    g.set_name("IntegrationGpuBatchnormTrainingUnhappySpatial");
    g.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    const auto& dims = tc.dims;

    // Derived channel dimensions
    const auto cDims = getDerivedShape(dims);

    // Input tensor
    auto x = std::make_shared<TensorAttributes>(
        makeTensorAttributes("x", DataType::FLOAT, dims, generateStrides(dims)));

    // Required BN parameters
    auto scale = std::make_shared<TensorAttributes>(
        makeTensorAttributes("scale", DataType::FLOAT, cDims, generateStrides(cDims)));

    auto bias = std::make_shared<TensorAttributes>(
        makeTensorAttributes("bias", DataType::FLOAT, cDims, generateStrides(cDims)));

    // Forward BatchNorm training
    const BatchnormAttributes bn;

    const auto bnOutputs = g.batchnorm(x, scale, bias, bn);

    // Output tensor
    const auto& y = bnOutputs[0];
    y->set_output(true);

    return g;
}

// Test fixture
class IntegrationGpuBatchnormTrainingUnhappySpatial
    : public ::testing::TestWithParam<UnhappyBnTrainingSpatialCase>
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);

        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }

        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }
};

} // anonymous namespace

// Test body
TEST_P(IntegrationGpuBatchnormTrainingUnhappySpatial, RejectsInsufficientSpatial)
{
    const auto& tc = GetParam();

    auto graph = makeUnhappySpatialGraph(tc);
    auto result = graph.build(_handle);

    // Must fail
    EXPECT_NE(result.code, ErrorCode::OK) << "Unexpected success. err_msg: " << result.err_msg;

    // Must provide error message
    EXPECT_FALSE(result.err_msg.empty());
}

// Test cases
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuBatchnormTrainingUnhappySpatial,
    ::testing::Values(UnhappyBnTrainingSpatialCase{{1, 4, 1, 1}, "SingleElement"},
                      UnhappyBnTrainingSpatialCase{{0, 4, 8, 8}, "ZeroBatch"},
                      UnhappyBnTrainingSpatialCase{{2, 4, 0, 8}, "ZeroHeight"},
                      UnhappyBnTrainingSpatialCase{{2, 4, 8, 0}, "ZeroWidth"},
                      UnhappyBnTrainingSpatialCase{{2, 4, 0, 0}, "ZeroSpatial"},
                      UnhappyBnTrainingSpatialCase{{0, 4, 1, 1}, "ZeroBatchSingleSpatial"},
                      UnhappyBnTrainingSpatialCase{{2, 4, 1}, "Rank3SingleSpatial"}),
    [](const ::testing::TestParamInfo<UnhappyBnTrainingSpatialCase>& info) {
        return std::string(info.param.name);
    });
