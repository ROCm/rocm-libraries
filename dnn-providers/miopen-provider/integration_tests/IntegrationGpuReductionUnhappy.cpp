// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

// Engine registration of the MIOpen reduction route (legacy miopenReduceTensor), checked end to
// end through the frontend. The registration under test is in MiopenContainer.cpp.

// Loads only the MIOpen plugin, so every ranked engine comes from the MIOpen provider.
class IntegrationGpuReductionUnhappyEngine : public ::testing::Test
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
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }
};

// fp32 ADD over the two spatial axes of a packed NCHW tensor: a graph the default MIOpen engine
// serves.
Graph makePlainAddReductionGraph()
{
    Graph graph;
    graph.set_name("IntegrationGpuReductionUnhappy");
    graph.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    const std::vector<int64_t> xDims = {2, 4, 8, 8};
    const std::vector<int64_t> yDims = {2, 4, 1, 1};
    auto x = std::make_shared<TensorAttributes>(
        makeTensorAttributes("x", DataType::FLOAT, xDims, generateStrides(xDims)));
    auto y = std::make_shared<TensorAttributes>(
        makeTensorAttributes("y", DataType::FLOAT, yDims, generateStrides(yDims)));

    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);

    graph.reduction(x, y, attrs)->set_output(true);
    return graph;
}

bool containsEngine(const std::vector<int64_t>& engineIds, int64_t engineId)
{
    return std::find(engineIds.begin(), engineIds.end(), engineId) != engineIds.end();
}

} // namespace

// MiopenContainer registers no reduction plan builder on MIOPEN_ENGINE_DETERMINISTIC, so that
// engine is never applicable to a reduction, not even one the default engine serves.
TEST_F(IntegrationGpuReductionUnhappyEngine, DeterministicEngineIsNotApplicable)
{
    auto graph = makePlainAddReductionGraph();

    auto result = graph.build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = graph.get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    // Also the only test of the reduction plan builder's registration on the default MIOPEN_ENGINE.
    EXPECT_TRUE(containsEngine(rankedEngineIds, MIOPEN_ENGINE_ID));
    EXPECT_FALSE(containsEngine(rankedEngineIds, MIOPEN_ENGINE_DETERMINISTIC_ID))
        << "MIOPEN_ENGINE_DETERMINISTIC must not claim a reduction graph";
}
