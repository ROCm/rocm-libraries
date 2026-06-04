// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <optional>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

struct EngineFilteringTestCase
{
    std::string description;
    std::optional<int64_t> preferredEngineId;
    std::optional<bool> shouldSucceed;

    friend std::ostream& operator<<(std::ostream& os, const EngineFilteringTestCase& tc)
    {
        os << "EngineFilteringTestCase{description: " << tc.description
           << ", preferred_engine_id: ";
        if(tc.preferredEngineId.has_value())
        {
            os << tc.preferredEngineId.value();
        }
        else
        {
            os << "none";
        }
        os << ", should_succeed: ";
        if(tc.shouldSucceed.has_value())
        {
            os << (tc.shouldSucceed.value() ? "true" : "false");
        }
        else
        {
            os << "none";
        }
        os << "}";
        return os;
    }
};

class IntegrationGraphEngineFiltering : public ::testing::TestWithParam<EngineFilteringTestCase>
{
protected:
    template <typename DataType>
    struct SimpleTensorBundle
    {
        SimpleTensorBundle(const std::vector<int64_t>& dims)
            : xTensor(Tensor<DataType>(dims))
            , yTensor(Tensor<DataType>(dims))
        {
            xTensor.fillWithValue(static_cast<DataType>(1.0f));
            yTensor.fillWithValue(static_cast<DataType>(0.0f));
        }

        Tensor<DataType> xTensor;
        Tensor<DataType> yTensor;
    };

    // This suite verifies preferred_engine_id behavior, which the frontend
    // resolves as a post-hoc reorder of the heuristic-ranked engine configs
    // (see Graph::initializeEngineConfig). The HIPDNN_HEUR_CONFIG_PATH
    // env knob lives in the SelectionHeuristic::Config built-in instead. We
    // only need to chain test_good_heuristic_plugin so the heuristic loop has
    // a ranked list to reorder against.
    static void SetUpTestSuite()
    {
        const std::array<const char*, 1> heuristicPaths
            = {hipdnn_tests::plugin_constants::testGoodHeuristicPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                      heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        sPolicyOrderEnv.emplace("HIPDNN_HEUR_POLICY_ORDER",
                                hipdnn_tests::plugin_constants::testGoodHeuristicPolicyName());
    }

    static void TearDownTestSuite()
    {
        sPolicyOrderEnv.reset();
        const std::array<const char*, 1> heuristicPaths
            = {hipdnn_tests::plugin_constants::testGoodHeuristicPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetHeuristicPluginPaths_ext(
                      heuristicPaths.size(), heuristicPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
    }

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
    }

    static hipdnnHandle_t createHandle()
    {
        // Load both plugins once for all tests
        const std::array<const char*, 2> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str(),
               hipdnn_tests::plugin_constants::testExecuteFailsPluginPath().c_str()};

        EXPECT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        hipdnnHandle_t handle = nullptr;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        return handle;
    }

    static std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>
        sPolicyOrderEnv;

    static std::shared_ptr<Graph> createSimplePointwiseGraph(const std::string& graphName,
                                                             const std::vector<int64_t>& dims)
    {
        auto graph = std::make_shared<Graph>();
        graph->set_name(graphName)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(1)
            .set_name("X")
            .set_dim(dims)
            .set_stride({dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1})
            .set_data_type(DataType::FLOAT);

        PointwiseAttributes attrs;
        attrs.set_name("relu_node");
        attrs.set_mode(PointwiseMode::RELU_FWD);

        auto y = graph->pointwise(x, attrs);
        y->set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

        return graph;
    }

    void runTest()
    {
        const auto& testCase = GetParam();

        _handle = createHandle();

        const std::vector<int64_t> dims = {1, 3, 4, 4};
        SimpleTensorBundle<float> tensorBundle(dims);

        auto graph = createSimplePointwiseGraph("EngineFilteringTest", dims);

        // Set preferred engine ID if specified
        if(testCase.preferredEngineId.has_value())
        {
            graph->set_preferred_engine_id_ext(testCase.preferredEngineId);
        }

        auto result = graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // Capture the heuristic-ranked engine list before plan creation. The
        // preferred-engine setter is a post-hoc reorder over this list, so when
        // the preferred ID isn't present, execute() must follow rankedEngineIds[0].
        std::vector<int64_t> rankedEngineIds;
        result = graph->get_ranked_engine_ids(rankedEngineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_FALSE(rankedEngineIds.empty());

        result = graph->create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::unordered_map<int64_t, void*> variantPack;
        variantPack[1] = tensorBundle.xTensor.memory().deviceData();
        variantPack[2] = tensorBundle.yTensor.memory().deviceData();

        result = graph->execute(_handle, variantPack, nullptr);

        if(testCase.shouldSucceed.has_value() && testCase.shouldSucceed.value())
        {
            ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        }
        else if(testCase.shouldSucceed.has_value() && !testCase.shouldSucceed.value())
        {
            ASSERT_NE(result.code, ErrorCode::OK) << "Execute should have failed";
        }
        else
        {
            // No fixed expectation: derive it from the ranked list. For the
            // nonexistent-preferred-ID case, confirm the ID is not among the
            // candidates (so we're actually exercising the fallback path),
            // then assert execute() outcome matches the heuristic's top pick.
            ASSERT_TRUE(testCase.preferredEngineId.has_value());
            ASSERT_EQ(std::find(rankedEngineIds.begin(),
                                rankedEngineIds.end(),
                                testCase.preferredEngineId.value()),
                      rankedEngineIds.end())
                << "Nonexistent preferred engine ID unexpectedly found among candidates";

            const int64_t failingEngineId
                = hipdnn_tests::plugin_constants::engineId<ExecuteFailsPlugin>();
            if(rankedEngineIds.front() == failingEngineId)
            {
                ASSERT_NE(result.code, ErrorCode::OK)
                    << "Top-ranked engine is the failing plugin; execute should have failed";
            }
            else
            {
                ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
            }
        }
    }

private:
    hipdnnHandle_t _handle = nullptr;
};

std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>
    IntegrationGraphEngineFiltering::sPolicyOrderEnv;

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationGraphEngineFiltering,
    ::testing::Values(
        EngineFilteringTestCase{"PreferGoodPluginExplicitly",
                                hipdnn_tests::plugin_constants::engineId<GoodPlugin>(),
                                true},
        EngineFilteringTestCase{"PreferNonExistentEngineId", 999999, std::nullopt},
        EngineFilteringTestCase{"PreferExecuteFailsPlugin",
                                hipdnn_tests::plugin_constants::engineId<ExecuteFailsPlugin>(),
                                false}),
    [](const ::testing::TestParamInfo<EngineFilteringTestCase>& info) {
        return info.param.description;
    });

TEST_P(IntegrationGraphEngineFiltering, EngineSelection)
{
    runTest();
}
