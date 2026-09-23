// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <optional>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
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

// Shared plumbing for the engine-filtering suites: the chained heuristic
// plugin, the engine plugin load set, and the simple pointwise graph the
// filters are applied to.
class EngineFilteringTestBase : public ::testing::Test
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

    // The engine plugins every test in these suites needs: one that executes
    // successfully and one that fails at execute.
    static std::vector<const char*> defaultEnginePluginPaths()
    {
        return {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str(),
                hipdnn_tests::plugin_constants::testExecuteFailsPluginPath().c_str()};
    }

    static hipdnnHandle_t createHandle(const std::vector<const char*>& pluginPaths)
    {
        EXPECT_EQ(hipdnnSetEnginePluginPaths_ext(
                      pluginPaths.size(), pluginPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        hipdnnHandle_t handle = nullptr;
        EXPECT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

        return handle;
    }

    static hipdnnHandle_t createHandle()
    {
        return createHandle(defaultEnginePluginPaths());
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

    hipdnnHandle_t _handle = nullptr;
};

class IntegrationGraphEngineFiltering
    : public EngineFilteringTestBase,
      public ::testing::WithParamInterface<EngineFilteringTestCase>
{
protected:
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
};

// Engine filtering driven by engine names rather than raw engine IDs. The names
// under test are plugin-supplied and absent from the built-in name registry.
class IntegrationGraphEngineNameFiltering : public EngineFilteringTestBase
{
protected:
    static const std::string& pluginEngineName()
    {
        static const std::string s_pluginEngineName
            = hipdnn_tests::plugin_constants::K_EXECUTE_FAILS_PLUGIN_ENGINE_NAME;
        return s_pluginEngineName;
    }

    // The engine name of the plugin whose engine id is the hash of that same
    // name, so a name-hashing filter resolves it to a loaded engine.
    static const std::string& hashedPluginEngineName()
    {
        static const std::string s_hashedPluginEngineName
            = hipdnn_tests::plugin_constants::K_HASHED_NAME_PLUGIN_ENGINE_NAME;
        return s_hashedPluginEngineName;
    }

    static std::vector<const char*> pluginPathsWithHashedNameEngine()
    {
        auto paths = defaultEnginePluginPaths();
        paths.push_back(hipdnn_tests::plugin_constants::testHashedNamePluginPath().c_str());
        return paths;
    }

    std::shared_ptr<Graph> buildGraph(const std::string& graphName,
                                      const std::vector<int64_t>& dims)
    {
        return buildGraph(graphName, dims, defaultEnginePluginPaths());
    }

    std::shared_ptr<Graph> buildGraph(const std::string& graphName,
                                      const std::vector<int64_t>& dims,
                                      const std::vector<const char*>& pluginPaths)
    {
        _handle = createHandle(pluginPaths);

        auto graph = createSimplePointwiseGraph(graphName, dims);

        auto result = graph->validate();
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        return graph;
    }
};

std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>
    EngineFilteringTestBase::sPolicyOrderEnv;

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

// An engine is removed from consideration by the name its plugin reports.
TEST_F(IntegrationGraphEngineNameFiltering, DeselectByPluginSuppliedEngineName)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::isEngineNameRegistered(pluginEngineName()));

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = buildGraph("DeselectByPluginSuppliedEngineName", dims);

    std::vector<EngineConfigInfo> configs;
    auto result = graph->get_engine_configs(_handle, configs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t failingEngineId = hipdnn_tests::plugin_constants::engineId<ExecuteFailsPlugin>();
    const auto failingConfig
        = std::find_if(configs.begin(), configs.end(), [failingEngineId](const auto& config) {
              return config.engineId == failingEngineId;
          });
    ASSERT_NE(failingConfig, configs.end()) << "Execute-fails plugin engine not among candidates";
    EXPECT_EQ(failingConfig->engineName, pluginEngineName());

    // A plugin only keeps an engine whose ID is the hash of the name it reports.
    ASSERT_EQ(hipdnn_data_sdk::utilities::engineNameToId(pluginEngineName()), failingEngineId);

    // create_execution_plans() resets the filter set, so the deselection is
    // applied afterwards.
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = graph->get_execution_plan_count();
    ASSERT_GT(planCount, 1) << "Need a second plan to show the bar is engine-specific";

    int64_t failingPlanIndex = -1;
    int64_t otherPlanIndex = -1;
    for(int64_t index = 0; index < planCount; ++index)
    {
        std::string planName;
        result = graph->get_plan_name_at_index(_handle, index, planName);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(planName == pluginEngineName())
        {
            failingPlanIndex = index;
        }
        else if(otherPlanIndex < 0)
        {
            otherPlanIndex = index;
        }
    }
    ASSERT_GE(failingPlanIndex, 0) << "No plan reported the plugin-supplied engine name";
    ASSERT_GE(otherPlanIndex, 0) << "No plan from another engine to use as a control";

    const Graph& chained = graph->deselect_engines(std::vector<std::string>{pluginEngineName()});
    EXPECT_EQ(&chained, graph.get());

    result = graph->build_plan_at_index(failingPlanIndex);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("barred"), std::string::npos) << result.err_msg;

    result = graph->build_plan_at_index(otherPlanIndex);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// A name matching none of the candidate engines bars nothing and leaves every
// plan buildable.
TEST_F(IntegrationGraphEngineNameFiltering, DeselectByUnmatchedEngineNameBarsNothing)
{
    const std::string unmatchedName = "NO_SUCH_ENGINE_NAME_AICK1901";
    ASSERT_FALSE(hipdnn_data_sdk::utilities::isEngineNameRegistered(unmatchedName));

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = buildGraph("DeselectByUnmatchedEngineName", dims);

    auto result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = graph->get_execution_plan_count();
    ASSERT_GT(planCount, 0);

    graph->deselect_engines(std::vector<std::string>{unmatchedName});

    for(int64_t index = 0; index < planCount; ++index)
    {
        result = graph->build_plan_at_index(index);
        EXPECT_EQ(result.err_msg.find("barred"), std::string::npos)
            << "Plan " << index << " should not be barred by an unmatched name";
    }
}

// create_execution_plans() resets the name filter as well as the engine ID filter.
TEST_F(IntegrationGraphEngineNameFiltering, CreateExecutionPlansResetsNameFilter)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = buildGraph("CreateExecutionPlansResetsNameFilter", dims);

    graph->deselect_engines(std::vector<std::string>{pluginEngineName()});

    auto result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = graph->get_execution_plan_count();
    ASSERT_GT(planCount, 0);

    int64_t failingPlanIndex = -1;
    for(int64_t index = 0; index < planCount; ++index)
    {
        std::string planName;
        result = graph->get_plan_name_at_index(_handle, index, planName);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(planName == pluginEngineName())
        {
            failingPlanIndex = index;
            break;
        }
    }
    ASSERT_GE(failingPlanIndex, 0) << "No plan reported the plugin-supplied engine name";

    result = graph->build_plan_at_index(failingPlanIndex);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// A preference expressed as a plugin-supplied name selects that engine.
TEST_F(IntegrationGraphEngineNameFiltering, PreferByPluginSuppliedEngineName)
{
    const int64_t failingEngineId = hipdnn_tests::plugin_constants::engineId<ExecuteFailsPlugin>();
    ASSERT_EQ(hipdnn_data_sdk::utilities::engineNameToId(pluginEngineName()), failingEngineId);

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    _handle = createHandle(defaultEnginePluginPaths());

    auto graph = createSimplePointwiseGraph("PreferByPluginSuppliedEngineName", dims);
    graph->set_preferred_engine_id_ext(pluginEngineName());

    // The hashed ID is stored too, so the preference is observable before the
    // graph is built.
    EXPECT_EQ(
        graph->get_preferred_engine_id_ext(),
        std::optional<int64_t>{hipdnn_data_sdk::utilities::engineNameToId(pluginEngineName())});

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t selectedEngineId = -1;
    result = graph->get_execution_plan_engine_id(selectedEngineId);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(selectedEngineId, failingEngineId);

    std::string activePlanName;
    result = graph->get_plan_name(_handle, activePlanName);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(activePlanName, pluginEngineName());
}

// Preferring an engine ID discards a preference previously given as a name.
TEST_F(IntegrationGraphEngineNameFiltering, PreferByIdAfterNameDiscardsTheName)
{
    const int64_t goodEngineId = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    const int64_t failingEngineId = hipdnn_tests::plugin_constants::engineId<ExecuteFailsPlugin>();
    ASSERT_NE(goodEngineId, failingEngineId);

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    _handle = createHandle(defaultEnginePluginPaths());

    auto graph = createSimplePointwiseGraph("PreferByIdAfterNameDiscardsTheName", dims);
    graph->set_preferred_engine_id_ext(pluginEngineName());
    graph->set_preferred_engine_id_ext(std::optional<int64_t>{goodEngineId});

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t selectedEngineId = -1;
    result = graph->get_execution_plan_engine_id(selectedEngineId);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_EQ(selectedEngineId, goodEngineId);
}

// The plan belonging to a plugin engine is identified by the plugin-supplied name
// and then barred by engine ID.
TEST_F(IntegrationGraphEngineNameFiltering, DeselectBarsPluginEnginePlan)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = buildGraph("DeselectBarsPluginEnginePlan", dims);

    auto result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = graph->get_execution_plan_count();
    ASSERT_GT(planCount, 0);

    int64_t namedPlanIndex = -1;
    for(int64_t index = 0; index < planCount; ++index)
    {
        std::string planName;
        result = graph->get_plan_name_at_index(_handle, index, planName);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(planName == pluginEngineName())
        {
            namedPlanIndex = index;
            break;
        }
    }
    ASSERT_GE(namedPlanIndex, 0) << "No plan reported the plugin-supplied engine name";

    graph->deselect_engines(
        std::vector<int64_t>{hipdnn_tests::plugin_constants::engineId<ExecuteFailsPlugin>()});

    result = graph->build_plan_at_index(namedPlanIndex);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("barred"), std::string::npos) << result.err_msg;
}

// A plugin whose engine ID is the hash of its name -- the identity
// HIPDNN_REGISTER_ENGINE gives production plugins -- is deselectable by that name.
TEST_F(IntegrationGraphEngineNameFiltering, DeselectByHashedPluginSuppliedEngineName)
{
    // Guards against drift between the name and the hardcoded engine ID literal in
    // TestPluginEngineIdMap.hpp; without the identity this test covers nothing new.
    ASSERT_EQ(hipdnn_tests::plugin_constants::engineId<HashedNamePlugin>(),
              hipdnn_data_sdk::utilities::engineNameToId(hashedPluginEngineName()));

    ASSERT_FALSE(hipdnn_data_sdk::utilities::isEngineNameRegistered(hashedPluginEngineName()));

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = buildGraph(
        "DeselectByHashedPluginSuppliedEngineName", dims, pluginPathsWithHashedNameEngine());

    // Confirm the engine carrying this name really is the hashed-name plugin's,
    // so the plan located below cannot belong to one of the other two plugins.
    std::vector<EngineConfigInfo> configs;
    auto result = graph->get_engine_configs(_handle, configs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t hashedEngineId = hipdnn_tests::plugin_constants::engineId<HashedNamePlugin>();
    const auto hashedConfig
        = std::find_if(configs.begin(), configs.end(), [hashedEngineId](const auto& config) {
              return config.engineId == hashedEngineId;
          });
    ASSERT_NE(hashedConfig, configs.end()) << "Hashed-name plugin engine not among candidates";
    ASSERT_EQ(hashedConfig->engineName, hashedPluginEngineName());

    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = graph->get_execution_plan_count();
    ASSERT_GT(planCount, 1) << "Need a second plan to show the bar is engine-specific";

    int64_t hashedPlanIndex = -1;
    int64_t otherPlanIndex = -1;
    for(int64_t index = 0; index < planCount; ++index)
    {
        std::string planName;
        result = graph->get_plan_name_at_index(_handle, index, planName);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(planName == hashedPluginEngineName())
        {
            hashedPlanIndex = index;
        }
        else if(otherPlanIndex < 0)
        {
            otherPlanIndex = index;
        }
    }
    ASSERT_GE(hashedPlanIndex, 0) << "No plan reported the hashed plugin-supplied engine name";
    ASSERT_GE(otherPlanIndex, 0) << "No plan from another engine to use as a control";

    const Graph& chained
        = graph->deselect_engines(std::vector<std::string>{hashedPluginEngineName()});
    EXPECT_EQ(&chained, graph.get());

    result = graph->build_plan_at_index(hashedPlanIndex);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("barred"), std::string::npos) << result.err_msg;

    result = graph->build_plan_at_index(otherPlanIndex);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}
