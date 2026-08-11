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

// Engine filtering driven by engine names rather than raw engine IDs. The
// engines here come from the two loaded test plugins, so the names under test
// are plugin-supplied and absent from the frontend's built-in name registry.
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

    // Adds the hashed-name plugin to the suite's usual load set, so a graph can
    // be offered both an engine that a name filter can reach and engines it
    // must leave alone.
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

// deselect_engines(names) hashes every name with engineNameToId() and never
// consults the built-in engine name registry, so a plugin-supplied name is a
// usable filter key. The name asserted here is the one TestExecuteFailsPlugin
// reports, and it is deliberately not a registered engine name.
TEST_F(IntegrationGraphEngineNameFiltering, DeselectByPluginSuppliedEngineName)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::isEngineNameRegistered(pluginEngineName()));

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = buildGraph("DeselectByPluginSuppliedEngineName", dims);

    // The frontend surfaces the plugin-supplied name for the plugin's engine,
    // which is what makes the name available to deselect_engines() at all.
    std::vector<EngineConfigInfo> configs;
    auto result = graph->get_engine_configs(configs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t failingEngineId = hipdnn_tests::plugin_constants::engineId<ExecuteFailsPlugin>();
    const auto failingConfig
        = std::find_if(configs.begin(), configs.end(), [failingEngineId](const auto& config) {
              return config.engineId == failingEngineId;
          });
    ASSERT_NE(failingConfig, configs.end()) << "Execute-fails plugin engine not among candidates";
    EXPECT_EQ(failingConfig->engineName, pluginEngineName());

    // create_execution_plans() resets the filter set, so the deselection is
    // applied afterwards.
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // The unregistered name is accepted rather than warned about and dropped.
    const Graph& chained = graph->deselect_engines(std::vector<std::string>{pluginEngineName()});
    EXPECT_EQ(&chained, graph.get());

    // The filter key applied is the hash of the name. This plugin hardcodes its
    // engine ID instead of deriving it with HIPDNN_REGISTER_ENGINE, so that hash
    // does not coincide with its own engine ID and this call bars nothing --
    // which is precisely why the name is accepted without error here yet every
    // plan still builds. DeselectByHashedPluginSuppliedEngineName covers a
    // plugin whose ID is the hash of its name, where the bar does take effect.
    EXPECT_NE(hipdnn_data_sdk::utilities::engineNameToId(pluginEngineName()), failingEngineId);

    result = graph->build_plans(BuildPlanPolicy::ALL);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// The plan belonging to a plugin engine is identified by the plugin-supplied
// name and then barred by engine ID, covering the name-to-plan association and
// ID-keyed barring. Barring the same plan through its name alone is covered by
// DeselectByHashedPluginSuppliedEngineName.
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
        result = graph->get_plan_name_at_index(index, planName);
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

// The headline capability: a plugin engine is removed from consideration using
// nothing but the name the plugin itself supplied. deselect_engines(names)
// resolves a name by hashing it with engineNameToId(), so this only works for a
// plugin whose engine ID is that same hash -- the identity
// HIPDNN_REGISTER_ENGINE gives production plugins and test_hashed_name_plugin
// reproduces.
TEST_F(IntegrationGraphEngineNameFiltering, DeselectByHashedPluginSuppliedEngineName)
{
    // Guards against drift between the name and the hardcoded engine ID literal
    // in TestPluginEngineIdMap.hpp. Without the identity the deselection below
    // would bar nothing and the test would silently prove nothing.
    ASSERT_EQ(hipdnn_tests::plugin_constants::engineId<HashedNamePlugin>(),
              hipdnn_data_sdk::utilities::engineNameToId(hashedPluginEngineName()));

    // The name is plugin-supplied, not one the frontend's built-in registry
    // knows, which is the case deselect_engines() has to handle.
    ASSERT_FALSE(hipdnn_data_sdk::utilities::isEngineNameRegistered(hashedPluginEngineName()));

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = buildGraph(
        "DeselectByHashedPluginSuppliedEngineName", dims, pluginPathsWithHashedNameEngine());

    // Confirm the engine carrying this name really is the hashed-name plugin's,
    // so the plan located below cannot belong to one of the other two plugins.
    std::vector<EngineConfigInfo> configs;
    auto result = graph->get_engine_configs(configs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t hashedEngineId = hipdnn_tests::plugin_constants::engineId<HashedNamePlugin>();
    const auto hashedConfig
        = std::find_if(configs.begin(), configs.end(), [hashedEngineId](const auto& config) {
              return config.engineId == hashedEngineId;
          });
    ASSERT_NE(hashedConfig, configs.end()) << "Hashed-name plugin engine not among candidates";
    ASSERT_EQ(hashedConfig->engineName, hashedPluginEngineName());

    // create_execution_plans() clears the barred-engine set, so the deselection
    // has to be applied after the plans exist.
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = graph->get_execution_plan_count();
    ASSERT_GT(planCount, 1) << "Need a second plan to show the bar is engine-specific";

    int64_t hashedPlanIndex = -1;
    int64_t otherPlanIndex = -1;
    for(int64_t index = 0; index < planCount; ++index)
    {
        std::string planName;
        result = graph->get_plan_name_at_index(index, planName);
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

    // Deselect by the name string alone. No engine ID is ever mentioned.
    const Graph& chained
        = graph->deselect_engines(std::vector<std::string>{hashedPluginEngineName()});
    EXPECT_EQ(&chained, graph.get());

    result = graph->build_plan_at_index(hashedPlanIndex);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(result.err_msg.find("barred"), std::string::npos) << result.err_msg;

    // A plan from an engine the name does not hash to still builds, so the
    // failure above is the name filter and not a graph-wide problem.
    result = graph->build_plan_at_index(otherPlanIndex);
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}
