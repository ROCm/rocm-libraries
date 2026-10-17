// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x errata filter against plan tags and graphs built through the
// shim and the mock hipDNN backend. Built only when the JSON library is.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Errata.h>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuilderFailure;
using hipdnn_shim_test::builderFailureOf;

constexpr const char* ERRATA_VARIABLE = "CUDNN_ERRATA_JSON_FILE";

bool always()
{
    return true;
}

json rules(std::vector<json> entries)
{
    return json{{"version", 1}, {"rules", std::move(entries)}};
}

json rule(const std::string& operation, std::int64_t engine)
{
    return json{{"rule_id", "test_rule"}, {"operation", operation}, {"engine", engine}};
}

class TestCudnnShimV8Errata : public hipdnn_shim_test::ShimV8PlanFixture
{
};

// A rule names "<op>_eng<N>_" so engine 1 never blocks engine 10.
TEST_F(TestCudnnShimV8Errata, RuleMatchesIndexPrefix)
{
    installEngines(11);
    const fe::OperationGraph_v8 graph = convGraph();
    const fe::ExecutionPlan_v8 engineOne = planFor(graph, 1);
    const fe::ExecutionPlan_v8 engineTen = planFor(graph, 10);
    ASSERT_EQ(engineOne.getTag(), "ConvFwd_eng1");
    ASSERT_EQ(engineTen.getTag(), "ConvFwd_eng10");

    const json blockOne = rules({rule("ConvFwd", 1)});
    EXPECT_TRUE(fe::check_errata(blockOne, engineOne.getTag(), _handle, always));
    EXPECT_FALSE(fe::check_errata(blockOne, engineTen.getTag(), _handle, always));
    EXPECT_FALSE(fe::check_errata(blockOne, engineOne.getTag(), _handle, [] { return false; }));
    EXPECT_FALSE(
        fe::check_errata(rules({rule("ConvBwdData", 1)}), engineOne.getTag(), _handle, always));
}

// The rules PyTorch hard-codes, at the shim's CUDNN_VERSION of 92200: only the
// open-ended 3-D ConvBwdData engine-23 rule is in its version window.
TEST_F(TestCudnnShimV8Errata, PyTorchRulesAt92200)
{
    const json issue188288 = json::parse(R"({"version":1,"rules":[
        {"rule_id":"188288","operation":"ConvFwd","engine":5,
         "cudnn_version_start":92301,"cudnn_version_end":92400}]})");
    const json sm120 = json::parse(R"({"version":1,"rules":[
        {"rule_id":"sm120_conv_fwd_engine_58","operation":"ConvFwd","engine":58,
         "cudnn_version_start":92300,"cudnn_version_end":-1},
        {"rule_id":"sm120_conv_bwd_data_engine_63","operation":"ConvBwdData","engine":63,
         "cudnn_version_start":92300,"cudnn_version_end":-1},
        {"rule_id":"sm120_conv_bwd_filter_engine_63","operation":"ConvBwdFilter","engine":63,
         "cudnn_version_start":92300,"cudnn_version_end":-1}]})");
    const json threeD = json::parse(R"({"version":1,"rules":[
        {"rule_id":"163539","operation":"ConvFwd","engine":23,
         "cudnn_version_start":90800,"cudnn_version_end":91500},
        {"rule_id":"ConvBwdData","operation":"ConvBwdData","engine":23,
         "cudnn_version_start":8000,"cudnn_version_end":-1}]})");

    EXPECT_FALSE(fe::check_errata(issue188288, "ConvFwd_eng5", _handle, always));
    for(const char* tag : {"ConvFwd_eng58", "ConvBwdData_eng63", "ConvBwdFilter_eng63"})
    {
        EXPECT_FALSE(fe::check_errata(sm120, tag, _handle, always)) << tag;
    }
    EXPECT_FALSE(fe::check_errata(threeD, "ConvFwd_eng23", _handle, always));
    EXPECT_TRUE(fe::check_errata(threeD, "ConvBwdData_eng23", _handle, always));
    EXPECT_TRUE(fe::check_errata(threeD, "ConvBwdData_eng23_k25=16", _handle, always));
}

// Plan tags stream the shim's cudnnBackendKnobType_t integers, so a knob
// substring only matches when written with them.
TEST_F(TestCudnnShimV8Errata, KnobSubstringUsesShimEnum)
{
    installEngineKnobs(0, {{"prov.tile_size", 8, 0, 64, 8, {}}});
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 knobbed = engine(graph, 0);
    ASSERT_EQ(knobbed.getSupportedKnobs().size(), 1U);
    knobbed.getSupportedKnobs().front().setChoice(16);
    const fe::EngineConfig_v8 config = fe::EngineConfigBuilder_v8().setEngine(knobbed).build();
    const fe::ExecutionPlan_v8 plan = fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();

    const std::string tileSize = std::to_string(static_cast<int>(CUDNN_KNOB_TYPE_TILE_SIZE));
    const std::string stages = std::to_string(static_cast<int>(CUDNN_KNOB_TYPE_STAGES));
    json matching = rule("ConvFwd", 0);
    matching["knob"] = json::array({"k" + tileSize + "=16"});
    json otherValue = rule("ConvFwd", 0);
    otherValue["knob"] = json::array({"k" + tileSize + "=8"});
    json otherType = rule("ConvFwd", 0);
    otherType["knob"] = json::array({"k" + stages + "=16"});

    EXPECT_TRUE(fe::check_errata(rules({matching}), plan.getTag(), _handle, always));
    EXPECT_FALSE(fe::check_errata(rules({otherValue}), plan.getTag(), _handle, always));
    EXPECT_FALSE(fe::check_errata(rules({otherType}), plan.getTag(), _handle, always));
}

TEST_F(TestCudnnShimV8Errata, RulesFileFromEnvironment)
{
    const hipdnn_test_sdk::utilities::ScopedDirectory scratch
        = hipdnn_test_sdk::utilities::claimScratchDirectory("cudnn_shim_v8_errata");
    const std::filesystem::path fromEnvironment = scratch.path() / "environment.json";
    const std::filesystem::path fromArgument = scratch.path() / "argument.json";
    std::ofstream(fromEnvironment) << rules({rule("ConvFwd", 1)}).dump();
    std::ofstream(fromArgument) << rules({rule("ConvFwd", 2)}).dump();

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter variable(ERRATA_VARIABLE);
    hipdnn_data_sdk::utilities::unsetEnv(ERRATA_VARIABLE);

    json loaded;
    EXPECT_FALSE(fe::load_from_config(loaded, ""));
    ASSERT_TRUE(fe::load_from_config(loaded, fromArgument.string()));
    EXPECT_TRUE(fe::check_errata(loaded, "ConvFwd_eng2", _handle, always));

    // The variable overrides the argument.
    variable.setValue(fromEnvironment.string());
    json overridden;
    ASSERT_TRUE(fe::load_from_config(overridden, fromArgument.string()));
    EXPECT_TRUE(fe::check_errata(overridden, "ConvFwd_eng1", _handle, always));
    EXPECT_FALSE(fe::check_errata(overridden, "ConvFwd_eng2", _handle, always));

    variable.setValue((scratch.path() / "missing.json").string());
    json missing;
    EXPECT_FALSE(fe::load_from_config(missing, fromArgument.string()));
}

TEST_F(TestCudnnShimV8Errata, ShapeRuleWithoutFormatRejected)
{
    const fe::OperationGraph_v8 graph = convGraph();
    json shaped = rule("ConvFwd", 0);
    shaped["input_shape"] = json::array({-1, 4, -1, -1});

    const BuilderFailure failure = builderFailureOf([&] {
        [[maybe_unused]] const bool blocked
            = fe::check_errata(rules({shaped}), "ConvFwd_eng0", _handle, graph, always);
    });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("shape_format"), std::string::npos) << failure.what;

    // A rule for another engine never reaches the shape check.
    json otherEngine = shaped;
    otherEngine["engine"] = 1;
    EXPECT_FALSE(fe::check_errata(rules({otherEngine}), "ConvFwd_eng0", _handle, graph, always));
}

// check_shape reads five dimensions, so a 4-D tensor's fifth is the
// value-initialized zero (upstream reads it uninitialized). With the
// wildcard chaining only the last compared dimension decides.
TEST_F(TestCudnnShimV8Errata, ShapeRuleReadsMissingDimensionAsZero)
{
    const fe::OperationGraph_v8 graph = convGraph();
    json zeroWidth = rule("ConvFwd", 0);
    zeroWidth["shape_format"] = "NCHW";
    zeroWidth["input_shape"] = json::array({-1, -1, -1, 0});
    json realWidth = zeroWidth;
    realWidth["input_shape"] = json::array({-1, -1, -1, 8});
    json filterWidth = rule("ConvFwd", 0);
    filterWidth["filter_shape"] = json::array({-1, -1, -1, 0});

    EXPECT_TRUE(fe::check_errata(rules({zeroWidth}), "ConvFwd_eng0", _handle, graph, always));
    EXPECT_FALSE(fe::check_errata(rules({realWidth}), "ConvFwd_eng0", _handle, graph, always));
    EXPECT_TRUE(fe::check_errata(rules({filterWidth}), "ConvFwd_eng0", _handle, graph, always));
}

} // namespace
