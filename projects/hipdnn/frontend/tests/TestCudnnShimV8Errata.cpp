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

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;
using fe::json;

using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::failureOf;

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
protected:
    // A 3-D backward convolution without padding: X/DX {1,4,5,6,8}, W/DW {8,4,3,3,3},
    // DY {1,8,3,4,6}. Every slot has its own last dimension (8, 3, 6).
    fe::OperationGraph_v8 backwardGraph(fe::DescriptorType_t kind)
    {
        const std::array<std::int64_t, 3> zeros = {0, 0, 0};
        const std::array<std::int64_t, 3> ones = {1, 1, 1};
        const fe::ConvDesc_v8 conv = fe::ConvDescBuilder_v8()
                                         .setComputeType(fe::DataType_t::FLOAT)
                                         .setMathMode(CUDNN_CROSS_CORRELATION)
                                         .setSpatialDimCount(3)
                                         .setPrePadding(3, zeros.data())
                                         .setPostPadding(3, zeros.data())
                                         .setDilation(3, ones.data())
                                         .setSpatialStride(3, ones.data())
                                         .build();
        const fe::Tensor_v8 image = tensor(1, {1, 4, 5, 6, 8});
        const fe::Tensor_v8 filter = tensor(2, {8, 4, 3, 3, 3});
        const fe::Tensor_v8 gradient = tensor(3, {1, 8, 3, 4, 6});
        fe::OperationBuilder_v8 builder(kind);
        if(kind == fe::DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
        {
            builder.setxDesc(image).setdwDesc(filter);
        }
        else
        {
            builder.setdxDesc(image).setwDesc(filter);
        }
        const fe::Operation_v8 op = builder.setdyDesc(gradient).setcDesc(conv).build();
        return graphOf(op);
    }
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

    const BuildFailure failure = failureOf([&] {
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

// ConvBwdFilter reads input X and filter DW; ConvBwdData reads input DX and filter W. With a
// wildcard only the last compared entry decides (see above), and DY's width 6 would unblock a
// rule written for 8. blocked_shape[i] meets dims[i + 1], so X/DX {1,4,5,6,8} reads as
// {4,5,6,8} in either format; NHWC compares W, blocked_shape[2], last.
TEST_F(TestCudnnShimV8Errata, ShapeRuleReadsBackwardSlots)
{
    const std::array<std::pair<const char*, fe::DescriptorType_t>, 2> kinds = {{
        {"ConvBwdFilter", fe::DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR},
        {"ConvBwdData", fe::DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR},
    }};
    for(const auto& [name, kind] : kinds)
    {
        const std::string operation = name;
        SCOPED_TRACE(operation);
        const fe::OperationGraph_v8 graph = backwardGraph(kind);
        const std::string tag = operation + "_eng0";
        const auto blocksShape
            = [&](const char* format, const char* key, const std::vector<std::int64_t>& shape) {
                  json shaped = rule(operation, 0);
                  shaped["shape_format"] = format;
                  shaped[key] = shape;
                  return fe::check_errata(rules({shaped}), tag, _handle, graph, always);
              };
        const auto blocks = [&](const char* key, std::int64_t width) {
            return blocksShape("NCHW", key, {-1, -1, -1, width});
        };

        EXPECT_TRUE(blocks("input_shape", 8));
        EXPECT_FALSE(blocks("input_shape", 6));
        EXPECT_TRUE(blocks("filter_shape", 3));
        EXPECT_FALSE(blocks("filter_shape", 8));

        EXPECT_TRUE(blocksShape("NHWC", "input_shape", {-1, -1, 6, -1}));
        EXPECT_FALSE(blocksShape("NHWC", "input_shape", {-1, -1, 8, -1}));

        // Fully specified, every entry decides.
        EXPECT_TRUE(blocksShape("NCHW", "input_shape", {4, 5, 6, 8}));
        EXPECT_FALSE(blocksShape("NCHW", "input_shape", {9, 5, 6, 8}));
        EXPECT_FALSE(blocksShape("NCHW", "input_shape", {4, 9, 6, 8}));
        EXPECT_FALSE(blocksShape("NCHW", "input_shape", {4, 5, 9, 8}));
        EXPECT_FALSE(blocksShape("NCHW", "input_shape", {4, 5, 6, 9}));
    }
}

} // namespace
