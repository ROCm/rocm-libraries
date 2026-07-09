// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestConfigBuiltIn.cpp
 * @brief Tests for the SelectionHeuristic::Config built-in.
 *
 * The built-in lives inside hipdnn_backend_private as a function-pointer table
 * (ConfigBuiltIn::populateFunctionTable) wrapped by HeuristicPlugin via
 * createBuiltIn. There is no .so to dlopen; the wrapper reaches the same code
 * paths used in production registration through HeuristicPluginManager.
 *
 * Wraps the table once via HeuristicPlugin::createBuiltIn and exercises both
 * the C-ABI rejection paths (null pointers, unknown policy IDs) and the
 * policy's end-to-end behavior driven by HIPDNN_HEUR_CONFIG_PATH:
 * matching rule reorders the candidate list, miss paths decline so the
 * outer policy loop falls through.
 */

#include "heuristics/config/ConfigBuiltIn.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

namespace fb = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_backend::heuristics::config::populateFunctionTable;
using hipdnn_backend::plugin::HeuristicPlugin;
using hipdnn_backend::plugin::HeuristicPluginFunctionTable;
using hipdnn_data_sdk::utilities::engineNameToId;

namespace
{
namespace config_criterion = hipdnn_data_sdk::detail::autotune_config::criterion;
namespace config_json = hipdnn_data_sdk::detail::autotune_config::json;
namespace config_op = hipdnn_data_sdk::detail::autotune_config::op;
namespace config_tensor = hipdnn_data_sdk::detail::autotune_config::tensor;

constexpr const char* MIOPEN_ENGINE_NAME = "MIOPEN_ENGINE";
constexpr const char* MIOPEN_DETERMINISTIC_ENGINE_NAME = "MIOPEN_ENGINE_DETERMINISTIC";
constexpr const char* CUSTOM_ENGINE_NAME = "Plugin1::CustomEngine";

const int64_t MIOPEN_ENGINE_ID = engineNameToId(MIOPEN_ENGINE_NAME);
const int64_t MIOPEN_DETERMINISTIC_ID = engineNameToId(MIOPEN_DETERMINISTIC_ENGINE_NAME);
const int64_t CUSTOM_ENGINE_ID = engineNameToId(CUSTOM_ENGINE_NAME);

const int64_t CONFIG_POLICY_ID
    = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::Config");

constexpr const char* OVERRIDE_ENV = "HIPDNN_HEUR_CONFIG_PATH";

/// Build a minimal serialized Graph FlatBuffer with no nodes.
std::vector<uint8_t> buildEmptyGraphBuffer()
{
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             nullptr,
                                             nullptr,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// Build a serialized Graph with a single ConvolutionFwd node referencing
/// (x, w) tensors of the requested shapes.
std::vector<uint8_t> buildConvFwdGraphBuffer(const std::vector<int64_t>& xDims,
                                             const std::vector<int64_t>& xStrides,
                                             const std::vector<int64_t>& wDims,
                                             const std::vector<int64_t>& wStrides)
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t X_UID = 1;
    constexpr int64_t W_UID = 2;
    constexpr int64_t Y_UID = 3;

    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(
            builder, X_UID, "x", fb::DataType::FLOAT, &xStrides, &xDims),
        fb::CreateTensorAttributesDirect(
            builder, W_UID, "w", fb::DataType::FLOAT, &wStrides, &wDims),
        fb::CreateTensorAttributesDirect(
            builder, Y_UID, "y", fb::DataType::FLOAT, nullptr, nullptr),
    };

    auto convAttrs = fb::CreateConvolutionFwdAttributesDirect(builder, X_UID, W_UID, Y_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "conv",
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::ConvolutionFwdAttributes,
                             convAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// Build a serialized Graph with two same-priority ConvolutionFwd nodes. The
/// first node intentionally has different tensor shapes from the common test
/// override while the second node matches it.
std::vector<uint8_t> buildTwoConvFwdGraphBuffer()
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t X0_UID = 1;
    constexpr int64_t W0_UID = 2;
    constexpr int64_t Y0_UID = 3;
    constexpr int64_t X1_UID = 4;
    constexpr int64_t W1_UID = 5;
    constexpr int64_t Y1_UID = 6;

    const std::vector<int64_t> firstXDims{9, 3, 4, 4};
    const std::vector<int64_t> firstXStrides{48, 16, 4, 1};
    const std::vector<int64_t> firstWDims{8, 3, 1, 1};
    const std::vector<int64_t> firstWStrides{3, 1, 1, 1};
    const std::vector<int64_t> secondXDims{1, 3, 4, 4};
    const std::vector<int64_t> secondXStrides{48, 16, 4, 1};
    const std::vector<int64_t> secondWDims{2, 3, 1, 1};
    const std::vector<int64_t> secondWStrides{3, 1, 1, 1};

    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(
            builder, X0_UID, "x0", fb::DataType::FLOAT, &firstXStrides, &firstXDims),
        fb::CreateTensorAttributesDirect(
            builder, W0_UID, "w0", fb::DataType::FLOAT, &firstWStrides, &firstWDims),
        fb::CreateTensorAttributesDirect(
            builder, Y0_UID, "y0", fb::DataType::FLOAT, nullptr, nullptr),
        fb::CreateTensorAttributesDirect(
            builder, X1_UID, "x1", fb::DataType::FLOAT, &secondXStrides, &secondXDims),
        fb::CreateTensorAttributesDirect(
            builder, W1_UID, "w1", fb::DataType::FLOAT, &secondWStrides, &secondWDims),
        fb::CreateTensorAttributesDirect(
            builder, Y1_UID, "y1", fb::DataType::FLOAT, nullptr, nullptr),
    };

    auto firstConvAttrs = fb::CreateConvolutionFwdAttributesDirect(builder, X0_UID, W0_UID, Y0_UID);
    auto secondConvAttrs
        = fb::CreateConvolutionFwdAttributesDirect(builder, X1_UID, W1_UID, Y1_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "conv0",
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::ConvolutionFwdAttributes,
                             firstConvAttrs.Union()),
        fb::CreateNodeDirect(builder,
                             "conv1",
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::ConvolutionFwdAttributes,
                             secondConvAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// Build a serialized Graph with a single ConvolutionBwd node referencing
/// (dy, w) tensors. Mirrors buildConvFwdGraphBuffer; matchOverrideConfig pulls
/// the rule's first two tensors against (dy, w) for "conv_dgrad".
std::vector<uint8_t> buildConvBwdGraphBuffer(const std::vector<int64_t>& dyDims,
                                             const std::vector<int64_t>& dyStrides,
                                             const std::vector<int64_t>& wDims,
                                             const std::vector<int64_t>& wStrides)
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t DY_UID = 1;
    constexpr int64_t W_UID = 2;
    constexpr int64_t DX_UID = 3;

    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(
            builder, DY_UID, "dy", fb::DataType::FLOAT, &dyStrides, &dyDims),
        fb::CreateTensorAttributesDirect(
            builder, W_UID, "w", fb::DataType::FLOAT, &wStrides, &wDims),
        fb::CreateTensorAttributesDirect(
            builder, DX_UID, "dx", fb::DataType::FLOAT, nullptr, nullptr),
    };

    auto convAttrs = fb::CreateConvolutionBwdAttributesDirect(builder, DY_UID, W_UID, DX_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "conv_bwd",
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::ConvolutionBwdAttributes,
                             convAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// Build a serialized Graph with a single ConvolutionWrw node referencing
/// (x, dy) tensors. matchOverrideConfig pairs (a=x, b=dy) for "conv_wgrad".
std::vector<uint8_t> buildConvWrwGraphBuffer(const std::vector<int64_t>& xDims,
                                             const std::vector<int64_t>& xStrides,
                                             const std::vector<int64_t>& dyDims,
                                             const std::vector<int64_t>& dyStrides)
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t X_UID = 1;
    constexpr int64_t DY_UID = 2;
    constexpr int64_t DW_UID = 3;

    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(
            builder, X_UID, "x", fb::DataType::FLOAT, &xStrides, &xDims),
        fb::CreateTensorAttributesDirect(
            builder, DY_UID, "dy", fb::DataType::FLOAT, &dyStrides, &dyDims),
        fb::CreateTensorAttributesDirect(
            builder, DW_UID, "dw", fb::DataType::FLOAT, nullptr, nullptr),
    };

    auto convAttrs = fb::CreateConvolutionWrwAttributesDirect(builder, X_UID, DY_UID, DW_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "conv_wrw",
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::ConvolutionWrwAttributes,
                             convAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

std::vector<uint8_t> buildPointwiseBinaryGraphBuffer(fb::PointwiseMode mode)
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t X_UID = 1;
    constexpr int64_t Y_UID = 2;
    constexpr int64_t OUT_UID = 3;
    const std::vector<int64_t> dims{1, 3, 4, 4};
    const std::vector<int64_t> strides{48, 16, 4, 1};

    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(builder, X_UID, "x", fb::DataType::FLOAT, &strides, &dims),
        fb::CreateTensorAttributesDirect(builder, Y_UID, "y", fb::DataType::FLOAT, &strides, &dims),
        fb::CreateTensorAttributesDirect(
            builder, OUT_UID, "out", fb::DataType::FLOAT, nullptr, nullptr),
    };

    auto pointwiseAttrs = fb::CreatePointwiseAttributes(builder,
                                                        mode,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        X_UID,
                                                        Y_UID,
                                                        ::flatbuffers::nullopt,
                                                        OUT_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             config_op::POINTWISE,
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::PointwiseAttributes,
                             pointwiseAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

std::vector<uint8_t> buildPointwiseBinaryGraphBufferWithoutSecondInputTensor()
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t X_UID = 1;
    constexpr int64_t MISSING_Y_UID = 2;
    constexpr int64_t OUT_UID = 3;
    const std::vector<int64_t> dims{1, 3, 4, 4};
    const std::vector<int64_t> strides{48, 16, 4, 1};

    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(builder, X_UID, "x", fb::DataType::FLOAT, &strides, &dims),
        fb::CreateTensorAttributesDirect(
            builder, OUT_UID, "out", fb::DataType::FLOAT, nullptr, nullptr),
    };

    auto pointwiseAttrs = fb::CreatePointwiseAttributes(builder,
                                                        fb::PointwiseMode::ADD,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        X_UID,
                                                        MISSING_Y_UID,
                                                        ::flatbuffers::nullopt,
                                                        OUT_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             config_op::POINTWISE,
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::PointwiseAttributes,
                             pointwiseAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

std::vector<uint8_t> buildPointwiseThenConvGraphBuffer()
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t PW_X_UID = 1;
    constexpr int64_t PW_Y_UID = 2;
    constexpr int64_t PW_OUT_UID = 3;
    constexpr int64_t CONV_X_UID = 4;
    constexpr int64_t CONV_W_UID = 5;
    constexpr int64_t CONV_Y_UID = 6;

    const std::vector<int64_t> pointwiseDims{1, 3, 4, 4};
    const std::vector<int64_t> pointwiseStrides{48, 16, 4, 1};
    const std::vector<int64_t> convXDims{1, 3, 4, 4};
    const std::vector<int64_t> convWDims{2, 3, 1, 1};
    const std::vector<int64_t> convXStrides{48, 16, 4, 1};
    const std::vector<int64_t> convWStrides{3, 1, 1, 1};

    const std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(
            builder, PW_X_UID, "pw_x", fb::DataType::FLOAT, &pointwiseStrides, &pointwiseDims),
        fb::CreateTensorAttributesDirect(
            builder, PW_Y_UID, "pw_y", fb::DataType::FLOAT, &pointwiseStrides, &pointwiseDims),
        fb::CreateTensorAttributesDirect(
            builder, PW_OUT_UID, "pw_out", fb::DataType::FLOAT, nullptr, nullptr),
        fb::CreateTensorAttributesDirect(
            builder, CONV_X_UID, "conv_x", fb::DataType::FLOAT, &convXStrides, &convXDims),
        fb::CreateTensorAttributesDirect(
            builder, CONV_W_UID, "conv_w", fb::DataType::FLOAT, &convWStrides, &convWDims),
        fb::CreateTensorAttributesDirect(
            builder, CONV_Y_UID, "conv_y", fb::DataType::FLOAT, nullptr, nullptr),
    };

    auto pointwiseAttrs = fb::CreatePointwiseAttributes(builder,
                                                        fb::PointwiseMode::ADD,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        ::flatbuffers::nullopt,
                                                        PW_X_UID,
                                                        PW_Y_UID,
                                                        ::flatbuffers::nullopt,
                                                        PW_OUT_UID);
    auto convAttrs
        = fb::CreateConvolutionFwdAttributesDirect(builder, CONV_X_UID, CONV_W_UID, CONV_Y_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             config_op::POINTWISE,
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::PointwiseAttributes,
                             pointwiseAttrs.Union()),
        fb::CreateNodeDirect(builder,
                             "conv",
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::ConvolutionFwdAttributes,
                             convAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

std::vector<uint8_t> buildBatchnormTrainingGraphBuffer(bool includePeerStatTensor)
{
    flatbuffers::FlatBufferBuilder builder;

    constexpr int64_t X_UID = 1;
    constexpr int64_t SCALE_UID = 2;
    constexpr int64_t BIAS_UID = 3;
    constexpr int64_t EPSILON_UID = 4;
    constexpr int64_t PEER_STAT_UID = 5;
    constexpr int64_t Y_UID = 6;

    const std::vector<int64_t> xDims{1, 3, 4, 4};
    const std::vector<int64_t> xStrides{48, 16, 4, 1};
    const std::vector<int64_t> channelDims{3};
    const std::vector<int64_t> channelStrides{1};
    const std::vector<int64_t> scalarDims{1};
    const std::vector<int64_t> scalarStrides{1};
    const std::vector<int64_t> peerStats{PEER_STAT_UID};

    std::vector<flatbuffers::Offset<fb::TensorAttributes>> tensors{
        fb::CreateTensorAttributesDirect(
            builder, X_UID, "x", fb::DataType::FLOAT, &xStrides, &xDims),
        fb::CreateTensorAttributesDirect(
            builder, SCALE_UID, "scale", fb::DataType::FLOAT, &channelStrides, &channelDims),
        fb::CreateTensorAttributesDirect(
            builder, BIAS_UID, "bias", fb::DataType::FLOAT, &channelStrides, &channelDims),
        fb::CreateTensorAttributesDirect(
            builder, EPSILON_UID, "epsilon", fb::DataType::FLOAT, &scalarStrides, &scalarDims),
        fb::CreateTensorAttributesDirect(
            builder, Y_UID, "y", fb::DataType::FLOAT, nullptr, nullptr),
    };
    if(includePeerStatTensor)
    {
        tensors.push_back(fb::CreateTensorAttributesDirect(builder,
                                                           PEER_STAT_UID,
                                                           "peer_stat",
                                                           fb::DataType::FLOAT,
                                                           &channelStrides,
                                                           &channelDims));
    }

    auto batchnormAttrs = fb::CreateBatchnormAttributesDirect(builder,
                                                              X_UID,
                                                              SCALE_UID,
                                                              BIAS_UID,
                                                              EPSILON_UID,
                                                              &peerStats,
                                                              ::flatbuffers::nullopt,
                                                              ::flatbuffers::nullopt,
                                                              ::flatbuffers::nullopt,
                                                              Y_UID);

    const std::vector<flatbuffers::Offset<fb::Node>> nodes{
        fb::CreateNodeDirect(builder,
                             "batchnorm",
                             fb::DataType::FLOAT,
                             fb::NodeAttributes::BatchnormAttributes,
                             batchnormAttrs.Union())};

    auto graphOffset = fb::CreateGraphDirect(builder,
                                             nullptr,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             fb::DataType::UNSET,
                                             &tensors,
                                             &nodes,
                                             ::flatbuffers::nullopt);
    fb::FinishGraphBuffer(builder, graphOffset);
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

/// RAII temp directory + JSON file. Returns a path that can be assigned to
/// HIPDNN_HEUR_CONFIG_PATH; the directory is removed on destruction.
class TempJsonOverrideFile
{
public:
    explicit TempJsonOverrideFile(const std::string& contents)
        : _dir(makeUniqueDir())
        , _path(_dir.path() / "override.json")
    {
        std::ofstream(_path) << contents;
    }

    std::string path() const
    {
        return _path.string();
    }

private:
    static std::filesystem::path makeUniqueDir()
    {
        static std::atomic<uint64_t> s_counter{0};
        const auto path = std::filesystem::temp_directory_path()
                          / ("hipdnn_test_config_" + std::to_string(s_counter.fetch_add(1)));
        std::filesystem::remove_all(path);
        return path;
    }

    hipdnn_test_sdk::utilities::ScopedDirectory _dir;
    std::filesystem::path _path;
};
nlohmann::json makeRuleTensor(std::vector<int64_t> dim)
{
    return {{config_json::DIM, std::move(dim)}};
}

nlohmann::json makeNamedRuleTensor(const char* tensorId, std::vector<int64_t> dim)
{
    auto tensor = makeRuleTensor(std::move(dim));
    tensor[config_json::TENSOR_ID] = tensorId;
    return tensor;
}

nlohmann::json makeOverrideConfig(std::initializer_list<nlohmann::json> rules)
{
    auto overrides = nlohmann::json::array();
    for(const auto& rule : rules)
    {
        overrides.push_back(rule);
    }
    return {{config_json::VERSION, hipdnn_data_sdk::detail::autotune_config::version::CURRENT},
            {config_json::ENGINE_OVERRIDES, std::move(overrides)}};
}

constexpr const char* DETERMINISTIC_RULE_JSON = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
      "tensors": [
        { "dim": [1, 3, 4, 4] },
        { "dim": [2, 3, 1, 1] }
      ]
    }
  ]
})";

const std::vector<int64_t> X_DIMS{1, 3, 4, 4};
const std::vector<int64_t> X_STRIDES{48, 16, 4, 1};
const std::vector<int64_t> W_DIMS{2, 3, 1, 1};
const std::vector<int64_t> W_STRIDES{3, 1, 1, 1};

class TestConfigBuiltIn : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _plugin = HeuristicPlugin::createBuiltIn(populateFunctionTable(), "built-in:Config-test");
        _handle = _plugin->createHandle();
        ASSERT_NE(_handle, nullptr);
        _desc = _plugin->createPolicyDescriptor(_handle, CONFIG_POLICY_ID);
        ASSERT_NE(_desc, nullptr);
    }

    void TearDown() override
    {
        if(_desc != nullptr)
        {
            _plugin->destroyPolicyDescriptor(_desc);
        }
        if(_handle != nullptr)
        {
            _plugin->destroyHandle(_handle);
        }
    }

    void setEngineIds(const std::vector<int64_t>& ids)
    {
        _plugin->setEngineIds(_desc, ids.data(), ids.size());
    }

    void setSerializedGraph(const std::vector<uint8_t>& buffer)
    {
        const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
        _plugin->setSerializedGraph(_desc, &data);
    }

    std::shared_ptr<HeuristicPlugin> _plugin;
    hipdnnHeuristicHandle_t _handle = nullptr;
    hipdnnHeuristicPolicyDescriptor_t _desc = nullptr;
};

// Convenience: grab the raw function table once for direct C-ABI rejection tests.
const HeuristicPluginFunctionTable& configAbi()
{
    static const HeuristicPluginFunctionTable s_funcs = populateFunctionTable();
    return s_funcs;
}

} // namespace

// ========== Built-in metadata exposed via the wrapper ==========

TEST_F(TestConfigBuiltIn, ReportsHeuristicPluginType)
{
    EXPECT_EQ(_plugin->type(), HIPDNN_PLUGIN_TYPE_HEURISTIC);
}

TEST_F(TestConfigBuiltIn, EnumeratesSingleConfigPolicy)
{
    const auto ids = _plugin->getAllPolicyIds();
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], CONFIG_POLICY_ID);
    EXPECT_EQ(_plugin->getPolicyName(CONFIG_POLICY_ID), "SelectionHeuristic::Config");
}

// ========== Policy Descriptor Lifecycle (BAD_PARAM via raw ABI) ==========

TEST(TestConfigBuiltInRejection, DescriptorCreateRejectsNullHandle)
{
    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(configAbi().policyDescriptorCreate(nullptr, CONFIG_POLICY_ID, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestConfigBuiltIn, DescriptorCreateRejectsNullOutPointer)
{
    EXPECT_EQ(configAbi().policyDescriptorCreate(_handle, CONFIG_POLICY_ID, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigBuiltInRejection, DescriptorDestroyRejectsNullDescriptor)
{
    EXPECT_EQ(configAbi().policyDescriptorDestroy(nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, DescriptorCreateRejectsUnknownPolicyId)
{
    const int64_t unknownId = hipdnn_data_sdk::utilities::policyNameToId("Vendor::NotARealPolicy");
    ASSERT_NE(unknownId, CONFIG_POLICY_ID);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    EXPECT_EQ(configAbi().policyDescriptorCreate(_handle, unknownId, &desc),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(desc, nullptr);
}

TEST_F(TestConfigBuiltIn, GetPolicyNameRejectsUnknownPolicyId)
{
    const int64_t unknownId = hipdnn_data_sdk::utilities::policyNameToId("Vendor::NotARealPolicy");
    ASSERT_NE(unknownId, CONFIG_POLICY_ID);

    const char* name = nullptr;
    EXPECT_EQ(configAbi().getPolicyName(unknownId, &name), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    EXPECT_EQ(name, nullptr);
}

// ========== SetEngineIds / SetSerializedGraph BAD_PARAM ==========

TEST(TestConfigBuiltInRejection, SetEngineIdsRejectsNullDescriptor)
{
    const std::array<int64_t, 1> ids{MIOPEN_ENGINE_ID};
    EXPECT_EQ(configAbi().policySetEngineIds(nullptr, ids.data(), ids.size()),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, SetEngineIdsRejectsNullPointerWithCount)
{
    EXPECT_EQ(configAbi().policySetEngineIds(_desc, nullptr, 3), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigBuiltInRejection, SetSerializedGraphRejectsNullDescriptor)
{
    const std::array<uint8_t, 1> buffer{0x00};
    const hipdnnPluginConstData_t data{buffer.data(), buffer.size()};
    EXPECT_EQ(configAbi().policySetSerializedGraph(nullptr, &data), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, SetSerializedGraphRejectsNullBufferStruct)
{
    EXPECT_EQ(configAbi().policySetSerializedGraph(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// ========== Finalize BAD_PARAM / NOT_INITIALIZED ==========

TEST(TestConfigBuiltInRejection, FinalizeRejectsNullDescriptor)
{
    int32_t applied = -1; // NOLINT(misc-const-correctness)
    EXPECT_EQ(configAbi().policyFinalize(nullptr, &applied), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, FinalizeRejectsNullOutApplied)
{
    EXPECT_EQ(configAbi().policyFinalize(_desc, nullptr), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestConfigBuiltInRejection, GetSortedRejectsNullDescriptor)
{
    size_t count = 0; // NOLINT(misc-const-correctness)
    EXPECT_EQ(configAbi().policyGetSortedEngineIds(nullptr, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, GetSortedRejectsNullCountPointer)
{
    EXPECT_EQ(configAbi().policyGetSortedEngineIds(_desc, nullptr, nullptr),
              HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST_F(TestConfigBuiltIn, GetSortedReturnsNotInitializedBeforeFinalize)
{
    size_t count = 0; // NOLINT(misc-const-correctness)
    EXPECT_EQ(configAbi().policyGetSortedEngineIds(_desc, nullptr, &count),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);
}

// ========== End-to-end: miss paths decline so the policy loop continues ==========

TEST_F(TestConfigBuiltIn, FinalizeWithEmptyCandidatesDeclines)
{
    // Even with a valid env file, no candidates means the policy can't pick
    // anything — decline rather than producing an empty list.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithNoEnvDeclines)
{
    // Make sure no override file leaks in from the environment.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter overrideEnv(OVERRIDE_ENV, "");

    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithMissingFileDeclines)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(
        OVERRIDE_ENV, "/nonexistent/path/hipdnn_no_such_file.json");

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithInvalidGraphBufferDeclines)
{
    // Garbage bytes large enough to clear the null check but fail FlatBuffers
    // verification — must be tolerated quietly so the policy loop still runs.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    const std::vector<uint8_t> garbage(64, 0xFF);
    setEngineIds({MIOPEN_ENGINE_ID});
    setSerializedGraph(garbage);
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithNoMatchingRuleDeclines)
{
    // Rule targets dim [99, 99, 99, 99] — no conv in the test graph matches.
    constexpr const char* JSON = R"({
      "engine_overrides": [
        {
          "op": "conv_fprop",
          "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
          "tensors": [
            { "dim": [99, 99, 99, 99] },
            { "dim": [99, 99, 99, 99] }
          ]
        }
      ]
    })";
    const TempJsonOverrideFile json(JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithMatchedEngineNotInCandidatesDeclines)
{
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    // Rule selects DETERMINISTIC; candidate list omits it.
    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeWithGraphMissingNodesDeclines)
{
    // Empty graph: nothing to walk; nothing matches. Decline.
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildEmptyGraphBuffer());
    EXPECT_FALSE(_plugin->finalize(_desc));
}

// ========== End-to-end: matching rule reorders candidates ==========

TEST_F(TestConfigBuiltIn, FinalizeMatchedRuleMovesEngineToFront)
{
    const TempJsonOverrideFile json(DETERMINISTIC_RULE_JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    ASSERT_TRUE(_plugin->finalize(_desc));
    const auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted[0], MIOPEN_DETERMINISTIC_ID);
    EXPECT_EQ(sorted[1], MIOPEN_ENGINE_ID);
    EXPECT_EQ(sorted[2], CUSTOM_ENGINE_ID);
}

TEST_F(TestConfigBuiltIn, FinalizeLegacyRuleWithoutTensorIdsUsesPositionalFallback)
{
    constexpr const char* JSON = R"({
      "engine_overrides": [
        {
          "op": "conv_fprop",
          "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
          "tensors": [
            { "dim": [1, 3, 4, 4] },
            { "dim": [2, 3, 1, 1] }
          ]
        }
      ]
    })";
    const TempJsonOverrideFile json(JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    ASSERT_TRUE(_plugin->finalize(_desc));
    const auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted.front(), MIOPEN_DETERMINISTIC_ID);
}

TEST_F(TestConfigBuiltIn, FinalizeTriesLaterSamePriorityNodeAfterFirstMiss)
{
    constexpr const char* JSON = R"({
      "engine_overrides": [
        {
          "op": "conv_fprop",
          "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
          "tensors": [
            { "tensor_id": "x_tensor_uid", "dim": [1, 3, 4, 4] },
            { "tensor_id": "w_tensor_uid", "dim": [2, 3, 1, 1] }
          ]
        }
      ]
    })";
    const TempJsonOverrideFile json(JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildTwoConvFwdGraphBuffer());

    ASSERT_TRUE(_plugin->finalize(_desc));
    const auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted.front(), MIOPEN_DETERMINISTIC_ID);
}

TEST_F(TestConfigBuiltIn, FinalizeRereadsEnvOnEachInvocation)
{
    // loadFromEnv must not be process-cached: pointing the env at a different
    // file between invocations picks up the new rule.
    const TempJsonOverrideFile firstFile(DETERMINISTIC_RULE_JSON);
    constexpr const char* SECOND_RULE = R"({
      "engine_overrides": [
        {
          "op": "conv_fprop",
          "engine_name": "MIOPEN_ENGINE",
          "tensors": [
            { "dim": [1, 3, 4, 4] },
            { "dim": [2, 3, 1, 1] }
          ]
        }
      ]
    })";
    const TempJsonOverrideFile secondFile(SECOND_RULE);

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV, firstFile.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    ASSERT_TRUE(_plugin->finalize(_desc));
    {
        const auto sorted = _plugin->getSortedEngineIds(_desc);
        ASSERT_FALSE(sorted.empty());
        EXPECT_EQ(sorted.front(), MIOPEN_DETERMINISTIC_ID);
    }

    env.setValue(secondFile.path());

    // Rerun finalize — the rule from secondFile should win this time.
    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvFwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));
    ASSERT_TRUE(_plugin->finalize(_desc));
    {
        const auto sorted = _plugin->getSortedEngineIds(_desc);
        ASSERT_FALSE(sorted.empty());
        EXPECT_EQ(sorted.front(), MIOPEN_ENGINE_ID);
    }
}

// ========== End-to-end: ConvolutionBwd / ConvolutionWrw node parsing ==========

TEST_F(TestConfigBuiltIn, FinalizeMatchedRuleMovesEngineToFrontBwdNode)
{
    // Drives the conv_dgrad branch in matchOverrideConfig — the rule pairs
    // (dy, w), so dim entries here must match the Bwd node's tensor pair.
    constexpr const char* JSON = R"({
      "engine_overrides": [
        {
          "op": "conv_dgrad",
          "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
          "tensors": [
            { "dim": [1, 3, 4, 4] },
            { "dim": [2, 3, 1, 1] }
          ]
        }
      ]
    })";
    const TempJsonOverrideFile json(JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvBwdGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    ASSERT_TRUE(_plugin->finalize(_desc));
    const auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted[0], MIOPEN_DETERMINISTIC_ID);
}

TEST_F(TestConfigBuiltIn, FinalizeMatchedRuleMovesEngineToFrontWrwNode)
{
    // Drives the conv_wgrad branch in matchOverrideConfig — the rule pairs
    // (x, dy).
    constexpr const char* JSON = R"({
      "engine_overrides": [
        {
          "op": "conv_wgrad",
          "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
          "tensors": [
            { "dim": [1, 3, 4, 4] },
            { "dim": [2, 3, 1, 1] }
          ]
        }
      ]
    })";
    const TempJsonOverrideFile json(JSON);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, CUSTOM_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildConvWrwGraphBuffer(X_DIMS, X_STRIDES, W_DIMS, W_STRIDES));

    ASSERT_TRUE(_plugin->finalize(_desc));
    const auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted[0], MIOPEN_DETERMINISTIC_ID);
}

TEST_F(TestConfigBuiltIn, FinalizePointwiseCriteriaPreventsModeOvermatch)
{
    const TempJsonOverrideFile json(
        makeOverrideConfig(
            {
                {{config_json::OP, config_op::POINTWISE},
                 {config_json::CRITERIA, {{config_criterion::POINTWISE_MODE, 2}}},
                 {config_json::ENGINE_NAME, MIOPEN_DETERMINISTIC_ENGINE_NAME},
                 {config_json::TENSORS,
                  nlohmann::json::array({makeNamedRuleTensor(config_tensor::IN_0, {1, 3, 4, 4}),
                                         makeNamedRuleTensor(config_tensor::IN_1, {1, 3, 4, 4})})}},
            })
            .dump(2));
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildPointwiseBinaryGraphBuffer(fb::PointwiseMode::ADD));
    ASSERT_TRUE(_plugin->finalize(_desc));
    auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_FALSE(sorted.empty());
    EXPECT_EQ(sorted.front(), MIOPEN_DETERMINISTIC_ID);

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildPointwiseBinaryGraphBuffer(fb::PointwiseMode::MUL));
    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeUsesConstructedPriorityToPreferConvOverEarlierPointwise)
{
    const TempJsonOverrideFile json(
        makeOverrideConfig(
            {
                {{config_json::OP, config_op::POINTWISE},
                 {config_json::CRITERIA, {{config_criterion::POINTWISE_MODE, 2}}},
                 {config_json::ENGINE_NAME, MIOPEN_DETERMINISTIC_ENGINE_NAME},
                 {config_json::TENSORS,
                  nlohmann::json::array({makeNamedRuleTensor(config_tensor::IN_0, {1, 3, 4, 4}),
                                         makeNamedRuleTensor(config_tensor::IN_1, {1, 3, 4, 4})})}},
                {{config_json::OP, config_op::CONV_FPROP},
                 {config_json::ENGINE_NAME, CUSTOM_ENGINE_NAME},
                 {config_json::TENSORS,
                  nlohmann::json::array({makeNamedRuleTensor(config_tensor::X, {1, 3, 4, 4}),
                                         makeNamedRuleTensor(config_tensor::W, {2, 3, 1, 1})})}},
            })
            .dump(2));
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID, CUSTOM_ENGINE_ID});
    setSerializedGraph(buildPointwiseThenConvGraphBuffer());

    ASSERT_TRUE(_plugin->finalize(_desc));
    const auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted.front(), CUSTOM_ENGINE_ID);
}

TEST_F(TestConfigBuiltIn, FinalizeRejectsMissingOptionalTensorUidWhenPresent)
{
    const TempJsonOverrideFile json(
        makeOverrideConfig(
            {
                {{config_json::OP, config_op::POINTWISE},
                 {config_json::CRITERIA, {{config_criterion::POINTWISE_MODE, 2}}},
                 {config_json::ENGINE_NAME, MIOPEN_DETERMINISTIC_ENGINE_NAME},
                 {config_json::TENSORS,
                  nlohmann::json::array({makeNamedRuleTensor(config_tensor::IN_0, {1, 3, 4, 4}),
                                         makeNamedRuleTensor(config_tensor::IN_1, {1, 3, 4, 4})})}},
            })
            .dump(2));
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildPointwiseBinaryGraphBufferWithoutSecondInputTensor());

    EXPECT_FALSE(_plugin->finalize(_desc));
}

TEST_F(TestConfigBuiltIn, FinalizeBatchnormPeerStatsDoNotParticipateInMatching)
{
    const TempJsonOverrideFile json(
        makeOverrideConfig(
            {
                {{config_json::OP, config_op::BATCHNORM_TRAINING},
                 {config_json::ENGINE_NAME, MIOPEN_DETERMINISTIC_ENGINE_NAME},
                 {config_json::TENSORS,
                  nlohmann::json::array({makeNamedRuleTensor(config_tensor::X, {1, 3, 4, 4}),
                                         makeNamedRuleTensor(config_tensor::SCALE, {3}),
                                         makeNamedRuleTensor(config_tensor::BIAS, {3}),
                                         makeNamedRuleTensor(config_tensor::EPSILON, {1})})}},
            })
            .dump(2));
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter env(OVERRIDE_ENV,
                                                                          json.path());

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildBatchnormTrainingGraphBuffer(true));
    ASSERT_TRUE(_plugin->finalize(_desc));
    auto sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_FALSE(sorted.empty());
    EXPECT_EQ(sorted.front(), MIOPEN_DETERMINISTIC_ID);

    setEngineIds({MIOPEN_ENGINE_ID, MIOPEN_DETERMINISTIC_ID});
    setSerializedGraph(buildBatchnormTrainingGraphBuffer(false));
    ASSERT_TRUE(_plugin->finalize(_desc));
    sorted = _plugin->getSortedEngineIds(_desc);
    ASSERT_FALSE(sorted.empty());
    EXPECT_EQ(sorted.front(), MIOPEN_DETERMINISTIC_ID);
}

// ========== Logging callback / getLastErrorString ABI shape ==========

namespace
{
// Counter and severity capture for the logging-callback test. File-scope so a
// plain C function pointer can mutate them.
std::atomic<int> gCallbackInvocations{0};
std::atomic<hipdnnSeverity_t> gCallbackLastSeverity{HIPDNN_SEV_INFO};

void testLoggingCallback(hipdnnSeverity_t severity, const char* /*message*/)
{
    gCallbackInvocations.fetch_add(1);
    gCallbackLastSeverity.store(severity);
}
} // namespace

TEST(TestConfigBuiltInLogging, LoggingCallbackReceivesErrorOnUnknownPolicyId)
{
    // Drive the STATIC_ORDERING_LOG-equivalent macro body in ConfigBuiltIn.
    // getPolicyName(unknownId) logs at ERROR severity before returning
    // BAD_PARAM; with a callback installed and log level SEV_ERROR we should
    // observe at least one invocation tagged at HIPDNN_SEV_ERROR.
    gCallbackInvocations.store(0);
    gCallbackLastSeverity.store(HIPDNN_SEV_INFO);

    ASSERT_EQ(configAbi().setLoggingCallback(&testLoggingCallback), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(configAbi().setLogLevel(HIPDNN_SEV_ERROR), HIPDNN_PLUGIN_STATUS_SUCCESS);

    const int64_t unknownId = hipdnn_data_sdk::utilities::policyNameToId("Vendor::NotARealPolicy");
    ASSERT_NE(unknownId, CONFIG_POLICY_ID);

    const char* name = nullptr;
    EXPECT_EQ(configAbi().getPolicyName(unknownId, &name), HIPDNN_PLUGIN_STATUS_BAD_PARAM);

    EXPECT_GE(gCallbackInvocations.load(), 1);
    EXPECT_EQ(gCallbackLastSeverity.load(), HIPDNN_SEV_ERROR);

    // Reset globals so other tests in the binary do not see a dangling callback.
    EXPECT_EQ(configAbi().setLoggingCallback(nullptr), HIPDNN_PLUGIN_STATUS_SUCCESS);
    EXPECT_EQ(configAbi().setLogLevel(HIPDNN_SEV_INFO), HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST(TestConfigBuiltInLogging, GetLastErrorStringHandlesNullOutPointer)
{
    // Pure ABI-shape branch coverage: getLastErrorString(nullptr) must return
    // (void) without dereferencing the null pointer.
    EXPECT_NO_FATAL_FAILURE(configAbi().getLastErrorString(nullptr));
}

TEST(TestConfigBuiltInLogging, GetLastErrorStringWritesPlaceholder)
{
    const char* msg = nullptr;
    configAbi().getLastErrorString(&msg);
    ASSERT_NE(msg, nullptr);
    EXPECT_STRNE(msg, "");
}

// ========== Empty serialized graph buffer ==========

TEST_F(TestConfigBuiltIn, SetSerializedGraphAcceptsZeroSizeBuffer)
{
    // Drives the size==0 branch in policySetSerializedGraph that clears the
    // descriptor's stored buffer instead of copying. The validation macro
    // rejects ptr==nullptr unconditionally, so we pass a real (but unused)
    // byte alongside size==0.
    const std::array<uint8_t, 1> placeholder{0x00};
    const hipdnnPluginConstData_t data{placeholder.data(), 0};
    EXPECT_EQ(configAbi().policySetSerializedGraph(_desc, &data), HIPDNN_PLUGIN_STATUS_SUCCESS);

    // With no graph and no candidates, finalize declines (covers the empty
    // buffer → parseGraphBuffer null-return path through finalize).
    setEngineIds({MIOPEN_ENGINE_ID});
    EXPECT_FALSE(_plugin->finalize(_desc));
}
