// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPattern.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/// Builders shared by the graph-pattern parser, compiler, and matcher tests.
namespace hipdnn_plugin_sdk::ingestor::testing
{

/// The UED name every pattern test passes as `where`.
inline constexpr const char* PATTERN_TEST_UED = "test.ued.json";

inline GraphPattern parsePatternText(const std::string& nodes)
{
    return parseGraphPattern(nlohmann::json::parse(nodes), PATTERN_TEST_UED);
}

inline std::shared_ptr<const CompiledGraphPattern> compilePatternText(const std::string& nodes)
{
    return compileGraphPattern(parsePatternText(nodes), PATTERN_TEST_UED);
}

/// A 1x1x1x1 packed tensor.
inline std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>
    patternTensor(int64_t uid, bool isVirtual = false)
{
    auto tensor = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>();
    tensor->uid = uid;
    tensor->name = "t" + std::to_string(uid);
    tensor->data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    tensor->dims = {1, 1, 1, 1};
    tensor->strides = {1, 1, 1, 1};
    tensor->virtual_ = isVirtual;
    return tensor;
}

inline std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    pointwiseNode(hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributesT attributes)
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = "pointwise";
    node->compute_data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    node->attributes.Set(attributes);
    return node;
}

/// Builds `out = in0 <operation> in1`. Omit @p in1 for a unary node.
inline std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    pointwiseNode(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                  int64_t in0,
                  std::optional<int64_t> in1,
                  int64_t out)
{
    hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributesT attributes;
    attributes.operation = operation;
    attributes.in_0_tensor_uid = in0;
    attributes.in_1_tensor_uid = in1;
    attributes.out_0_tensor_uid = out;
    return pointwiseNode(attributes);
}

inline std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    convolutionFwdNode(int64_t x, int64_t w, int64_t y)
{
    hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributesT attributes;
    attributes.x_tensor_uid = x;
    attributes.w_tensor_uid = w;
    attributes.y_tensor_uid = y;
    attributes.stride = {1, 1};
    attributes.dilation = {1, 1};
    attributes.pre_padding = {0, 0};
    attributes.post_padding = {0, 0};
    attributes.conv_mode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION;
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = "conv";
    node->compute_data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    node->attributes.Set(std::move(attributes));
    return node;
}

inline std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    convolutionBwdDataNode(int64_t dy, int64_t w, int64_t dx)
{
    hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributesT attributes;
    attributes.dy_tensor_uid = dy;
    attributes.w_tensor_uid = w;
    attributes.dx_tensor_uid = dx;
    attributes.conv_mode = hipdnn_flatbuffers_sdk::data_objects::ConvMode::CONVOLUTION;
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = "conv_bwd_data";
    node->compute_data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    node->attributes.Set(std::move(attributes));
    return node;
}

/// Layer normalization of @p x into @p y. @p mean is the optional mean output.
inline std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> layernormNode(
    int64_t x, int64_t scale, int64_t bias, int64_t epsilon, int64_t y, std::optional<int64_t> mean)
{
    hipdnn_flatbuffers_sdk::data_objects::LayernormAttributesT attributes;
    attributes.x_tensor_uid = x;
    attributes.scale_tensor_uid = scale;
    attributes.bias_tensor_uid = bias;
    attributes.epsilon_tensor_uid = epsilon;
    attributes.y_tensor_uid = y;
    attributes.mean_tensor_uid = mean;
    attributes.normalized_dim_count = 1;
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = "layernorm";
    node->compute_data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    node->attributes.Set(attributes);
    return node;
}

/// A custom op reading the @p inputs uid list and writing the @p outputs uid list.
inline std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    customOpNode(std::vector<int64_t> inputs, std::vector<int64_t> outputs)
{
    hipdnn_flatbuffers_sdk::data_objects::CustomOpAttributesT attributes;
    attributes.custom_op_id = "test.custom";
    attributes.input_tensor_uids = std::move(inputs);
    attributes.output_tensor_uids = std::move(outputs);
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = "custom";
    node->compute_data_type = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    node->attributes.Set(std::move(attributes));
    return node;
}

/// A serialized graph and the IGraph over it, kept alive together.
class PatternTestGraph
{
public:
    explicit PatternTestGraph(const hipdnn_flatbuffers_sdk::data_objects::GraphT& source)
    {
        _builder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(_builder, &source));
        _graph = std::make_unique<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper>(
            _builder.GetBufferPointer(), _builder.GetSize());
    }

    /// Wraps an already finished Graph buffer, for a graph the object API cannot express.
    explicit PatternTestGraph(flatbuffers::FlatBufferBuilder finished)
        : _builder(std::move(finished))
        , _graph(std::make_unique<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper>(
              _builder.GetBufferPointer(), _builder.GetSize()))
    {
    }

    // The wrapper points into the builder's buffer, so the object cannot be copied or moved.
    PatternTestGraph(const PatternTestGraph&) = delete;
    PatternTestGraph& operator=(const PatternTestGraph&) = delete;
    PatternTestGraph(PatternTestGraph&&) = delete;
    PatternTestGraph& operator=(PatternTestGraph&&) = delete;
    ~PatternTestGraph() = default;

    MatchContext context() const
    {
        return MatchContext{*_graph, 0, _properties};
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper> _graph;
    DeviceProperties _properties;
};

/// Assembles a GraphT from tensors and nodes, in the order given.
class PatternGraphBuilder
{
public:
    PatternGraphBuilder& tensor(int64_t uid, bool isVirtual = false)
    {
        _source.tensors.push_back(patternTensor(uid, isVirtual));
        return *this;
    }

    PatternGraphBuilder& node(std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT> node)
    {
        _source.nodes.push_back(std::move(node));
        return *this;
    }

    PatternTestGraph build() const
    {
        return PatternTestGraph(_source);
    }

private:
    hipdnn_flatbuffers_sdk::data_objects::GraphT _source;
};

} // namespace hipdnn_plugin_sdk::ingestor::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
