// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/OutputSpecs.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

/**
 * @file TestOutputSpecs.cpp
 * @brief Unit tests for OutputSpecs.hpp: which tensors of a graph are outputs a validating
 *        sweep fills and reads back, which are inputs, and how many bytes each output holds.
 */
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_test_sdk::utilities::SharedLogRecorder;

constexpr size_t FLOAT_BYTES = sizeof(float);

struct TensorDef
{
    int64_t uid = 0;
    DataType dataType = DataType::FLOAT;
    std::vector<int64_t> dims{4};
    std::vector<int64_t> strides{1};
    bool isVirtual = false;
};

/// A packed graph the specs are computed over.
class PackedGraph
{
public:
    explicit PackedGraph(const GraphT& graph)
    {
        _buffer.Finish(Graph::Pack(_buffer, &graph));
        _graph = std::make_unique<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper>(
            _buffer.GetBufferPointer(), _buffer.GetSize());
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph() const
    {
        return *_graph;
    }

private:
    flatbuffers::FlatBufferBuilder _buffer;
    std::unique_ptr<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper> _graph;
};

std::unique_ptr<TensorAttributesT> tensorOf(const TensorDef& def)
{
    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->uid = def.uid;
    tensor->data_type = def.dataType;
    tensor->dims = def.dims;
    tensor->strides = def.strides;
    tensor->virtual_ = def.isVirtual;
    return tensor;
}

template <typename TAttributes>
std::unique_ptr<NodeT> nodeOf(TAttributes attributes)
{
    auto node = std::make_unique<NodeT>();
    node->attributes.Set(std::move(attributes));
    return node;
}

std::unique_ptr<NodeT> pointwise(int64_t in0, int64_t in1, int64_t out)
{
    PointwiseAttributesT attributes;
    attributes.in_0_tensor_uid = in0;
    attributes.in_1_tensor_uid = in1;
    attributes.out_0_tensor_uid = out;
    return nodeOf(attributes);
}

GraphT graphOf(const std::vector<TensorDef>& tensors, std::vector<std::unique_ptr<NodeT>> nodes)
{
    GraphT graph;
    graph.name = "output_specs";
    for(const auto& def : tensors)
    {
        graph.tensors.push_back(tensorOf(def));
    }
    graph.nodes = std::move(nodes);
    return graph;
}

using Described = std::vector<std::tuple<int64_t, DataType, size_t>>;

Described describe(const std::vector<OutputBufferSpec>& outputs)
{
    Described described;
    for(const auto& spec : outputs)
    {
        described.emplace_back(spec.uid, spec.dataType, spec.byteSize);
    }
    return described;
}

template <typename... TNodes>
std::vector<std::unique_ptr<NodeT>> nodes(TNodes... nodeList)
{
    std::vector<std::unique_ptr<NodeT>> list;
    (list.push_back(std::move(nodeList)), ...);
    return list;
}

/// tensorByteSize() of one tensor packed into a graph of its own.
std::optional<size_t> byteSizeOf(const TensorDef& def)
{
    const PackedGraph packed(graphOf({def}, {}));
    return tensorByteSize(*packed.graph().getTensorMap().at(def.uid));
}

std::vector<std::string> warnings(const SharedLogRecorder& recorder)
{
    std::vector<std::string> messages;
    for(const auto& recorded : recorder.getRecordedLogs())
    {
        if(recorded.severity == HIPDNN_SEV_WARN)
        {
            messages.push_back(recorded.message);
        }
    }
    return messages;
}

bool anyContains(const std::vector<std::string>& messages, const std::string& needle)
{
    for(const auto& message : messages)
    {
        if(message.find(needle) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

TEST(TestIngestorOutputSpecs, APointwiseNodeWritesItsOutputAndReadsItsTwoInputs)
{
    const PackedGraph packed(graphOf({{1}, {2}, {3}}, nodes(pointwise(1, 2, 3))));

    const auto specs = outputSpecsFor(packed.graph());

    EXPECT_EQ(describe(specs.outputs), (Described{{3, DataType::FLOAT, 4 * FLOAT_BYTES}}));
    EXPECT_EQ(specs.inputUids, (std::vector<int64_t>{1, 2}));
}

TEST(TestIngestorOutputSpecs, AVirtualIntermediateIsNeitherAnOutputNorAnInput)
{
    const PackedGraph packed(graphOf({{1}, {2}, {3, DataType::FLOAT, {4}, {1}, true}, {4}, {5}},
                                     nodes(pointwise(1, 2, 3), pointwise(3, 4, 5))));

    const auto specs = outputSpecsFor(packed.graph());

    EXPECT_EQ(describe(specs.outputs), (Described{{5, DataType::FLOAT, 4 * FLOAT_BYTES}}));
    EXPECT_EQ(specs.inputUids, (std::vector<int64_t>{1, 2, 4}));
}

TEST(TestIngestorOutputSpecs, ANonVirtualIntermediateIsAnOutputAndOutputsComeInUidOrder)
{
    // Declared out of uid order, so the order the specs come back in is the sort's.
    const PackedGraph packed(
        graphOf({{5}, {4}, {3}, {2}, {1}}, nodes(pointwise(1, 2, 3), pointwise(3, 4, 5))));

    const auto specs = outputSpecsFor(packed.graph());

    EXPECT_EQ(
        describe(specs.outputs),
        (Described{{3, DataType::FLOAT, 4 * FLOAT_BYTES}, {5, DataType::FLOAT, 4 * FLOAT_BYTES}}));
    EXPECT_EQ(specs.inputUids, (std::vector<int64_t>{1, 2, 4}))
        << "the intermediate is written by a node, so it is not an input";
}

TEST(TestIngestorOutputSpecs, BatchnormTrainingStatisticsAreOutputsButInferenceMeanIsAnInput)
{
    BatchnormAttributesT training;
    training.x_tensor_uid = 1;
    training.scale_tensor_uid = 2;
    training.bias_tensor_uid = 3;
    training.y_tensor_uid = 4;
    training.mean_tensor_uid = 5;
    training.inv_variance_tensor_uid = 6;
    const PackedGraph trainingGraph(
        graphOf({{1}, {2}, {3}, {4}, {5}, {6}}, nodes(nodeOf(std::move(training)))));

    const auto trainingSpecs = outputSpecsFor(trainingGraph.graph());

    EXPECT_EQ(describe(trainingSpecs.outputs),
              (Described{{4, DataType::FLOAT, 4 * FLOAT_BYTES},
                         {5, DataType::FLOAT, 4 * FLOAT_BYTES},
                         {6, DataType::FLOAT, 4 * FLOAT_BYTES}}));
    EXPECT_EQ(trainingSpecs.inputUids, (std::vector<int64_t>{1, 2, 3}));

    BatchnormInferenceAttributesT inference;
    inference.x_tensor_uid = 1;
    inference.mean_tensor_uid = 2;
    inference.inv_variance_tensor_uid = 3;
    inference.scale_tensor_uid = 4;
    inference.bias_tensor_uid = 5;
    inference.y_tensor_uid = 6;
    const PackedGraph inferenceGraph(
        graphOf({{1}, {2}, {3}, {4}, {5}, {6}}, nodes(nodeOf(inference))));

    const auto inferenceSpecs = outputSpecsFor(inferenceGraph.graph());

    EXPECT_EQ(describe(inferenceSpecs.outputs), (Described{{6, DataType::FLOAT, 4 * FLOAT_BYTES}}));
    EXPECT_EQ(inferenceSpecs.inputUids, (std::vector<int64_t>{1, 2, 3, 4, 5}));
}

TEST(TestIngestorOutputSpecs, TheByteSizeFollowsTheStrideSpanNotTheElementCount)
{
    // Rows of 3 elements padded to a stride of 10: the last element sits at 1*10 + 2*1.
    EXPECT_EQ(byteSizeOf({1, DataType::FLOAT, {2, 3}, {10, 1}}),
              std::optional<size_t>(13 * FLOAT_BYTES));
    // A rank-0 tensor holds one element.
    EXPECT_EQ(byteSizeOf({1, DataType::HALF, {}, {}}), std::optional<size_t>(2));
    // A transposed layout spans as far as its largest stride reaches.
    EXPECT_EQ(byteSizeOf({1, DataType::DOUBLE, {3, 2}, {1, 3}}), std::optional<size_t>(6 * 8));
}

TEST(TestIngestorOutputSpecs, SubByteElementsRoundTheSpanUpToWholeBytes)
{
    EXPECT_EQ(byteSizeOf({1, DataType::FP4_E2M1, {3}, {1}}), std::optional<size_t>(2));
    EXPECT_EQ(byteSizeOf({1, DataType::INT4, {5}, {1}}), std::optional<size_t>(3));
    EXPECT_EQ(byteSizeOf({1, DataType::FP6_E3M2, {3}, {1}}), std::optional<size_t>(3));
}

TEST(TestIngestorOutputSpecs, ATensorWithAnUnknownExtentOrTypeHasNoByteSize)
{
    // Each case differs from this sized tensor in one respect.
    ASSERT_EQ(byteSizeOf({1, DataType::FLOAT, {4, 4}, {4, 1}}), std::optional<size_t>(16 * 4));
    EXPECT_FALSE(byteSizeOf({1, DataType::FLOAT, {4, 0}, {4, 1}}).has_value()) << "zero extent";
    EXPECT_FALSE(byteSizeOf({1, DataType::FLOAT, {-2, 4}, {4, 1}}).has_value())
        << "negative extent";
    EXPECT_FALSE(byteSizeOf({1, DataType::UNSET, {4, 4}, {4, 1}}).has_value()) << "unset type";
    EXPECT_FALSE(byteSizeOf({1, DataType::FLOAT, {4, 4}, {1}}).has_value())
        << "dims and strides of different ranks";
}

TEST(TestIngestorOutputSpecs, AnOutputWithoutAByteSizeIsDroppedWithAWarning)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    // Distinctive uids, so a warning naming one cannot match by accident.
    const PackedGraph packed(
        graphOf({{1}, {902, DataType::FLOAT, {0}, {1}}, {903, DataType::UNSET, {4}, {1}}, {904}},
                nodes(pointwise(1, 1, 902), pointwise(1, 1, 903), pointwise(1, 1, 904))));

    const auto specs = outputSpecsFor(packed.graph());

    EXPECT_EQ(describe(specs.outputs), (Described{{904, DataType::FLOAT, 4 * FLOAT_BYTES}}));
    const auto messages = warnings(recorder);
    EXPECT_TRUE(anyContains(messages, "902")) << "the zero-extent output is named";
    EXPECT_FALSE(anyContains(messages, "904")) << "a sized output draws no warning";
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
