// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "harness/bundle/MoeRoutingUnroutedRows.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;
using namespace hipdnn_integration_tests::bundle;

namespace
{

// topk_ids=1 [2, 2]; first_token_offset=2 [E + 1, 1, 1]; token_index=3, token_ks=4 [1, 4, 1].
constexpr int64_t K_IDS_UID = 1;
constexpr int64_t K_OFFSETS_UID = 2;
constexpr int64_t K_INDEX_UID = 3;
constexpr int64_t K_KS_UID = 4;
constexpr int64_t K_EXPERTS = 3;
constexpr int64_t K_ROWS = 4;
constexpr int32_t K_SENTINEL = std::numeric_limits<int32_t>::max();

flatbuffers::DetachedBuffer buildRoutingGraph(DataType offsetType)
{
    flatbuffers::FlatBufferBuilder b;
    const std::vector<int64_t> idDims = {2, 2};
    const std::vector<int64_t> idStrides = {2, 1};
    const std::vector<int64_t> offsetDims = {K_EXPERTS + 1, 1, 1};
    const std::vector<int64_t> offsetStrides = {1, 1, 1};
    const std::vector<int64_t> rowDims = {1, K_ROWS, 1};
    const std::vector<int64_t> rowStrides = {K_ROWS, 1, 1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(
        b, K_IDS_UID, "topk_ids", DataType::INT32, &idStrides, &idDims));
    tensors.push_back(CreateTensorAttributesDirect(
        b, K_OFFSETS_UID, "first_token_offset", offsetType, &offsetStrides, &offsetDims));
    tensors.push_back(CreateTensorAttributesDirect(
        b, K_INDEX_UID, "token_index", DataType::INT32, &rowStrides, &rowDims));
    tensors.push_back(CreateTensorAttributesDirect(
        b, K_KS_UID, "token_ks", DataType::INT32, &rowStrides, &rowDims));

    const auto routing
        = CreateMoeRoutingAttributes(b, K_IDS_UID, K_OFFSETS_UID, K_INDEX_UID, K_KS_UID, K_EXPERTS);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(
        b, "moe_routing", DataType::FLOAT, NodeAttributes::MoeRoutingAttributes, routing.Union()));
    b.Finish(CreateGraphDirect(
        b, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes));
    return b.Release();
}

template <class T>
std::shared_ptr<ITensor> tensorOf(const std::vector<int64_t>& dims,
                                  const std::vector<int64_t>& strides,
                                  const std::vector<T>& values)
{
    auto tensor = std::make_shared<Tensor<T>>(dims, strides);
    tensor->fillWithData(values.data(), values.size() * sizeof(T));
    return tensor;
}

std::shared_ptr<ITensor> rows(const std::vector<int32_t>& values)
{
    return tensorOf<int32_t>({1, K_ROWS, 1}, {K_ROWS, 1, 1}, values);
}

std::vector<int32_t> rowValues(ITensor& tensor)
{
    const auto* data = static_cast<const int32_t*>(tensor.rawHostData());
    return {data, data + K_ROWS};
}

// Two of four ids routed (offsets end at 2): rows 2 and 3 are unrouted.
struct Sides
{
    OutputTensors engine;
    OutputTensors expected;
};

template <class OffsetT>
Sides twoRoutedRows(int32_t engineIndexRow3)
{
    const std::vector<OffsetT> offsets = {0, 1, 1, 2};
    Sides sides;
    for(auto* side : {&sides.engine, &sides.expected})
    {
        (*side)[K_OFFSETS_UID] = tensorOf<OffsetT>({K_EXPERTS + 1, 1, 1}, {1, 1, 1}, offsets);
        (*side)[K_KS_UID] = rows({1, 0, K_SENTINEL, K_SENTINEL});
    }
    sides.expected[K_INDEX_UID] = rows({0, 1, K_SENTINEL, K_SENTINEL});
    sides.engine[K_INDEX_UID] = rows({0, 1, K_SENTINEL, engineIndexRow3});
    return sides;
}

ExpectedTensorLookup lookupIn(OutputTensors& tensors)
{
    return [&tensors](int64_t uid) -> ITensor& { return *tensors.at(uid); };
}

} // namespace

// Rows past first_token_offset[E] that both sides left untouched are zeroed on both, so the
// integer validator's sentinel check grades only the routed rows; the routed rows are kept.
TEST(TestMoeRoutingUnroutedRows, UntouchedUnroutedRowsAreTakenOutOfTheComparison)
{
    const auto graph = buildRoutingGraph(DataType::INT32);
    auto sides = twoRoutedRows<int32_t>(K_SENTINEL);

    const auto problem = settleMoeRoutingUnroutedRows(
        GraphWrapper(graph.data(), graph.size()), sides.engine, lookupIn(sides.expected));

    EXPECT_TRUE(problem.empty()) << problem;
    for(auto* side : {&sides.engine, &sides.expected})
    {
        EXPECT_EQ(rowValues(*side->at(K_INDEX_UID)), (std::vector<int32_t>{0, 1, 0, 0}));
        EXPECT_EQ(rowValues(*side->at(K_KS_UID)), (std::vector<int32_t>{1, 0, 0, 0}));
    }
}

// The routed count may come from int64 offsets.
TEST(TestMoeRoutingUnroutedRows, ReadsInt64Offsets)
{
    const auto graph = buildRoutingGraph(DataType::INT64);
    auto sides = twoRoutedRows<int64_t>(K_SENTINEL);

    const auto problem = settleMoeRoutingUnroutedRows(
        GraphWrapper(graph.data(), graph.size()), sides.engine, lookupIn(sides.expected));

    EXPECT_TRUE(problem.empty()) << problem;
    EXPECT_EQ(rowValues(*sides.engine.at(K_INDEX_UID)), (std::vector<int32_t>{0, 1, 0, 0}));
}

// An engine that writes an unrouted row is reported, by tensor and row.
TEST(TestMoeRoutingUnroutedRows, EngineWritingAnUnroutedRowIsReported)
{
    const auto graph = buildRoutingGraph(DataType::INT32);
    auto sides = twoRoutedRows<int32_t>(7);

    const auto problem = settleMoeRoutingUnroutedRows(
        GraphWrapper(graph.data(), graph.size()), sides.engine, lookupIn(sides.expected));

    EXPECT_NE(problem.find("token_index row 3"), std::string::npos) << problem;
    EXPECT_NE(problem.find("engine wrote 7"), std::string::npos) << problem;
    EXPECT_EQ(problem.find("token_ks"), std::string::npos) << problem;
}

// A graph without a MoE routing node is left alone.
TEST(TestMoeRoutingUnroutedRows, OtherGraphsAreUntouched)
{
    flatbuffers::FlatBufferBuilder b;
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const std::vector<flatbuffers::Offset<Node>> nodes;
    b.Finish(CreateGraphDirect(
        b, "empty", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes));
    const auto graph = b.Release();
    auto sides = twoRoutedRows<int32_t>(K_SENTINEL);

    const auto problem = settleMoeRoutingUnroutedRows(
        GraphWrapper(graph.data(), graph.size()), sides.engine, lookupIn(sides.expected));

    EXPECT_TRUE(problem.empty());
    EXPECT_EQ(rowValues(*sides.engine.at(K_INDEX_UID)),
              (std::vector<int32_t>{0, 1, K_SENTINEL, K_SENTINEL}));
}
