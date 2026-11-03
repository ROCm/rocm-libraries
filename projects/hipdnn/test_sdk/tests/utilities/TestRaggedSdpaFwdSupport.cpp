// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaFwdSupport.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper;

namespace
{

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t RAGGED_OFFSET_Q_UID = 10;
constexpr int64_t RAGGED_OFFSET_KV_UID = 11;

// [B = 1, H = 2, S = 4, D = 8]. Token-major strides are {64, 8, 16, 1}.
const std::vector<int64_t> DIMS = raggedDims(1, 4, 2, 8);

// A fully ragged bf16 graph with the ragged_offset dropped from every primary not in `ragged`.
flatbuffers::DetachedBuffer makeGraph(const std::vector<int64_t>& ragged)
{
    const auto graph = createRaggedSdpaFwdGraph(Q_UID,
                                                K_UID,
                                                V_UID,
                                                O_UID,
                                                RAGGED_OFFSET_Q_UID,
                                                RAGGED_OFFSET_KV_UID,
                                                /*batch=*/1,
                                                DIMS,
                                                DIMS,
                                                DIMS,
                                                DIMS,
                                                DataType::BFLOAT16);
    auto graphT = std::unique_ptr<GraphT>(GetGraph(graph.GetBufferPointer())->UnPack());
    for(auto& tensor : graphT->tensors)
    {
        const bool primary = tensor->uid == Q_UID || tensor->uid == K_UID || tensor->uid == V_UID
                             || tensor->uid == O_UID;
        if(primary && std::find(ragged.begin(), ragged.end(), tensor->uid) == ragged.end())
        {
            tensor->ragged_offset_tensor_uid = flatbuffers::nullopt;
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateGraph(builder, graphT.get()));
    return builder.Release();
}

const SdpaAttributes& sdpaAttributes(const GraphWrapper& graphWrapper)
{
    return *graphWrapper.getNode(0).attributes_as_SdpaAttributes();
}

// A standalone ragged tensor with the given strides and ragged_offset_multiplier.
flatbuffers::FlatBufferBuilder makeRaggedTensor(const std::vector<int64_t>& strides,
                                                int64_t multiplier)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateTensorAttributesDirect(builder,
                                                Q_UID,
                                                "Q",
                                                DataType::BFLOAT16,
                                                &strides,
                                                &DIMS,
                                                /*virtual_=*/false,
                                                TensorValue::NONE,
                                                /*value=*/0,
                                                /*is_runtime_pass_by_value=*/false,
                                                RAGGED_OFFSET_Q_UID,
                                                /*alignment=*/16,
                                                multiplier));
    return builder;
}

bool isSupportedLayout(const std::vector<int64_t>& strides, int64_t multiplier)
{
    const auto builder = makeRaggedTensor(strides, multiplier);
    return isSupportedRaggedSdpaLayout(
        *flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer()));
}

} // namespace

TEST(TestRaggedSdpaFwdSupport, HasRaggedSdpaPrimaryForEachSingleRaggedPrimary)
{
    for(const auto uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        const auto graph = makeGraph({uid});
        const GraphWrapper graphWrapper(graph.data(), graph.size());
        EXPECT_TRUE(hasRaggedSdpaPrimary(sdpaAttributes(graphWrapper), graphWrapper.getTensorMap()))
            << "ragged uid " << uid;
    }
}

TEST(TestRaggedSdpaFwdSupport, HasRaggedSdpaPrimaryFalseForDenseNode)
{
    const auto graph = makeGraph({});
    const GraphWrapper graphWrapper(graph.data(), graph.size());
    EXPECT_FALSE(hasRaggedSdpaPrimary(sdpaAttributes(graphWrapper), graphWrapper.getTensorMap()));
}

// Only Q is ragged. Without Q in the map, or with a null entry for it, no ragged primary is seen.
TEST(TestRaggedSdpaFwdSupport, HasRaggedSdpaPrimaryIgnoresMissingAndNullUids)
{
    const auto graph = makeGraph({Q_UID});
    const GraphWrapper graphWrapper(graph.data(), graph.size());
    const auto& attributes = sdpaAttributes(graphWrapper);
    ASSERT_TRUE(hasRaggedSdpaPrimary(attributes, graphWrapper.getTensorMap()));

    auto tensorMap = graphWrapper.getTensorMap();
    tensorMap.erase(Q_UID);
    EXPECT_FALSE(hasRaggedSdpaPrimary(attributes, tensorMap));

    tensorMap[Q_UID] = nullptr;
    EXPECT_FALSE(hasRaggedSdpaPrimary(attributes, tensorMap));
}

TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutForTokenMajorStrides)
{
    EXPECT_TRUE(isSupportedLayout(raggedStrides(DIMS), 1));
}

// RFC-0014 requires ragged_offset_multiplier >= 1.
TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutFalseForZeroMultiplier)
{
    EXPECT_FALSE(isSupportedLayout(raggedStrides(DIMS), 0));
}

// Contiguous [B, H, S, D] strides put the heads outside the tokens, so tokens are not packed.
TEST(TestRaggedSdpaFwdSupport, IsSupportedRaggedSdpaLayoutFalseForContiguousBhsdStrides)
{
    EXPECT_FALSE(isSupportedLayout(generateStrides(DIMS), 1));
}
