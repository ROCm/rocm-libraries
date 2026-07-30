// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "MoeGroupedMatmulGraphUtils.hpp"
#include "MoeGroupedMatmulTensorBundles.hpp"
#include "PointwiseGraphUtils.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/MoeGroupedMatmulValidation.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/node/MoeGroupedMatmulNode.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulPlan.hpp>

using namespace hipdnn_sdk_test_utils;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace
{

// A hand-built FlatBuffers graph with independent knobs for every field the
// shared routing contract inspects, so malformed configurations the frontend
// packer would never emit can still be exercised against isApplicable.
flatbuffers::FlatBufferBuilder buildRawMoeGraph(MoeGroupedMatmulMode mode,
                                                int32_t topK,
                                                bool includeTokenIndex,
                                                bool includeTokenKs,
                                                DataType tokenDataType = DataType::FLOAT,
                                                DataType weightDataType = DataType::FLOAT,
                                                DataType outputDataType = DataType::FLOAT,
                                                DataType firstTokenOffsetDataType = DataType::INT32,
                                                DataType computeDataType = DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    const std::vector<int64_t> tokenDims = {1, 4, 3};
    const std::vector<int64_t> tokenStrides = {12, 3, 1};
    const std::vector<int64_t> weightDims = {2, 3, 5};
    const std::vector<int64_t> weightStrides = {15, 5, 1};
    const std::vector<int64_t> routingDims = {2, 1, 1};
    const std::vector<int64_t> routingStrides = {1, 1, 1};
    const std::vector<int64_t> indexDims = {1, 4, 1};
    const std::vector<int64_t> indexStrides = {4, 1, 1};
    const std::vector<int64_t> outputDims = {1, 4, 5};
    const std::vector<int64_t> outputStrides = {20, 5, 1};

    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, 1, "token", tokenDataType, &tokenStrides, &tokenDims));
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, 2, "weight", weightDataType, &weightStrides, &weightDims));
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, 3, "first_token_offset", firstTokenOffsetDataType, &routingStrides, &routingDims));
    if(includeTokenIndex)
    {
        tensorAttributes.push_back(CreateTensorAttributesDirect(
            builder, 4, "token_index", DataType::INT32, &indexStrides, &indexDims));
    }
    if(includeTokenKs)
    {
        tensorAttributes.push_back(CreateTensorAttributesDirect(
            builder, 5, "token_ks", DataType::INT32, &indexStrides, &indexDims));
    }
    tensorAttributes.push_back(CreateTensorAttributesDirect(
        builder, 6, "output", outputDataType, &outputStrides, &outputDims));

    auto moeAttributes = CreateMoeGroupedMatmulAttributes(
        builder,
        1,
        2,
        3,
        includeTokenIndex ? ::flatbuffers::Optional<int64_t>(4) : ::flatbuffers::nullopt,
        includeTokenKs ? ::flatbuffers::Optional<int64_t>(5) : ::flatbuffers::nullopt,
        6,
        mode,
        topK);

    std::vector<::flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "moe_grouped_matmul",
                                     computeDataType,
                                     NodeAttributes::MoeGroupedMatmulAttributes,
                                     moeAttributes.Union()));

    auto graphOffset = CreateGraphDirect(builder,
                                         "test",
                                         computeDataType,
                                         computeDataType,
                                         computeDataType,
                                         &tensorAttributes,
                                         &nodes);
    builder.Finish(graphOffset);
    return builder;
}

hipdnn_frontend::graph::MoeGroupedMatmulAttributes validFrontendAttributes()
{
    hipdnn_frontend::graph::MoeGroupedMatmulAttributes attrs;

    auto token = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    token->set_uid(1)
        .set_dim({1, 4, 3})
        .set_stride({12, 3, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attrs.set_token(token);

    auto weight = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    weight->set_uid(2)
        .set_dim({2, 3, 5})
        .set_stride({15, 5, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attrs.set_weight(weight);

    auto firstTokenOffset = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    firstTokenOffset->set_uid(3)
        .set_dim({2, 1, 1})
        .set_stride({1, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffset);

    auto output = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    output->set_uid(6)
        .set_dim({1, 4, 5})
        .set_stride({20, 5, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attrs.set_output(output);

    attrs.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    return attrs;
}

std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    makeRoutingTensor(int64_t uid, hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::INT32)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    tensor->set_uid(uid).set_dim({1, 4, 1}).set_stride({4, 1, 1}).set_data_type(dataType);
    return tensor;
}

} // namespace

class TestMoeGroupedMatmulPlan : public ::testing::Test
{
protected:
    template <typename T>
    static void initTensorValues(hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorAttr,
                                 DataType dataType,
                                 const Tensor<T>& tensor,
                                 int64_t uid)
    {
        tensorAttr.data_type = dataType;
        tensorAttr.dims = tensor.dims();
        tensorAttr.strides = tensor.strides();
        tensorAttr.uid = uid;
    }
};

TEST_F(TestMoeGroupedMatmulPlan, ExecutePlanNoneMode)
{
    const unsigned int seed = getGlobalTestSeed();
    MoeGroupedMatmulTensorBundle<float> planBundle(2, 3, 4, 6, 6, MoeGroupedMatmulMode::NONE, 0, 1, false, seed);
    MoeGroupedMatmulTensorBundle<float> directBundle(
        2, 3, 4, 6, 6, MoeGroupedMatmulMode::NONE, 0, 1, false, seed);

    MoeGroupedMatmulParams params;
    initTensorValues(params.tokenTensor, DataType::FLOAT, planBundle.tokenTensor, 1);
    initTensorValues(params.weightTensor, DataType::FLOAT, planBundle.weightTensor, 2);
    initTensorValues(params.firstTokenOffsetTensor, DataType::INT32, planBundle.firstTokenOffsetTensor, 3);
    initTensorValues(params.outputTensor, DataType::FLOAT, planBundle.outputTensor, 6);
    params.mode = MoeGroupedMatmulMode::NONE;
    params.topK = 0;

    MoeGroupedMatmulPlan<float, float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planBundle.tokenTensor.memory().hostData();
    variantPack[2] = planBundle.weightTensor.memory().hostData();
    variantPack[3] = planBundle.firstTokenOffsetTensor.memory().hostData();
    variantPack[6] = planBundle.outputTensor.memory().hostData();

    patient.execute(variantPack);

    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        directBundle.tokenTensor,
        directBundle.weightTensor,
        directBundle.firstTokenOffsetTensor,
        directBundle.outputTensor,
        MoeGroupedMatmulMode::NONE,
        0);

    const CpuFpReferenceValidation<float> validator(0.0F, 0.0F);
    EXPECT_TRUE(validator.allClose(directBundle.outputTensor, planBundle.outputTensor));
}

TEST_F(TestMoeGroupedMatmulPlan, ExecutePlanGatherMode)
{
    const unsigned int seed = getGlobalTestSeed();
    MoeGroupedMatmulTensorBundle<float> planBundle(
        2, 3, 4, 5, 7, MoeGroupedMatmulMode::GATHER, 0, 1, false, seed);
    MoeGroupedMatmulTensorBundle<float> directBundle(
        2, 3, 4, 5, 7, MoeGroupedMatmulMode::GATHER, 0, 1, false, seed);

    MoeGroupedMatmulParams params;
    initTensorValues(params.tokenTensor, DataType::FLOAT, planBundle.tokenTensor, 1);
    initTensorValues(params.weightTensor, DataType::FLOAT, planBundle.weightTensor, 2);
    initTensorValues(params.firstTokenOffsetTensor, DataType::INT32, planBundle.firstTokenOffsetTensor, 3);
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tokenIndexAttr;
    initTensorValues(tokenIndexAttr, DataType::INT32, *planBundle.tokenIndexTensor, 4);
    params.tokenIndexTensor = tokenIndexAttr;
    initTensorValues(params.outputTensor, DataType::FLOAT, planBundle.outputTensor, 6);
    params.mode = MoeGroupedMatmulMode::GATHER;
    params.topK = 0;

    MoeGroupedMatmulPlan<float, float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planBundle.tokenTensor.memory().hostData();
    variantPack[2] = planBundle.weightTensor.memory().hostData();
    variantPack[3] = planBundle.firstTokenOffsetTensor.memory().hostData();
    variantPack[4] = planBundle.tokenIndexTensor->memory().hostData();
    variantPack[6] = planBundle.outputTensor.memory().hostData();

    patient.execute(variantPack);

    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        directBundle.tokenTensor,
        directBundle.weightTensor,
        directBundle.firstTokenOffsetTensor,
        directBundle.outputTensor,
        MoeGroupedMatmulMode::GATHER,
        0,
        &(*directBundle.tokenIndexTensor));

    const CpuFpReferenceValidation<float> validator(0.0F, 0.0F);
    EXPECT_TRUE(validator.allClose(directBundle.outputTensor, planBundle.outputTensor));
}

TEST_F(TestMoeGroupedMatmulPlan, ExecutePlanScatterMode)
{
    const unsigned int seed = getGlobalTestSeed();
    MoeGroupedMatmulTensorBundle<float> planBundle(
        2, 3, 4, 6, 6, MoeGroupedMatmulMode::SCATTER, 2, 1, false, seed);
    MoeGroupedMatmulTensorBundle<float> directBundle(
        2, 3, 4, 6, 6, MoeGroupedMatmulMode::SCATTER, 2, 1, false, seed);

    MoeGroupedMatmulParams params;
    initTensorValues(params.tokenTensor, DataType::FLOAT, planBundle.tokenTensor, 1);
    initTensorValues(params.weightTensor, DataType::FLOAT, planBundle.weightTensor, 2);
    initTensorValues(params.firstTokenOffsetTensor, DataType::INT32, planBundle.firstTokenOffsetTensor, 3);
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tokenIndexAttr;
    initTensorValues(tokenIndexAttr, DataType::INT32, *planBundle.tokenIndexTensor, 4);
    params.tokenIndexTensor = tokenIndexAttr;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tokenKsAttr;
    initTensorValues(tokenKsAttr, DataType::INT32, *planBundle.tokenKsTensor, 5);
    params.tokenKsTensor = tokenKsAttr;
    initTensorValues(params.outputTensor, DataType::FLOAT, planBundle.outputTensor, 6);
    params.mode = MoeGroupedMatmulMode::SCATTER;
    params.topK = 2;

    MoeGroupedMatmulPlan<float, float, float, float> patient(std::move(params));

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = planBundle.tokenTensor.memory().hostData();
    variantPack[2] = planBundle.weightTensor.memory().hostData();
    variantPack[3] = planBundle.firstTokenOffsetTensor.memory().hostData();
    variantPack[4] = planBundle.tokenIndexTensor->memory().hostData();
    variantPack[5] = planBundle.tokenKsTensor->memory().hostData();
    variantPack[6] = planBundle.outputTensor.memory().hostData();

    patient.execute(variantPack);

    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        directBundle.tokenTensor,
        directBundle.weightTensor,
        directBundle.firstTokenOffsetTensor,
        directBundle.outputTensor,
        MoeGroupedMatmulMode::SCATTER,
        2,
        &(*directBundle.tokenIndexTensor),
        &(*directBundle.tokenKsTensor));

    const CpuFpReferenceValidation<float> validator(0.0F, 0.0F);
    EXPECT_TRUE(validator.allClose(directBundle.outputTensor, planBundle.outputTensor));
}

TEST(TestMoeGroupedMatmulPlanBuilder, IsApplicableAcceptsEveryValidMode)
{
    const MoeGroupedMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    for(auto mode :
       {MoeGroupedMatmulMode::NONE, MoeGroupedMatmulMode::GATHER, MoeGroupedMatmulMode::SCATTER})
    {
        auto builder = createValidMoeGroupedMatmulGraph(mode);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_TRUE(patient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()))
            << "mode=" << static_cast<int>(mode);
    }
}

TEST(TestMoeGroupedMatmulPlanBuilder, IsApplicableRejectsInvalidConfigurations)
{
    const MoeGroupedMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPatient;

    // Wrong compute type.
    {
        auto builder = createValidMoeGroupedMatmulGraph(MoeGroupedMatmulMode::NONE);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        const MoeGroupedMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF>
            wrongComputePatient;
        EXPECT_FALSE(wrongComputePatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }

    // Wrong token/weight/output dtype tuple.
    {
        auto builder = createValidMoeGroupedMatmulGraph(MoeGroupedMatmulMode::NONE);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        const MoeGroupedMatmulPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT>
            mismatchedPatient;
        EXPECT_FALSE(mismatchedPatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }

    // Erased tensor UID.
    {
        auto builder = createValidMoeGroupedMatmulGraph(MoeGroupedMatmulMode::NONE);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        auto tensorMapCopy = graphWrap.getTensorMap();
        tensorMapCopy.erase(2);
        EXPECT_FALSE(floatPatient.isApplicable(graphWrap.getNode(0), tensorMapCopy));
    }

    // Incompatible node type.
    {
        const std::vector<int64_t> dims = {1, 3, 4, 4};
        auto graphTuple = buildPointwiseUnaryGraph(dims,
                                                   dims,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                                   1,
                                                   TensorLayout::NCHW);
        auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_FALSE(floatPatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }

    // NONE mode carrying a token_index UID.
    {
        auto builder = buildRawMoeGraph(MoeGroupedMatmulMode::NONE, 0, true, false);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(floatPatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }

    // GATHER mode missing token_index.
    {
        auto builder = buildRawMoeGraph(MoeGroupedMatmulMode::GATHER, 0, false, false);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(floatPatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }

    // GATHER mode with top_k != 0.
    {
        auto builder = buildRawMoeGraph(MoeGroupedMatmulMode::GATHER, 1, true, false);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(floatPatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }

    // SCATTER mode missing token_ks.
    {
        auto builder = buildRawMoeGraph(MoeGroupedMatmulMode::SCATTER, 2, true, false);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(floatPatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }

    // SCATTER with top_k == 0.
    {
        auto builder = buildRawMoeGraph(MoeGroupedMatmulMode::SCATTER, 0, true, true);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(floatPatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }

    // Non-INT32 first_token_offset.
    {
        auto builder = buildRawMoeGraph(MoeGroupedMatmulMode::NONE,
                                        0,
                                        false,
                                        false,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_FALSE(floatPatient.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    }
}

TEST(TestMoeGroupedMatmulPlanBuilder, BuildNodePlan)
{
    const MoeGroupedMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;

    // Correct case.
    {
        auto builder = createValidMoeGroupedMatmulGraph(MoeGroupedMatmulMode::SCATTER);
        const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());
        EXPECT_NO_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)));
    }

    // Incompatible node type.
    {
        const std::vector<int64_t> dims = {1, 3, 4, 4};
        auto graphTuple = buildPointwiseUnaryGraph(dims,
                                                   dims,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   DataType::FLOAT,
                                                   hipdnn_frontend::PointwiseMode::RELU_FWD,
                                                   1,
                                                   TensorLayout::NCHW);
        auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        const GraphWrapper graphWrap(serializedGraph.data(), serializedGraph.size());
        EXPECT_THROW(patient.buildNodePlan(graphWrap, graphWrap.getNode(0)), std::runtime_error);
    }
}

TEST(TestMoeGroupedMatmulPlanBuilder, PlanConstruction)
{
    auto builder = createValidMoeGroupedMatmulGraph(MoeGroupedMatmulMode::SCATTER);
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const MoeGroupedMatmulPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrap, graphWrap.getNode(0));

    const bool result
        = dynamic_cast<MoeGroupedMatmulPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

// The teeth behind Step 0's "keep in sync" note: for every rule the frontend node
// and the FlatBuffers-side contract both express, they must reach the same
// verdict; the two configurations where they intentionally diverge are pinned
// explicitly below.
TEST(TestMoeGroupedMatmulPlanBuilder, RoutingContractMatchesFrontendNode)
{
    using hipdnn_flatbuffers_sdk::utilities::checkMoeGroupedMatmulRouting;
    using hipdnn_flatbuffers_sdk::utilities::MoeGroupedMatmulRouting;

    const hipdnn_frontend::graph::GraphAttributes graphAttrs{};

    // Valid NONE / GATHER / SCATTER: both layers accept.
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::NONE);
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_TRUE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{
            MoeGroupedMatmulMode::NONE, false, false, DataType::INT32, DataType::UNSET, DataType::UNSET, 0, 2};
        EXPECT_EQ(checkMoeGroupedMatmulRouting(routing), nullptr);
    }
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::GATHER);
        attrs.set_token_index(makeRoutingTensor(4));
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_TRUE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{MoeGroupedMatmulMode::GATHER,
                                              true,
                                              false,
                                              DataType::INT32,
                                              DataType::INT32,
                                              DataType::UNSET,
                                              0,
                                              2};
        EXPECT_EQ(checkMoeGroupedMatmulRouting(routing), nullptr);
    }
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::SCATTER);
        attrs.set_token_index(makeRoutingTensor(4));
        attrs.set_token_ks(makeRoutingTensor(5));
        attrs.set_top_k(2);
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_TRUE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{MoeGroupedMatmulMode::SCATTER,
                                              true,
                                              true,
                                              DataType::INT32,
                                              DataType::INT32,
                                              DataType::INT32,
                                              2,
                                              2};
        EXPECT_EQ(checkMoeGroupedMatmulRouting(routing), nullptr);
    }

    // GATHER without token_index: both layers reject.
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::GATHER);
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_FALSE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{MoeGroupedMatmulMode::GATHER,
                                              false,
                                              false,
                                              DataType::INT32,
                                              DataType::UNSET,
                                              DataType::UNSET,
                                              0,
                                              2};
        EXPECT_NE(checkMoeGroupedMatmulRouting(routing), nullptr);
    }

    // GATHER with FLOAT token_index: both layers reject.
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::GATHER);
        attrs.set_token_index(makeRoutingTensor(4, hipdnn_frontend::DataType::FLOAT));
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_FALSE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{MoeGroupedMatmulMode::GATHER,
                                              true,
                                              false,
                                              DataType::INT32,
                                              DataType::FLOAT,
                                              DataType::UNSET,
                                              0,
                                              2};
        EXPECT_NE(checkMoeGroupedMatmulRouting(routing), nullptr);
    }

    // SCATTER without token_ks: both layers reject.
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::SCATTER);
        attrs.set_token_index(makeRoutingTensor(4));
        attrs.set_top_k(2);
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_FALSE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{MoeGroupedMatmulMode::SCATTER,
                                              true,
                                              false,
                                              DataType::INT32,
                                              DataType::INT32,
                                              DataType::UNSET,
                                              2,
                                              2};
        EXPECT_NE(checkMoeGroupedMatmulRouting(routing), nullptr);
    }

    // SCATTER with top_k == 0: both layers reject.
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::SCATTER);
        attrs.set_token_index(makeRoutingTensor(4));
        attrs.set_token_ks(makeRoutingTensor(5));
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_FALSE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{MoeGroupedMatmulMode::SCATTER,
                                              true,
                                              true,
                                              DataType::INT32,
                                              DataType::INT32,
                                              DataType::INT32,
                                              0,
                                              2};
        EXPECT_NE(checkMoeGroupedMatmulRouting(routing), nullptr);
    }

    // SCATTER with top_k == E + 1: both layers reject.
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::SCATTER);
        attrs.set_token_index(makeRoutingTensor(4));
        attrs.set_token_ks(makeRoutingTensor(5));
        attrs.set_top_k(3);
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_FALSE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{MoeGroupedMatmulMode::SCATTER,
                                              true,
                                              true,
                                              DataType::INT32,
                                              DataType::INT32,
                                              DataType::INT32,
                                              3,
                                              2};
        EXPECT_NE(checkMoeGroupedMatmulRouting(routing), nullptr);
    }

    // FLOAT first_token_offset: both layers reject.
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::NONE);
        attrs.get_first_token_offset()->set_data_type(hipdnn_frontend::DataType::FLOAT);
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_FALSE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{
            MoeGroupedMatmulMode::NONE, false, false, DataType::FLOAT, DataType::UNSET, DataType::UNSET, 0, 2};
        EXPECT_NE(checkMoeGroupedMatmulRouting(routing), nullptr);
    }

    // Intentional divergence 1: NONE mode carrying a token_index -- the node
    // accepts (the packer canonicalizes the stray tensor away before a
    // descriptor exists), the shared contract rejects (a hand-built FlatBuffers
    // graph can still express it).
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::NONE);
        attrs.set_token_index(makeRoutingTensor(4));
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_TRUE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{
            MoeGroupedMatmulMode::NONE, true, false, DataType::INT32, DataType::INT32, DataType::UNSET, 0, 2};
        EXPECT_NE(checkMoeGroupedMatmulRouting(routing), nullptr);
    }

    // Intentional divergence 2: GATHER mode with top_k == 1 -- the node accepts
    // (top_k is meaningless for GATHER and the packer drops it), the shared
    // contract rejects (GATHER requires top_k == 0).
    {
        auto attrs = validFrontendAttributes();
        attrs.set_mode(hipdnn_frontend::MoeGroupedMatmulMode::GATHER);
        attrs.set_token_index(makeRoutingTensor(4));
        attrs.set_top_k(1);
        hipdnn_frontend::graph::MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
        EXPECT_TRUE(node.pre_validate_node().is_good());

        const MoeGroupedMatmulRouting routing{MoeGroupedMatmulMode::GATHER,
                                              true,
                                              false,
                                              DataType::INT32,
                                              DataType::INT32,
                                              DataType::UNSET,
                                              1,
                                              2};
        EXPECT_NE(checkMoeGroupedMatmulRouting(routing), nullptr);
    }
}
