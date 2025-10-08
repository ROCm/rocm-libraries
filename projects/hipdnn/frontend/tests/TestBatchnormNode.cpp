// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBatchnormNode, PreValidateNode)
{
    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, PreValidateNodeMissingValues)
{
    BatchnormAttributes batchnormAttributes;

    GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    auto batchnormAttributesCopy = batchnormAttributes;
    BatchnormNode nodeWithX(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormNode nodeWithY(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithY.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormNode nodeWithScale(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormNode nodeWithBias(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithBias.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormNode nodeWithAllValues(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, InferPropertiesNode)
{
    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());

    auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormNode, InferPropertiesNodeWithStats)
{
    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_prev_running_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_prev_running_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_next_running_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_next_running_variance(std::make_shared<TensorAttributes>());

    auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_uid(3).set_name("MeanTensor");

    auto invVarianceTensor = batchnormAttributes.get_inv_variance();
    invVarianceTensor->set_uid(4).set_name("InvVarianceTensor");

    auto nextRunningMeanTensor = batchnormAttributes.get_next_running_mean();
    nextRunningMeanTensor->set_uid(5).set_name("NextRunningMeanTensor");

    auto nextRunningVarianceTensor = batchnormAttributes.get_next_running_variance();
    nextRunningVarianceTensor->set_uid(6).set_name("NextRunningVarianceTensor");

    GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(meanTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(invVarianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(nextRunningMeanTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(nextRunningVarianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
}

TEST(TestBatchnormNode, PackNode)
{
    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_name("Batchnorm");

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2)
        .set_name("YTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(4)
        .set_name("BiasTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(5)
        .set_name("EpsilonTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1})
        .set_stride({1});
    batchnormAttributes.set_epsilon(epsilonTensor);

    GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "Batchnorm");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_BatchnormAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->y_tensor_uid(), yTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->bias_tensor_uid(), biasTensor->get_uid());
    EXPECT_EQ(packedAttributes->epsilon_tensor_uid(), epsilonTensor->get_uid());
}

TEST(TestBatchnormNode, GatherHipdnnTensorIds)
{
    BatchnormAttributes batchnormAttributes;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");
    batchnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("YTensor");
    batchnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("ScaleTensor");
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(4).set_name("BiasTensor");
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(5).set_name("EpsilonTensor");
    batchnormAttributes.set_epsilon(epsilonTensor);

    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(9).set_name("PeerStat1");

    auto peerStat2 = std::make_shared<TensorAttributes>();
    peerStat2->set_uid(10).set_name("PeerStat2");

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    std::unordered_set<int64_t> usedIds;
    std::unordered_set<int64_t> duplicateIds;
    node.gather_hipdnn_tensor_ids(usedIds, duplicateIds);

    EXPECT_TRUE(usedIds.find(1) != usedIds.end());
    EXPECT_TRUE(usedIds.find(2) != usedIds.end());
    EXPECT_TRUE(usedIds.find(3) != usedIds.end());
    EXPECT_TRUE(usedIds.find(4) != usedIds.end());
    EXPECT_TRUE(usedIds.find(5) != usedIds.end());
    EXPECT_TRUE(usedIds.find(9) != usedIds.end());
    EXPECT_TRUE(usedIds.find(10) != usedIds.end());
    EXPECT_TRUE(duplicateIds.empty());
}

TEST(TestBatchnormNode, GatherHipdnnTensorsCollectsDuplicates)
{
    BatchnormAttributes batchnormAttributes;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");
    batchnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("YTensor");
    batchnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("ScaleTensor");
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(4).set_name("BiasTensor");
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(5).set_name("EpsilonTensor");
    batchnormAttributes.set_epsilon(epsilonTensor);

    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(3).set_name("peerStat1");

    auto peerStat2 = std::make_shared<TensorAttributes>();
    peerStat2->set_uid(4).set_name("peerStat2");

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    std::unordered_set<int64_t> usedIds;
    std::unordered_set<int64_t> duplicateIds;
    node.gather_hipdnn_tensor_ids(usedIds, duplicateIds);

    EXPECT_TRUE(usedIds.find(1) != usedIds.end());
    EXPECT_TRUE(usedIds.find(2) != usedIds.end());
    EXPECT_TRUE(usedIds.find(3) != usedIds.end());
    EXPECT_TRUE(usedIds.find(4) != usedIds.end());
    EXPECT_TRUE(usedIds.find(5) != usedIds.end());

    // Check that duplicates are collected
    EXPECT_TRUE(duplicateIds.find(3) != duplicateIds.end());
    EXPECT_TRUE(duplicateIds.find(4) != duplicateIds.end());
}

TEST(TestBatchnormNode, PopulateHipdnnTensorIds)
{
    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());

    auto peerStat1 = std::make_shared<TensorAttributes>();
    auto peerStat2 = std::make_shared<TensorAttributes>();

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorLookup;
    std::unordered_set<int64_t> usedIds;
    int64_t currentTensorId = 1;

    auto error = node.populate_hipdnn_tensor_ids(tensorLookup, currentTensorId, usedIds);
    EXPECT_EQ(error.code, ErrorCode::OK);

    std::vector<std::shared_ptr<TensorAttributes>> tensors;
    tensors.reserve(node.attributes.inputs.size() + node.attributes.outputs.size()
                    + node.attributes.peer_stats.size());

    for(const auto& inputPair : node.attributes.inputs)
    {
        tensors.emplace_back(inputPair.second);
    }

    for(const auto& outputPair : node.attributes.outputs)
    {
        tensors.emplace_back(outputPair.second);
    }

    for(const auto& peerStat : node.attributes.peer_stats)
    {
        tensors.emplace_back(peerStat);
    }

    std::unordered_set<int64_t> tensorIds;
    for(const auto& tensor : tensors)
    {
        ASSERT_TRUE(tensor->has_uid());
        EXPECT_TRUE(tensorIds.insert(tensor->get_uid()).second)
            << "Duplicate tensor ID found: " << tensor->get_uid();
    }
}
