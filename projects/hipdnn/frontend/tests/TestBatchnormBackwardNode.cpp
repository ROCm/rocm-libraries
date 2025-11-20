// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBatchnormBackwardNode, PreValidateNode)
{
    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_dy(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttributes;
    BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormBackwardNode, PreValidateNodeMissingValues)
{
    BatchnormBackwardAttributes batchnormAttributes;

    GraphAttributes graphAttributes;
    BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_dy(std::make_shared<TensorAttributes>());
    auto batchnormAttributesCopy = batchnormAttributes;
    BatchnormBackwardNode nodeWithDy(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithDy.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormBackwardNode nodeWithX(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormBackwardNode nodeWithScale(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormBackwardNode nodeWithDx(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithDx.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormBackwardNode nodeWithDscale(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithDscale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormBackwardNode nodeWithAllValues(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormBackwardNode, InferPropertiesNode)
{
    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_dy(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    auto xTensor = batchnormAttributes.get_x();
    xTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto dxTensor = batchnormAttributes.get_dx();
    dxTensor->set_uid(2).set_name("DxTensor");

    auto dscaleTensor = batchnormAttributes.get_dscale();
    dscaleTensor->set_uid(3).set_name("DscaleTensor");

    auto dbiasTensor = batchnormAttributes.get_dbias();
    dbiasTensor->set_uid(4).set_name("DbiasTensor");

    GraphAttributes graphAttributes;
    BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(dxTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dxTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(dscaleTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(dscaleTensor->get_stride(), (std::vector<int64_t>{2, 1, 2, 2}));

    EXPECT_EQ(dbiasTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(dbiasTensor->get_stride(), (std::vector<int64_t>{2, 1, 2, 2}));
}

TEST(TestBatchnormBackwardNode, GatherHipdnnTensors)
{
    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_dy(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(9).set_name("PeerStat1");

    auto peerStat2 = std::make_shared<TensorAttributes>();
    peerStat2->set_uid(10).set_name("PeerStat2");

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    GraphAttributes graphAttributes;
    BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(peerStat1) != allTensors.end());
    EXPECT_TRUE(allTensors.find(peerStat2) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 10);
}

TEST(TestBatchnormBackwardNode, PackNode)
{
    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormBackward");

    // Set up tensor attributes
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(2)
        .set_name("XTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(4)
        .set_name("MeanTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(5)
        .set_name("InvVarianceTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_inv_variance(invVarianceTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(6)
        .set_name("DxTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_dx(dxTensor);

    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(7)
        .set_name("DscaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_dscale(dscaleTensor);

    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_uid(8)
        .set_name("DbiasTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_dbias(dbiasTensor);

    GraphAttributes graphAttributes;
    BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    // Pack the node
    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "BatchnormBackward");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->dy_tensor_uid(), dyTensor->get_uid());
    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->mean_tensor_uid(), meanTensor->get_uid());
    EXPECT_EQ(packedAttributes->inv_variance_tensor_uid(), invVarianceTensor->get_uid());
    EXPECT_EQ(packedAttributes->dx_tensor_uid(), dxTensor->get_uid());
    EXPECT_EQ(packedAttributes->dscale_tensor_uid(), dscaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->dbias_tensor_uid(), dbiasTensor->get_uid());
}

TEST(TestBatchnormBackwardNode, PackNodeWithoutMeanAndInvVariance)
{
    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormBackward");

    // Set up tensor attributes
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(2)
        .set_name("XTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(4)
        .set_name("DxTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_dx(dxTensor);

    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(5)
        .set_name("DscaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_dscale(dscaleTensor);

    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_uid(6)
        .set_name("DbiasTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_dbias(dbiasTensor);

    GraphAttributes graphAttributes;
    BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "BatchnormBackward");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->dy_tensor_uid(), dyTensor->get_uid());
    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->mean_tensor_uid(), flatbuffers::nullopt);
    EXPECT_EQ(packedAttributes->inv_variance_tensor_uid(), flatbuffers::nullopt);
    EXPECT_EQ(packedAttributes->dx_tensor_uid(), dxTensor->get_uid());
    EXPECT_EQ(packedAttributes->dscale_tensor_uid(), dscaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->dbias_tensor_uid(), dbiasTensor->get_uid());
}
