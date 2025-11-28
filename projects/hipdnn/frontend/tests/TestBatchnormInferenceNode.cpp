#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBatchnormInferenceNode, BatchnormInferenceNodeProperties)
{
    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());

    auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    GraphAttributes graphAttributes;
    BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);
    auto error = node.infer_properties_node();

    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormInferenceNode, PreValidateNode)
{
    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttributes;
    BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormInferenceNode, PreValidateNodeMissingValues)
{
    BatchnormInferenceAttributes batchnormAttributes;

    GraphAttributes graphAttributes;
    BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    auto batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNode nodeWithX(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNode nodeWithY(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithY.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNode nodeWithScale(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNode nodeWithBias(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithBias.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNode nodeWithMean(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithMean.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNode nodeWithAllValues(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormInferenceNode, InferPropertiesNode)
{
    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());

    auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    GraphAttributes graphAttributes;
    BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormInferenceNode, PackNode)
{
    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormInference");

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

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(3)
        .set_name("MeanTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(4)
        .set_name("InvVarianceTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_inv_variance(invVarianceTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(5)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(6)
        .set_name("BiasTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "BatchnormInference");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->y_tensor_uid(), yTensor->get_uid());
    EXPECT_EQ(packedAttributes->mean_tensor_uid(), meanTensor->get_uid());
    EXPECT_EQ(packedAttributes->inv_variance_tensor_uid(), invVarianceTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->bias_tensor_uid(), biasTensor->get_uid());
}

TEST(TestBatchnormInferenceNode, GatherHipdnnTensors)
{
    BatchnormInferenceAttributes bnAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("X");
    bnAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(2).set_name("Scale");
    bnAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(3).set_name("Bias");
    bnAttributes.set_bias(biasTensor);

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(4).set_name("Mean");
    bnAttributes.set_mean(meanTensor);

    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(5).set_name("InvVariance");
    bnAttributes.set_inv_variance(invVarianceTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(7).set_name("Y");
    bnAttributes.set_y(yTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNode node(std::move(bnAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(biasTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(meanTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(invVarianceTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());

    EXPECT_EQ(allTensors.size(), 6);
}
