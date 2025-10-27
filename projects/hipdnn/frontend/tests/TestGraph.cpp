// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_sdk/data_objects/graph_generated.h>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace ::testing;

namespace hipdnn_frontend
{

// Utility class to access private members of Graph for testing purposes
class GraphTestUtils : public Graph
{
public:
    GraphTestUtils() = default;

    std::vector<std::shared_ptr<INode>>& getPrivateGraphSubnodes()
    {
        return _sub_nodes;
    }
};
}

class TestGraph : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;
    hipdnnHandle_t _handle;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);
        _handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    }
    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    void expectGraphSerializedToBackendDescriptor(
        std::unique_ptr<hipdnn_sdk::data_objects::GraphT>& deserializedGraph)
    {
        EXPECT_CALL(*_mockBackend,
                    backendCreateAndDeserializeGraphExt(::testing::_, ::testing::_, ::testing::_))
            .WillOnce([&deserializedGraph]([[maybe_unused]] hipdnnBackendDescriptor_t* descriptor,
                                           const uint8_t* serializedGraph,
                                           size_t graphByteSize) {
                deserializedGraph = hipdnn_sdk::data_objects::UnPackGraph(serializedGraph);
                EXPECT_NE(deserializedGraph, nullptr);
                EXPECT_GE(graphByteSize, 0);
                return HIPDNN_STATUS_SUCCESS;
            });
    }

    static std::shared_ptr<TensorAttributes> createBasicBatchnormGraph(Graph& graph)
    {
        graph.set_name("SerializedGraphTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::HALF)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(1)
            .set_name("X")
            .set_dim({1, 2, 3, 4})
            .set_stride({5, 6, 7, 8})
            .set_data_type(DataType::FLOAT);

        auto mean = std::make_shared<TensorAttributes>();
        mean->set_uid(2).set_name("Mean").set_data_type(DataType::FLOAT);

        auto invVariance = std::make_shared<TensorAttributes>();
        invVariance->set_uid(3).set_name("InvVariance").set_data_type(DataType::FLOAT);

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(4).set_name("Scale").set_data_type(DataType::FLOAT);

        auto bias = std::make_shared<TensorAttributes>();
        bias->set_uid(5).set_name("Bias").set_data_type(DataType::FLOAT);

        BatchnormInferenceAttributes batchnormAttributes;
        batchnormAttributes.set_name("BatchnormNode");

        return graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);
    }
};

TEST_F(TestGraph, SetAndGetAttributes)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    EXPECT_EQ(graph.get_name(), "TestGraph");
    EXPECT_EQ(graph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(graph.get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(graph.get_io_data_type(), DataType::FLOAT);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BatchnormNodeCreation)
{
    Graph graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    BatchnormAttributes attributes;
    attributes.set_name("BatchnormNode");
    attributes.set_epsilon(std::make_shared<TensorAttributes>());

    auto [y, mean, invVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    EXPECT_EQ(mean->get_name(), "BatchnormNode::MEAN");
    EXPECT_TRUE(mean->get_is_virtual());

    EXPECT_EQ(invVariance->get_name(), "BatchnormNode::INV_VARIANCE");
    EXPECT_TRUE(invVariance->get_is_virtual());

    EXPECT_FALSE(nextRunningMean);
    EXPECT_FALSE(nextRunningVariance);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BatchnormBackwardNodeCreation)
{
    Graph graph;

    auto dy = std::make_shared<TensorAttributes>();
    auto x = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();

    dy->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    BatchnormBackwardAttributes attributes;
    attributes.set_name("BatchnormBackwardNode");

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, attributes);

    EXPECT_EQ(dx->get_name(), "BatchnormBackwardNode::DX");
    EXPECT_TRUE(dx->get_is_virtual());

    EXPECT_EQ(dscale->get_name(), "BatchnormBackwardNode::DSCALE");
    EXPECT_TRUE(dscale->get_is_virtual());

    EXPECT_EQ(dbias->get_name(), "BatchnormBackwardNode::DBIAS");
    EXPECT_TRUE(dbias->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BatchnormInferenceNodeCreation)
{
    Graph graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");

    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, PointwiseNodeCreationSingleInput)
{
    Graph graph;

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    EXPECT_EQ(out0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out0->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, PointwiseNodeCreationTwoInputs)
{
    Graph graph;

    auto in0 = std::make_shared<TensorAttributes>();
    auto in1 = std::make_shared<TensorAttributes>();

    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    in1->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::ADD);

    auto out0 = graph.pointwise(in0, in1, attributes);

    EXPECT_EQ(out0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out0->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, PointwiseNodeCreationThreeInputs)
{
    Graph graph;

    auto in0 = std::make_shared<TensorAttributes>();
    auto in1 = std::make_shared<TensorAttributes>();
    auto in2 = std::make_shared<TensorAttributes>();

    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    in1->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    in2->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::BINARY_SELECT);

    auto out0 = graph.pointwise(in0, in1, in2, attributes);

    EXPECT_EQ(out0->get_name(), "PointwiseNode::OUT_0");
    EXPECT_TRUE(out0->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ConvolutionFwdNodeCreation)
{
    Graph graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 3, 32, 32}).set_stride({3072, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_dim({64, 3, 3, 3}).set_stride({27, 9, 3, 1}).set_data_type(DataType::FLOAT);

    ConvFpropAttributes attributes;
    attributes.set_name("ConvolutionFpropNode");
    attributes.set_pre_padding({1, 1});
    attributes.set_post_padding({1, 1});
    attributes.set_stride({1, 1});
    attributes.set_dilation({1, 1});

    auto y = graph.conv_fprop(x, w, attributes);

    EXPECT_EQ(y->get_name(), "ConvolutionFpropNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ConvolutionDgradNodeCreation)
{
    Graph graph;

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_dim({1, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_dim({64, 3, 3, 3}).set_stride({27, 9, 3, 1}).set_data_type(DataType::FLOAT);

    ConvDgradAttributes attributes;
    attributes.set_name("ConvolutionDgradNode");
    attributes.set_pre_padding({1, 1});
    attributes.set_post_padding({1, 1});
    attributes.set_stride({1, 1});
    attributes.set_dilation({1, 1});

    auto dx = graph.conv_dgrad(dy, w, attributes);

    EXPECT_EQ(dx->get_name(), "ConvolutionDgradNode::DX");
    EXPECT_TRUE(dx->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

static void validateTensor(const TensorAttributes& tensor,
                           const hipdnn_sdk::data_objects::TensorAttributesT& serializedTensor)
{
    EXPECT_EQ(tensor.get_name(), serializedTensor.name);
    EXPECT_EQ(tensor.get_uid(), serializedTensor.uid);
    EXPECT_EQ(toSdkType(tensor.get_data_type()), serializedTensor.data_type);
    EXPECT_EQ(tensor.get_dim(), serializedTensor.dims);
    EXPECT_EQ(tensor.get_stride(), serializedTensor.strides);
}

TEST_F(TestGraph, BuildAndSerializeBatchnormInferenceGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2).set_name("Mean").set_data_type(DataType::FLOAT);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3).set_name("InvVariance").set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4).set_name("Scale").set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5).set_name("Bias").set_data_type(DataType::FLOAT);

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 6);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*mean, tensorLookup[mean->get_uid()]);
    validateTensor(*invVariance, tensorLookup[invVariance->get_uid()]);
    validateTensor(*scale, tensorLookup[scale->get_uid()]);
    validateTensor(*bias, tensorLookup[bias->get_uid()]);
    validateTensor(*y, tensorLookup[y->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "BatchnormNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
    auto deserializedBatchnormAttributes
        = deserializedGraph->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    EXPECT_EQ(deserializedBatchnormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->inv_variance_tensor_uid, invVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->y_tensor_uid, y->get_uid());
}

TEST_F(TestGraph, BuildAndSerializeBatchnormGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedBatchnormGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2).set_name("Scale").set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3).set_name("Bias").set_data_type(DataType::FLOAT);

    auto prevRunningMean = std::make_shared<TensorAttributes>();
    prevRunningMean->set_uid(4).set_name("PrevRunningMean").set_data_type(DataType::FLOAT);

    auto prevRunningVariance = std::make_shared<TensorAttributes>();
    prevRunningVariance->set_uid(5).set_name("PrevRunningVariance").set_data_type(DataType::FLOAT);

    auto momentum = std::make_shared<TensorAttributes>();
    momentum->set_uid(6).set_name("Momentum").set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(7).set_name("Epsilon").set_data_type(DataType::FLOAT);

    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");
    batchnormAttributes.set_previous_running_stats(prevRunningMean, prevRunningVariance, momentum);
    batchnormAttributes.set_epsilon(epsilon);

    auto [y, mean, invVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedBatchnormGraph");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 12);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*scale, tensorLookup[scale->get_uid()]);
    validateTensor(*bias, tensorLookup[bias->get_uid()]);
    validateTensor(*epsilon, tensorLookup[epsilon->get_uid()]);
    validateTensor(*prevRunningMean, tensorLookup[prevRunningMean->get_uid()]);
    validateTensor(*prevRunningVariance, tensorLookup[prevRunningVariance->get_uid()]);
    validateTensor(*momentum, tensorLookup[momentum->get_uid()]);
    validateTensor(*y, tensorLookup[y->get_uid()]);
    validateTensor(*mean, tensorLookup[mean->get_uid()]);
    validateTensor(*invVariance, tensorLookup[invVariance->get_uid()]);
    validateTensor(*nextRunningMean, tensorLookup[nextRunningMean->get_uid()]);
    validateTensor(*nextRunningVariance, tensorLookup[nextRunningVariance->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "BatchnormNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes);
    auto deserializedBatchnormAttributes
        = deserializedGraph->nodes[0]->attributes.AsBatchnormAttributes();
    EXPECT_EQ(deserializedBatchnormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->epsilon_tensor_uid, epsilon->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->prev_running_mean_tensor_uid,
              prevRunningMean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->prev_running_variance_tensor_uid,
              prevRunningVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->momentum_tensor_uid, momentum->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->y_tensor_uid, y->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->inv_variance_tensor_uid, invVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->next_running_mean_tensor_uid,
              nextRunningMean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->next_running_variance_tensor_uid,
              nextRunningVariance->get_uid());
}

TEST_F(TestGraph, BuildAndSerializeBatchnormAndPointwiseGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedBatchnormAndPointwiseGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2).set_name("Scale").set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3).set_name("Bias").set_data_type(DataType::FLOAT);

    auto prevRunningMean = std::make_shared<TensorAttributes>();
    prevRunningMean->set_uid(4).set_name("PrevRunningMean").set_data_type(DataType::FLOAT);

    auto prevRunningVariance = std::make_shared<TensorAttributes>();
    prevRunningVariance->set_uid(5).set_name("PrevRunningVariance").set_data_type(DataType::FLOAT);

    auto momentum = std::make_shared<TensorAttributes>();
    momentum->set_uid(6).set_name("Momentum").set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(7).set_name("Epsilon").set_data_type(DataType::FLOAT);

    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");
    batchnormAttributes.set_previous_running_stats(prevRunningMean, prevRunningVariance, momentum);
    batchnormAttributes.set_epsilon(epsilon);

    auto [y, mean, invVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, batchnormAttributes);

    PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_name("PointwiseNode");
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(y, pointwiseAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedBatchnormAndPointwiseGraph");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 13);
    EXPECT_EQ(deserializedGraph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*scale, tensorLookup[scale->get_uid()]);
    validateTensor(*bias, tensorLookup[bias->get_uid()]);
    validateTensor(*epsilon, tensorLookup[epsilon->get_uid()]);
    validateTensor(*prevRunningMean, tensorLookup[prevRunningMean->get_uid()]);
    validateTensor(*prevRunningVariance, tensorLookup[prevRunningVariance->get_uid()]);
    validateTensor(*momentum, tensorLookup[momentum->get_uid()]);
    validateTensor(*y, tensorLookup[y->get_uid()]);
    validateTensor(*mean, tensorLookup[mean->get_uid()]);
    validateTensor(*invVariance, tensorLookup[invVariance->get_uid()]);
    validateTensor(*nextRunningMean, tensorLookup[nextRunningMean->get_uid()]);
    validateTensor(*nextRunningVariance, tensorLookup[nextRunningVariance->get_uid()]);
    validateTensor(*out0, tensorLookup[out0->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "BatchnormNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes);
    auto deserializedBatchnormAttributes
        = deserializedGraph->nodes[0]->attributes.AsBatchnormAttributes();
    EXPECT_EQ(deserializedBatchnormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->epsilon_tensor_uid, epsilon->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->prev_running_mean_tensor_uid,
              prevRunningMean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->prev_running_variance_tensor_uid,
              prevRunningVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->momentum_tensor_uid, momentum->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->y_tensor_uid, y->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->inv_variance_tensor_uid, invVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->next_running_mean_tensor_uid,
              nextRunningMean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->next_running_variance_tensor_uid,
              nextRunningVariance->get_uid());

    EXPECT_EQ(deserializedGraph->nodes[1]->name, "PointwiseNode");
    EXPECT_EQ(deserializedGraph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    auto deserializedPointwiseAttributes
        = deserializedGraph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserializedPointwiseAttributes->in_0_tensor_uid, y->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->out_0_tensor_uid, out0->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD);
}

TEST_F(TestGraph, BuildAndSerializePointwiseGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(1)
        .set_name("Input0")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_name("PointwiseNode");
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, pointwiseAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 2);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*in0, tensorLookup[in0->get_uid()]);
    validateTensor(*out0, tensorLookup[out0->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "PointwiseNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    auto deserializedPointwiseAttributes
        = deserializedGraph->nodes[0]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserializedPointwiseAttributes->in_0_tensor_uid, in0->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->out_0_tensor_uid, out0->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD);
}

TEST_F(TestGraph, BuildAndSerializePointwiseAndBatchnormInferenceGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2).set_name("Mean").set_data_type(DataType::FLOAT);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3).set_name("InvVariance").set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4).set_name("Scale").set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5).set_name("Bias").set_data_type(DataType::FLOAT);

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);

    PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_name("PointwiseNode");
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(y, pointwiseAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 7);
    EXPECT_EQ(deserializedGraph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*mean, tensorLookup[mean->get_uid()]);
    validateTensor(*invVariance, tensorLookup[invVariance->get_uid()]);
    validateTensor(*scale, tensorLookup[scale->get_uid()]);
    validateTensor(*bias, tensorLookup[bias->get_uid()]);
    validateTensor(*y, tensorLookup[y->get_uid()]);
    validateTensor(*out0, tensorLookup[out0->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "BatchnormNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
    auto deserializedBatchnormAttributes
        = deserializedGraph->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    EXPECT_EQ(deserializedBatchnormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->inv_variance_tensor_uid, invVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->y_tensor_uid, y->get_uid());

    EXPECT_EQ(deserializedGraph->nodes[1]->name, "PointwiseNode");
    EXPECT_EQ(deserializedGraph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    auto deserializedPointwiseAttributes
        = deserializedGraph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserializedPointwiseAttributes->in_0_tensor_uid, y->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->out_0_tensor_uid, out0->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD);
}

TEST_F(TestGraph, BuildAndSerializeBatchnormBackwardGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(1)
        .set_name("Dy")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(2)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(3).set_name("Scale").set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(4).set_name("Mean").set_data_type(DataType::FLOAT);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(5).set_name("InvVariance").set_data_type(DataType::FLOAT);

    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormBackwardNode");
    batchnormAttributes.set_saved_mean_and_inv_variance(mean, invVariance);

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 8);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*dy, tensorLookup[dy->get_uid()]);
    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*scale, tensorLookup[scale->get_uid()]);
    validateTensor(*mean, tensorLookup[mean->get_uid()]);
    validateTensor(*invVariance, tensorLookup[invVariance->get_uid()]);
    validateTensor(*dx, tensorLookup[dx->get_uid()]);
    validateTensor(*dscale, tensorLookup[dscale->get_uid()]);
    validateTensor(*dbias, tensorLookup[dbias->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "BatchnormBackwardNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes);
    auto deserializedBatchnormAttributes
        = deserializedGraph->nodes[0]->attributes.AsBatchnormBackwardAttributes();
    EXPECT_EQ(deserializedBatchnormAttributes->dy_tensor_uid, dy->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->inv_variance_tensor_uid, invVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->dx_tensor_uid, dx->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->dscale_tensor_uid, dscale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->dbias_tensor_uid, dbias->get_uid());
}

TEST_F(TestGraph, BuildAndSerializeConvolutionFwdGraph)
{
    Graph graph;

    graph.set_name("SerializedConvolutionGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 3, 32, 32})
        .set_stride({3072, 1024, 32, 1})
        .set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim({64, 3, 3, 3})
        .set_stride({27, 9, 3, 1})
        .set_data_type(DataType::FLOAT);

    ConvFpropAttributes convolutionAttributes;
    convolutionAttributes.set_name("ConvolutionFpropNode");
    convolutionAttributes.set_pre_padding({1, 1});
    convolutionAttributes.set_post_padding({1, 1});
    convolutionAttributes.set_stride({1, 1});
    convolutionAttributes.set_dilation({1, 1});

    auto y = graph.conv_fprop(x, w, convolutionAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedConvolutionGraph");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 3);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*w, tensorLookup[w->get_uid()]);
    validateTensor(*y, tensorLookup[y->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "ConvolutionFpropNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes);
    auto deserializedConvolutionAttributes
        = deserializedGraph->nodes[0]->attributes.AsConvolutionFwdAttributes();
    EXPECT_EQ(deserializedConvolutionAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedConvolutionAttributes->w_tensor_uid, w->get_uid());
    EXPECT_EQ(deserializedConvolutionAttributes->y_tensor_uid, y->get_uid());
    EXPECT_EQ(deserializedConvolutionAttributes->pre_padding, std::vector<int64_t>({1, 1}));
    EXPECT_EQ(deserializedConvolutionAttributes->post_padding, std::vector<int64_t>({1, 1}));
    EXPECT_EQ(deserializedConvolutionAttributes->stride, std::vector<int64_t>({1, 1}));
    EXPECT_EQ(deserializedConvolutionAttributes->dilation, std::vector<int64_t>({1, 1}));
}

TEST_F(TestGraph, BuildAndSerializeConvolutionDgradGraph)
{
    Graph graph;

    graph.set_name("SerializedConvolutionDgradGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(1)
        .set_name("DY")
        .set_dim({1, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1})
        .set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim({64, 3, 3, 3})
        .set_stride({27, 9, 3, 1})
        .set_data_type(DataType::FLOAT);

    ConvDgradAttributes convolutionAttributes;
    convolutionAttributes.set_name("ConvolutionDgradNode");
    convolutionAttributes.set_pre_padding({1, 1});
    convolutionAttributes.set_post_padding({1, 1});
    convolutionAttributes.set_stride({1, 1});
    convolutionAttributes.set_dilation({1, 1});

    auto dx = graph.conv_dgrad(dy, w, convolutionAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedConvolutionDgradGraph");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 3);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*dy, tensorLookup[dy->get_uid()]);
    validateTensor(*w, tensorLookup[w->get_uid()]);
    validateTensor(*dx, tensorLookup[dx->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "ConvolutionDgradNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes);
    auto deserializedConvolutionAttributes
        = deserializedGraph->nodes[0]->attributes.AsConvolutionBwdAttributes();
    EXPECT_EQ(deserializedConvolutionAttributes->dy_tensor_uid, dy->get_uid());
    EXPECT_EQ(deserializedConvolutionAttributes->w_tensor_uid, w->get_uid());
    EXPECT_EQ(deserializedConvolutionAttributes->dx_tensor_uid, dx->get_uid());
    EXPECT_EQ(deserializedConvolutionAttributes->pre_padding, std::vector<int64_t>({1, 1}));
    EXPECT_EQ(deserializedConvolutionAttributes->post_padding, std::vector<int64_t>({1, 1}));
    EXPECT_EQ(deserializedConvolutionAttributes->stride, std::vector<int64_t>({1, 1}));
    EXPECT_EQ(deserializedConvolutionAttributes->dilation, std::vector<int64_t>({1, 1}));
}

TEST_F(TestGraph, BuildAndSerializePointwiseAndBatchnormBackwardGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto xPointwise = std::make_shared<TensorAttributes>();
    xPointwise->set_uid(6)
        .set_name("X_Pointwise")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_name("PointwiseNode");
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto dy = graph.pointwise(xPointwise, pointwiseAttributes);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2).set_name("Scale").set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(3).set_name("Mean").set_data_type(DataType::FLOAT);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(4).set_name("InvVariance").set_data_type(DataType::FLOAT);

    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormBackwardNode");
    batchnormAttributes.set_saved_mean_and_inv_variance(mean, invVariance);

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_type, hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 9);
    EXPECT_EQ(deserializedGraph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*xPointwise, tensorLookup[xPointwise->get_uid()]);
    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*scale, tensorLookup[scale->get_uid()]);
    validateTensor(*mean, tensorLookup[mean->get_uid()]);
    validateTensor(*invVariance, tensorLookup[invVariance->get_uid()]);
    validateTensor(*dy, tensorLookup[dy->get_uid()]);
    validateTensor(*dx, tensorLookup[dx->get_uid()]);
    validateTensor(*dscale, tensorLookup[dscale->get_uid()]);
    validateTensor(*dbias, tensorLookup[dbias->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "PointwiseNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    auto deserializedPointwiseAttributes
        = deserializedGraph->nodes[0]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserializedPointwiseAttributes->in_0_tensor_uid, xPointwise->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->out_0_tensor_uid, dy->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->operation,
              hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD);

    EXPECT_EQ(deserializedGraph->nodes[1]->name, "BatchnormBackwardNode");
    EXPECT_EQ(deserializedGraph->nodes[1]->attributes.type,
              hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes);
    auto deserializedBatchnormAttributes
        = deserializedGraph->nodes[1]->attributes.AsBatchnormBackwardAttributes();
    EXPECT_EQ(deserializedBatchnormAttributes->dy_tensor_uid, dy->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->inv_variance_tensor_uid, invVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->dx_tensor_uid, dx->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->dscale_tensor_uid, dscale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->dbias_tensor_uid, dbias->get_uid());
}

// Test graph.tensor()
TEST_F(TestGraph, TensorGraphAttributes)
{
    auto tensor = Graph::tensor(TensorAttributes()
                                    .set_name("TestTensor")
                                    .set_uid(100)
                                    .set_stride({5, 6, 7, 8})
                                    .set_data_type(DataType::FLOAT)
                                    .set_is_virtual(false)
                                    .set_dim({1, 2, 3, 4}));

    EXPECT_EQ(tensor->get_data_type(), DataType::FLOAT);
    EXPECT_FALSE(tensor->get_is_virtual());
    EXPECT_EQ(tensor->get_dim(), std::vector<int64_t>({1, 2, 3, 4}));
    EXPECT_EQ(tensor->get_stride(), std::vector<int64_t>({5, 6, 7, 8}));
    EXPECT_EQ(tensor->get_name(), "TestTensor");
    EXPECT_EQ(tensor->get_uid(), 100);
}

// Test graph.tensorLike()
TEST_F(TestGraph, TensorLikeGraphAttributes)
{
    auto tensor = Graph::tensor(TensorAttributes()
                                    .set_name("TestTensor")
                                    .set_uid(100)
                                    .set_dim({1, 2, 3, 4})
                                    .set_stride({5, 6, 7, 8})
                                    .set_is_virtual(false)
                                    .set_data_type(DataType::FLOAT));

    auto tensorLike = Graph::tensor_like(tensor, "TensorLike");

    EXPECT_EQ(tensorLike->get_data_type(), DataType::FLOAT);
    EXPECT_FALSE(tensorLike->get_is_virtual());
    EXPECT_EQ(tensorLike->get_dim(), std::vector<int64_t>({1, 2, 3, 4}));
    EXPECT_EQ(tensorLike->get_stride(), std::vector<int64_t>({5, 6, 7, 8}));
    EXPECT_EQ(tensorLike->get_name(), "TensorLike");
    EXPECT_NE(tensorLike->get_uid(), 100);

    EXPECT_NE(tensorLike, tensor);

    auto tensorLikeNoName = Graph::tensor_like(tensorLike);
    EXPECT_EQ(tensorLikeNoName->get_name(), "");

    EXPECT_EQ(tensor->get_name(), "TestTensor");
    EXPECT_EQ(tensor->get_uid(), 100);
    EXPECT_NE(tensor->get_uid(), tensorLikeNoName->get_uid());
}

TEST_F(TestGraph, WillCorrectlyBuildOperationGraphDescriptor)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);

    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce([this](hipdnnBackendDescriptor_t,
                         hipdnnBackendAttributeName_t,
                         hipdnnBackendAttributeType_t,
                         int64_t,
                         const void* arrayOfElements) {
            hipdnnHandle_t handle = *static_cast<const hipdnnHandle_t*>(arrayOfElements);
            EXPECT_EQ(handle, this->_handle);
            return HIPDNN_STATUS_SUCCESS;
        });

    auto result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(result.is_good());
}

TEST_F(TestGraph, CreatingExecutionPlansFailsWithNoGraph)
{
    Graph graph;

    auto result = graph.create_execution_plans({HeuristicMode::FALLBACK});
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.get_message(),
              "Graph has not been built, build the operation graph first. Cannot create "
              "execution plan.");
}

TEST_F(TestGraph, CanSuccessfullyCreateExecutionPlans)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    std::vector<hipdnnBackendHeurMode_t> backendModes;
    backendModes.reserve(heurModes.size());
    for(const auto& mode : heurModes)
    {
        backendModes.push_back(toBackendType(mode));
    }
    auto tensorAttributes = createBasicBatchnormGraph(graph);

    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    graph.build_operation_graph(_handle);

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t,
                               hipdnnBackendAttributeName_t,
                               hipdnnBackendAttributeType_t,
                               int64_t,
                               const void* arrayOfElements) {
            EXPECT_EQ(graphDesc, *static_cast<const hipdnnBackendDescriptor_t*>(arrayOfElements));
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce([&backendModes](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t count,
                                  const void* arrayOfElements) {
            EXPECT_EQ(count, static_cast<int64_t>(backendModes.size()));
            auto modesPtr = static_cast<const hipdnnBackendHeurMode_t*>(arrayOfElements);
            for(size_t i = 0; i < backendModes.size(); ++i)
            {
                EXPECT_EQ(modesPtr[i], backendModes[i]);
            }
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    // Set up the mock to handle multiple calls with different arguments using .WillOnce()/.WillRepeatedly()
    // First call: elementCount query
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto execPlanResult = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(execPlanResult.is_good());
}

TEST_F(TestGraph, CheckSupportFailsIfNoExecutionPlanCreated)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    auto tensorAttributes = createBasicBatchnormGraph(graph);

    graph.build_operation_graph(_handle);

    auto result = graph.check_support();
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.get_message(), "Execution plan descriptor is not created or invalid.");
}

TEST_F(TestGraph, CheckSupportSucceedsWhenExecutionPlanCreated)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    graph.build_operation_graph(_handle);

    ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* elementCount,
                           void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    graph.create_execution_plans(heurModes);

    auto result = graph.check_support();
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(result.get_message(), "");
}

TEST_F(TestGraph, EngineConfigAndExecutionPlanAreFinalizedAfterBuildPlans)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    auto tensorAttributes = createBasicBatchnormGraph(graph);

    ON_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto result = graph.build_operation_graph(_handle);
    EXPECT_TRUE(result.is_good());

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* elementCount,
                           void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    result = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(result.is_good());

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(executionPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      const void* arrayOfElements) {
            EXPECT_EQ(engineConfigDesc,
                      *static_cast<const hipdnnBackendDescriptor_t*>(arrayOfElements));
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(executionPlanDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    result = graph.build_plans();
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(result.get_message(), "");
}

TEST_F(TestGraph, WorkspaceSizeIsRetrievedFromExecutionPlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    auto tensorAttributes = createBasicBatchnormGraph(graph);
    graph.build_operation_graph(_handle);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, _, _, _, _, _))
        .WillRepeatedly([](hipdnnBackendDescriptor_t,
                           hipdnnBackendAttributeName_t,
                           hipdnnBackendAttributeType_t,
                           int64_t,
                           int64_t* elementCount,
                           void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    graph.create_execution_plans(heurModes);

    int64_t workspaceSize = 123454;
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(executionPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([workspaceSize](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = workspaceSize;
            return HIPDNN_STATUS_SUCCESS;
        });

    int64_t workspaceSizeResult = 0;
    auto result = graph.get_workspace_size(workspaceSizeResult);

    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(workspaceSizeResult, workspaceSize);
}

TEST_F(TestGraph, ExecutePacksVariantPackAndPassesTheCorrectArguments)
{
    ::testing::FLAGS_gmock_verbose = "error";
    using ::testing::_;
    using ::testing::Invoke;
    using ::testing::NotNull;
    using ::testing::Return;

    Graph graph;
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(42)
        .set_name("InputTensor")
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_name("PointwiseNode");
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);
    auto outTensor = graph.pointwise(tensor, pointwiseAttributes);

    // build_operation_graph mocks
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1000);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* desc, const uint8_t*, size_t) {
            *desc = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // create_execution_plans mocks
    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2000);
    auto engineCfgDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3000);
    auto execPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4000);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce([&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = heurDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            heurDesc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* count,
                     void*) {
            *count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineCfgDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = engineCfgDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* count,
                     void*) {
            *count = 1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&execPlanDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
            *desc = execPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // build_plans mocks
    EXPECT_CALL(*_mockBackend, backendFinalize(engineCfgDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(execPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(execPlanDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // get_workspace_size mock
    int64_t expectedWorkspaceSize = 12345;
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(execPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                    HIPDNN_TYPE_INT64,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([expectedWorkspaceSize](hipdnnBackendDescriptor_t,
                                          hipdnnBackendAttributeName_t,
                                          hipdnnBackendAttributeType_t,
                                          int64_t,
                                          int64_t*,
                                          void* ptr) {
            *reinterpret_cast<int64_t*>(ptr) = expectedWorkspaceSize;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Prepare variant pack and workspace for execute
    auto tensor1 = std::make_shared<TensorAttributes>();
    tensor1->set_uid(42);

    auto tensor2 = std::make_shared<TensorAttributes>();
    tensor2->set_uid(22);

    auto tensor3 = std::make_shared<TensorAttributes>();
    tensor3->set_uid(33);

    auto tensor4 = std::make_shared<TensorAttributes>();
    tensor4->set_uid(1);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[tensor1->get_uid()] = reinterpret_cast<void*>(0xDEADBEEF);
    variantPack[tensor2->get_uid()] = reinterpret_cast<void*>(0xBEEFBEEF);
    variantPack[tensor3->get_uid()] = reinterpret_cast<void*>(0xBEEFDEAD);
    variantPack[tensor4->get_uid()] = reinterpret_cast<void*>(0xDEADBEE);

    std::unordered_map<std::shared_ptr<TensorAttributes>, void*> variantPackForExec;
    variantPackForExec[tensor1] = reinterpret_cast<void*>(0xDEADBEEF);
    variantPackForExec[tensor2] = reinterpret_cast<void*>(0xBEEFBEEF);
    variantPackForExec[tensor3] = reinterpret_cast<void*>(0xBEEFDEAD);
    variantPackForExec[tensor4] = reinterpret_cast<void*>(0xDEADBEE);

    void* workspace = reinterpret_cast<void*>(0xCAFEBABE);

    auto variantPackDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5000);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, _))
        .WillOnce(
            [&variantPackDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = variantPackDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                    HIPDNN_TYPE_VOID_PTR,
                                    static_cast<int64_t>(variantPack.size()),
                                    NotNull()))
        .WillOnce(Invoke([variantPack](hipdnnBackendDescriptor_t,
                                       hipdnnBackendAttributeName_t,
                                       hipdnnBackendAttributeType_t,
                                       int64_t count,
                                       const void* ptr) {
            EXPECT_EQ(count, 4);
            auto dataPtrs = static_cast<void* const*>(ptr);
            for(int i = 0; i < 4; i++)
            {
                auto targetValue = dataPtrs[i];
                auto it = std::find_if(
                    variantPack.begin(), variantPack.end(), [&targetValue](const auto& pair) {
                        return pair.second == targetValue;
                    });
                EXPECT_TRUE(it != variantPack.end());
            }

            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                    HIPDNN_TYPE_INT64,
                                    static_cast<int64_t>(variantPack.size()),
                                    NotNull()))
        .WillOnce(Invoke([variantPack](hipdnnBackendDescriptor_t,
                                       hipdnnBackendAttributeName_t,
                                       hipdnnBackendAttributeType_t,
                                       int64_t count,
                                       const void* ptr) {
            EXPECT_EQ(count, 4);
            auto keys = static_cast<const int64_t*>(ptr);
            for(int i = 0; i < 4; i++)
            {
                EXPECT_TRUE(variantPack.find(keys[i]) != variantPack.end());
            }
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(variantPackDesc,
                                    HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                    HIPDNN_TYPE_VOID_PTR,
                                    1,
                                    NotNull()))
        .WillOnce(Invoke([workspace](hipdnnBackendDescriptor_t,
                                     hipdnnBackendAttributeName_t,
                                     hipdnnBackendAttributeType_t,
                                     int64_t count,
                                     const void* ptr) {
            EXPECT_EQ(count, 1);
            auto workspacePtr = *static_cast<void* const*>(ptr);
            EXPECT_EQ(workspacePtr, workspace);
            return HIPDNN_STATUS_SUCCESS;
        }));
    EXPECT_CALL(*_mockBackend, backendFinalize(variantPackDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendExecute(_handle, execPlanDesc, variantPackDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Run the full sequence
    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    auto planResult = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(planResult.is_good());

    auto supportResult = graph.check_support();
    EXPECT_TRUE(supportResult.is_good());

    auto buildPlansResult = graph.build_plans();
    EXPECT_TRUE(buildPlansResult.is_good());

    int64_t workspaceSize = 0;
    auto wsResult = graph.get_workspace_size(workspaceSize);
    EXPECT_TRUE(wsResult.is_good());
    EXPECT_EQ(workspaceSize, expectedWorkspaceSize);

    auto execResult = graph.execute(_handle, variantPackForExec, workspace);
    EXPECT_TRUE(execResult.is_good());
}

TEST_F(TestGraph, TopologicalSortSucceedsOnNormalGraph)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    EXPECT_TRUE(graph.topologicallySortGraph().is_good());
}

TEST_F(TestGraph, TopologicalSortFailsWithOrphanedNode)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    // Connected nodes
    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    // Orphaned node (not connected to the main graph)
    auto x2 = std::make_shared<TensorAttributes>();
    x2->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x2->set_uid(2);
    BatchnormInferenceAttributes attributes3;
    attributes3.set_name("BatchnormNode3");
    auto y3 = graph.batchnorm_inference(x2, mean, invVariance, scale, bias, attributes3);

    EXPECT_FALSE(graph.topologicallySortGraph().is_good());
}

TEST_F(TestGraph, TopologicalSortFailsOnCircularDependency)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    // Introduce a cycle: set x of the first node to y2
    auto& subNodes = graph.getPrivateGraphSubnodes();
    auto* batchNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(batchNode, nullptr);
    batchNode->attributes.set_x(y2);

    EXPECT_FALSE(graph.topologicallySortGraph().is_good());
}

TEST_F(TestGraph, ValidateSortsNodesTopologically)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    // Node 0: batchnorm1
    BatchnormInferenceAttributes bnAttrs1;
    bnAttrs1.set_name("batchnorm1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, bnAttrs1);

    // Node 1: pointwise1 (depends on batchnorm1)
    PointwiseAttributes pwAttrs1;
    pwAttrs1.set_name("pointwise1");
    pwAttrs1.set_mode(PointwiseMode::RELU_FWD);
    auto out1 = graph.pointwise(y1, pwAttrs1);

    // Node 2: pointwise2 (depends on batchnorm1)
    PointwiseAttributes pwAttrs2;
    pwAttrs2.set_name("pointwise2");
    pwAttrs2.set_mode(PointwiseMode::RELU_FWD);
    auto out2 = graph.pointwise(y1, pwAttrs2);

    //Node 3 Create a combined input by using ADD pointwise
    PointwiseAttributes pwAdd;
    pwAdd.set_name("add");
    pwAdd.set_mode(PointwiseMode::ADD);
    auto combined = graph.pointwise(out1, out2, pwAdd);

    // Node 4: batchnorm2 (depends on both pointwise1 and pointwise2)
    BatchnormInferenceAttributes bnAttrs2;
    bnAttrs2.set_name("batchnorm2");
    auto y2 = graph.batchnorm_inference(combined, mean, invVariance, scale, bias, bnAttrs2);

    // Node 5: pointwise3 (final node)
    PointwiseAttributes pwAttrs3;
    pwAttrs3.set_name("pointwise3");
    pwAttrs3.set_mode(PointwiseMode::RELU_FWD);
    auto out3 = graph.pointwise(y2, pwAttrs3);

    auto sortedSubnodesDueToGraphConstructionOrderCopy = graph.getPrivateGraphSubnodes();

    auto& subNodes = graph.getPrivateGraphSubnodes();
    // Randomize the node order to mess it up
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(subNodes.begin(), subNodes.end(), gen);

    //verify for sure the nodes arent sorted.
    bool notSortedAnymore = false;
    for(size_t i = 0; i < subNodes.size(); i++)
    {
        if(sortedSubnodesDueToGraphConstructionOrderCopy[i] != subNodes[i])
        {
            notSortedAnymore = true;
            break;
        }
    }
    EXPECT_TRUE(notSortedAnymore);

    auto result = graph.validate();
    EXPECT_TRUE(result.is_good()) << result.get_message();

    ASSERT_EQ(subNodes.size(), sortedSubnodesDueToGraphConstructionOrderCopy.size());
    EXPECT_EQ(subNodes[0], sortedSubnodesDueToGraphConstructionOrderCopy[0]);

    //due to randomiztion, its possible that these nodes swap positions because the graph is valid
    //regardless of the order of these two nodes because the graph has a diamond shape.
    if(subNodes[1] == sortedSubnodesDueToGraphConstructionOrderCopy[1])
    {
        EXPECT_EQ(subNodes[1], sortedSubnodesDueToGraphConstructionOrderCopy[1]);
        EXPECT_EQ(subNodes[2], sortedSubnodesDueToGraphConstructionOrderCopy[2]);
    }
    else
    {
        EXPECT_EQ(subNodes[1], sortedSubnodesDueToGraphConstructionOrderCopy[2]);
        EXPECT_EQ(subNodes[2], sortedSubnodesDueToGraphConstructionOrderCopy[1]);
    }

    EXPECT_EQ(subNodes[3], sortedSubnodesDueToGraphConstructionOrderCopy[3]);
    EXPECT_EQ(subNodes[4], sortedSubnodesDueToGraphConstructionOrderCopy[4]);
    EXPECT_EQ(subNodes[5], sortedSubnodesDueToGraphConstructionOrderCopy[5]);
}

TEST_F(TestGraph, ValidateFailsWithDuplicateTensorUids)
{
    GraphTestUtils graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    //validate graph is good.
    auto result = graph.validate();
    EXPECT_TRUE(result.is_good()) << result.get_message();

    //introduce duplicate uids
    mean->set_uid(1);
    invVariance->set_uid(1);
    scale->set_uid(1);
    bias->set_uid(2);
    y1->set_uid(2);
    y2->set_uid(3);

    result = graph.validate();
    EXPECT_FALSE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CheckNoDuplicateTensorIdsPassesWithNoDuplicates)
{
    GraphTestUtils graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2);
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3);
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5);

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    auto result = graph.checkNoDuplicateTensorIds();
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CheckNoDuplicateTensorIdsFailsWithDuplicates)
{
    GraphTestUtils graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2);
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(2); // Duplicate UID
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(3);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3); // Duplicate UID

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    auto result = graph.checkNoDuplicateTensorIds();
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(result.get_message().find("Duplicate tensor UIDs") != std::string::npos);
    EXPECT_TRUE(result.get_message().find('2') != std::string::npos);
    EXPECT_TRUE(result.get_message().find('3') != std::string::npos);
}

TEST_F(TestGraph, CheckNoDuplicateTensorIdsPassesWithReusedTensors)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    // These tensors will be reused across multiple batchnorm nodes
    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2);
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3);
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5);

    // Node 1: Uses mean, invVariance, scale, bias
    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    // Node 2: REUSES the same mean, invVariance, scale, bias tensors
    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    // Should pass - same tensor objects being reused is fine
    auto result = graph.checkNoDuplicateTensorIds();
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CheckNoDuplicateTensorIdsFailsWithReusedUidsOnDifferentTensors)
{
    GraphTestUtils graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    // Node 1 tensors
    auto mean1 = std::make_shared<TensorAttributes>();
    mean1->set_uid(2);
    auto invVariance1 = std::make_shared<TensorAttributes>();
    invVariance1->set_uid(3);
    auto scale1 = std::make_shared<TensorAttributes>();
    scale1->set_uid(4);
    auto bias1 = std::make_shared<TensorAttributes>();
    bias1->set_uid(5);

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean1, invVariance1, scale1, bias1, attributes1);

    // Node 2 tensors - DIFFERENT objects but SAME UIDs
    auto mean2 = std::make_shared<TensorAttributes>();
    mean2->set_uid(2); // Same UID as mean1 but different object
    auto invVariance2 = std::make_shared<TensorAttributes>();
    invVariance2->set_uid(3); // Same UID as invVariance1 but different object
    auto scale2 = std::make_shared<TensorAttributes>();
    scale2->set_uid(4); // Same UID as scale1 but different object
    auto bias2 = std::make_shared<TensorAttributes>();
    bias2->set_uid(5); // Same UID as bias1 but different object

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean2, invVariance2, scale2, bias2, attributes2);

    // Should fail - different tensor objects with same UIDs
    auto result = graph.checkNoDuplicateTensorIds();
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(result.get_message().find("Duplicate tensor UIDs") != std::string::npos);
}

TEST_F(TestGraph, BuildOperationGraphAllMissingTensorUids)
{
    Graph graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    auto invVariance = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();
    auto bias = std::make_shared<TensorAttributes>();

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    // Before build_operation_graph, UIDs should not be set
    EXPECT_FALSE(x->has_uid());
    EXPECT_FALSE(mean->has_uid());
    EXPECT_FALSE(invVariance->has_uid());
    EXPECT_FALSE(scale->has_uid());
    EXPECT_FALSE(bias->has_uid());
    EXPECT_FALSE(y->has_uid());

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    // After build_operation_graph, all UIDs should be populated
    EXPECT_TRUE(x->has_uid());
    EXPECT_TRUE(mean->has_uid());
    EXPECT_TRUE(invVariance->has_uid());
    EXPECT_TRUE(scale->has_uid());
    EXPECT_TRUE(bias->has_uid());
    EXPECT_TRUE(y->has_uid());

    // Verify all UIDs are unique
    std::unordered_set<int64_t> uids;
    uids.insert(x->get_uid());
    uids.insert(mean->get_uid());
    uids.insert(invVariance->get_uid());
    uids.insert(scale->get_uid());
    uids.insert(bias->get_uid());
    uids.insert(y->get_uid());

    EXPECT_EQ(uids.size(), 6);
}

TEST_F(TestGraph, BuildOperationGraphPopulatesOnlyMissingUids)
{
    Graph graph;

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(100); // Pre-set UID

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(200); // Pre-set UID

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(300); // Pre-set UID

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(400); // Pre-set UID

    auto bias = std::make_shared<TensorAttributes>();
    // bias does not have a UID set

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);
    y->set_uid(500); // Pre-set UID

    EXPECT_TRUE(x->has_uid());
    EXPECT_EQ(x->get_uid(), 100);
    EXPECT_TRUE(mean->has_uid());
    EXPECT_EQ(mean->get_uid(), 200);
    EXPECT_TRUE(invVariance->has_uid());
    EXPECT_EQ(invVariance->get_uid(), 300);
    EXPECT_TRUE(scale->has_uid());
    EXPECT_EQ(scale->get_uid(), 400);
    EXPECT_FALSE(bias->has_uid());
    EXPECT_TRUE(y->has_uid());
    EXPECT_EQ(y->get_uid(), 500);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    // After build_operation_graph, bias should have a UID
    EXPECT_TRUE(bias->has_uid());

    // All pre-existing UIDs should remain unchanged
    EXPECT_EQ(x->get_uid(), 100);
    EXPECT_EQ(mean->get_uid(), 200);
    EXPECT_EQ(invVariance->get_uid(), 300);
    EXPECT_EQ(scale->get_uid(), 400);
    EXPECT_EQ(y->get_uid(), 500);

    // The new UID for bias should be unique
    int64_t biasUid = bias->get_uid();
    EXPECT_NE(biasUid, 100);
    EXPECT_NE(biasUid, 200);
    EXPECT_NE(biasUid, 300);
    EXPECT_NE(biasUid, 400);
    EXPECT_NE(biasUid, 500);
}
