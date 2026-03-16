// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "fake_backend/BackendTestMatchers.hpp"
#include "fake_backend/MockHipdnnBackend.hpp"

#include <algorithm>
#include <cstring>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::test;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using namespace ::testing;

namespace hipdnn_frontend
{

// Static assert checks to verify Move and Copy semantics
// Ensure INode cannot be copied, only moved
static_assert(!std::is_copy_constructible_v<INode>, "INode must not be copy constructible");
static_assert(!std::is_copy_assignable_v<INode>, "INode must not be copy assignable");

// Ensure Graph cannot be copied, only moved (inherits deleted copy from INode or explicitly deleted)
static_assert(!std::is_copy_constructible_v<Graph>, "Graph must not be copy constructible");
static_assert(!std::is_copy_assignable_v<Graph>, "Graph must not be copy assignable");

// Optional: Explicitly verify that move semantics ARE available
static_assert(std::is_move_constructible_v<Graph>, "Graph must be move constructible");
static_assert(std::is_move_assignable_v<Graph>, "Graph must be move assignable");

// Utility class to access private/protected members of Graph for testing purposes
class GraphTestUtils : public Graph
{
public:
    GraphTestUtils() = default;

    using Graph::build_operation_graph_via_descriptors;

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
        detail::IHipdnnBackend::setInstance(_mockBackend);
        _handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
    }
    void TearDown() override
    {
        detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    void expectGraphSerializedToBackendDescriptor(
        std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT>& deserializedGraph)
    {
        EXPECT_CALL(*_mockBackend,
                    backendCreateAndDeserializeGraphExt(::testing::_, ::testing::_, ::testing::_))
            .WillOnce([&deserializedGraph]([[maybe_unused]] hipdnnBackendDescriptor_t* descriptor,
                                           const uint8_t* serializedGraph,
                                           size_t graphByteSize) {
                deserializedGraph = hipdnn_data_sdk::data_objects::UnPackGraph(serializedGraph);
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
        mean->set_uid(2).set_name("Mean").set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1});

        auto invVariance = std::make_shared<TensorAttributes>();
        invVariance->set_uid(3)
            .set_name("InvVariance")
            .set_data_type(DataType::FLOAT)
            .set_dim({1, 2, 1, 1});

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(4).set_name("Scale").set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1});

        auto bias = std::make_shared<TensorAttributes>();
        bias->set_uid(5).set_name("Bias").set_data_type(DataType::FLOAT).set_dim({1, 2, 1, 1});

        BatchnormInferenceAttributes batchnormAttributes;
        batchnormAttributes.set_name("BatchnormNode");

        return graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);
    }
};

TEST_F(TestGraph, ValidateUnsetNodeComputeTypeUnsetGraphComputeType)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::NOT_SET)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    auto validationResult = graph.validate();

    EXPECT_FALSE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ValidateUnsetNodeComputeTypeSetGraphComputeType)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    auto validationResult = graph.validate();

    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ValidateUnsetTensorDataTypeUnsetGraphIoType)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::NOT_SET);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::NOT_SET);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    auto validationResult = graph.validate();

    EXPECT_FALSE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ValidateUnsetTensorDataTypeUnsetGraphIntermediateType)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::NOT_SET)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::NOT_SET);
    in0->set_is_virtual(true);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(in0, attributes);

    auto validationResult = graph.validate();

    EXPECT_FALSE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, ValidateUnsetTensorDataTypeSetGraphDataTypes)
{
    Graph graph;

    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::NOT_SET);
    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8})
        .set_data_type(DataType::NOT_SET)
        .set_is_virtual(true);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::ADD);

    auto out0 = graph.pointwise(in0, in1, attributes);

    auto validationResult = graph.validate();

    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

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
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    x->set_name("BatchnormNode::X");

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("BatchnormNode::SCALE").set_dim(derivedDims).set_stride(derivedStrides);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("BatchnormNode::BIAS").set_dim(derivedDims).set_stride(derivedStrides);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("BatchnormNode::EPSILON").set_value(0.001f);

    BatchnormAttributes attributes;
    attributes.set_name("BatchnormNode");
    attributes.set_epsilon(epsilon);

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
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto dy = std::make_shared<TensorAttributes>();
    auto x = std::make_shared<TensorAttributes>();
    auto scale = std::make_shared<TensorAttributes>();

    dy->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    x->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    scale->set_dim(derivedDims).set_stride(derivedStrides).set_data_type(DataType::FLOAT);

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
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1, 2, 1, 1});

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_dim({1, 2, 1, 1});

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 2, 1, 1});

    mean->set_dim(derivedDims).set_stride(derivedStrides);
    invVariance->set_dim(derivedDims).set_stride(derivedStrides);
    scale->set_dim(derivedDims).set_stride(derivedStrides);
    bias->set_dim(derivedDims).set_stride(derivedStrides);

    BatchnormInferenceAttributes attributes;
    attributes.set_name("BatchnormNode");

    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BatchnormInferenceNodeVarianceExtCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1, 2, 1, 1});

    auto variance = std::make_shared<TensorAttributes>();
    variance->set_dim({1, 2, 1, 1});

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 2, 1, 1});

    mean->set_dim(derivedDims).set_stride(derivedStrides);
    variance->set_dim(derivedDims).set_stride(derivedStrides);
    scale->set_dim(derivedDims).set_stride(derivedStrides);
    bias->set_dim(derivedDims).set_stride(derivedStrides);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("Epsilon").set_value(1e-5);

    BatchnormInferenceAttributesVarianceExt attributes;
    attributes.set_name("BatchnormNodeVariance");

    auto y = graph.batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, epsilon, attributes);

    EXPECT_EQ(y->get_name(), "BatchnormNodeVariance::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, PointwiseNodeCreationSingleInput)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

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
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

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
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

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
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

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
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

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
                           const hipdnn_data_sdk::data_objects::TensorAttributesT& serializedTensor)
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
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 6);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
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
              hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
    auto deserializedBatchnormAttributes
        = deserializedGraph->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    EXPECT_EQ(deserializedBatchnormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->inv_variance_tensor_uid, invVariance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->y_tensor_uid, y->get_uid());
}

TEST_F(TestGraph, BuildAndSerializeBatchnormInferenceVarianceExtGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto variance = std::make_shared<TensorAttributes>();
    variance->set_uid(3)
        .set_name("Variance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(6).set_name("Epsilon").set_value(1e-5);

    BatchnormInferenceAttributesVarianceExt batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNodeVariance");

    auto y = graph.batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, epsilon, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 7);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*mean, tensorLookup[mean->get_uid()]);
    validateTensor(*variance, tensorLookup[variance->get_uid()]);
    validateTensor(*scale, tensorLookup[scale->get_uid()]);
    validateTensor(*bias, tensorLookup[bias->get_uid()]);
    validateTensor(*epsilon, tensorLookup[epsilon->get_uid()]);
    validateTensor(*y, tensorLookup[y->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "BatchnormNodeVariance");
    EXPECT_EQ(
        deserializedGraph->nodes[0]->attributes.type,
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt);
    auto deserializedBatchnormAttributes
        = deserializedGraph->nodes[0]->attributes.AsBatchnormInferenceAttributesVarianceExt();
    EXPECT_EQ(deserializedBatchnormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->mean_tensor_uid, mean->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->variance_tensor_uid, variance->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserializedBatchnormAttributes->epsilon_tensor_uid, epsilon->get_uid());
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

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto prevRunningMean = std::make_shared<TensorAttributes>();
    prevRunningMean->set_uid(4)
        .set_name("PrevRunningMean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto prevRunningVariance = std::make_shared<TensorAttributes>();
    prevRunningVariance->set_uid(5)
        .set_name("PrevRunningVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto momentum = std::make_shared<TensorAttributes>();
    momentum->set_uid(6)
        .set_name("Momentum")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(7).set_name("Epsilon").set_value(1.f);

    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");
    batchnormAttributes.set_previous_running_stats(prevRunningMean, prevRunningVariance, momentum);
    batchnormAttributes.set_epsilon(epsilon);

    auto [y, mean, invVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedBatchnormGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 12);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
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
              hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes);
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

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto prevRunningMean = std::make_shared<TensorAttributes>();
    prevRunningMean->set_uid(4)
        .set_name("PrevRunningMean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto prevRunningVariance = std::make_shared<TensorAttributes>();
    prevRunningVariance->set_uid(5)
        .set_name("PrevRunningVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto momentum = std::make_shared<TensorAttributes>();
    momentum->set_uid(6)
        .set_name("Momentum")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(7).set_name("Epsilon").set_value(1.f);

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

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedBatchnormAndPointwiseGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 13);
    EXPECT_EQ(deserializedGraph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
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
              hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes);
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
              hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    auto deserializedPointwiseAttributes
        = deserializedGraph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserializedPointwiseAttributes->in_0_tensor_uid, y->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->out_0_tensor_uid, out0->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->operation,
              hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD);
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

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 2);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*in0, tensorLookup[in0->get_uid()]);
    validateTensor(*out0, tensorLookup[out0->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "PointwiseNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    auto deserializedPointwiseAttributes
        = deserializedGraph->nodes[0]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserializedPointwiseAttributes->in_0_tensor_uid, in0->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->out_0_tensor_uid, out0->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->operation,
              hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD);
}

TEST_F(TestGraph, BuildAndSerializePointwiseAndBatchnormInferenceGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);

    PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_name("PointwiseNode");
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph.pointwise(y, pointwiseAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 7);
    EXPECT_EQ(deserializedGraph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
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
              hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes);
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
              hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    auto deserializedPointwiseAttributes
        = deserializedGraph->nodes[1]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserializedPointwiseAttributes->in_0_tensor_uid, y->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->out_0_tensor_uid, out0->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->operation,
              hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD);
}

TEST_F(TestGraph, BuildAndSerializeBatchnormBackwardGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedGraphTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(1).set_name("Dy").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(2).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(3)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(4)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(5)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormBackwardNode");
    batchnormAttributes.set_saved_mean_and_inv_variance(mean, invVariance);

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 8);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
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
              hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes);
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

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedConvolutionGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 3);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*w, tensorLookup[w->get_uid()]);
    validateTensor(*y, tensorLookup[y->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "ConvolutionFpropNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes);
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

TEST_F(TestGraph, MatmulNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5}).set_stride({5, 1}).set_data_type(DataType::FLOAT);

    MatmulAttributes attributes;
    attributes.set_name("MatmulNode");

    auto c = graph.matmul(a, b, attributes);

    EXPECT_EQ(c->get_name(), "MatmulNode::C");
    EXPECT_TRUE(c->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BuildAndSerializeMatmulGraph)
{
    Graph graph;

    graph.set_name("SerializedMatmulGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto a = std::make_shared<TensorAttributes>();
    a->set_uid(1).set_name("A").set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto b = std::make_shared<TensorAttributes>();
    b->set_uid(2).set_name("B").set_dim({8, 5}).set_stride({5, 1}).set_data_type(DataType::FLOAT);

    MatmulAttributes attributes;
    attributes.set_name("MatmulNode");

    auto c = graph.matmul(a, b, attributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedMatmulGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 3);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*a, tensorLookup[a->get_uid()]);
    validateTensor(*b, tensorLookup[b->get_uid()]);
    validateTensor(*c, tensorLookup[c->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "MatmulNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::MatmulAttributes);
    auto deserializedMatmulAttributes
        = deserializedGraph->nodes[0]->attributes.AsMatmulAttributes();
    ASSERT_NE(deserializedMatmulAttributes, nullptr);
    EXPECT_EQ(deserializedMatmulAttributes->a_tensor_uid, a->get_uid());
    EXPECT_EQ(deserializedMatmulAttributes->b_tensor_uid, b->get_uid());
    EXPECT_EQ(deserializedMatmulAttributes->c_tensor_uid, c->get_uid());
}

TEST_F(TestGraph, LayernormNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 6, 4}).set_stride({24, 4, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({6, 4}).set_stride({4, 1}).set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({6, 4}).set_stride({4, 1}).set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("Epsilon").set_value(1e-5f);

    LayernormAttributes attributes;
    attributes.set_name("LayerNormNode");
    attributes.set_forward_phase(NormFwdPhase::INFERENCE);
    attributes.set_epsilon(epsilon);

    auto [y, mean, invVariance] = graph.layernorm(x, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "LayerNormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());
    // In inference mode, mean and inv_variance are not created
    EXPECT_EQ(mean, nullptr);
    EXPECT_EQ(invVariance, nullptr);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, RMSNormNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 1, 1}).set_stride({64, 1, 1, 1}).set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("RMSNormNode");
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto [y, invRms] = graph.rmsnorm(x, scale, rmsnormAttrs);

    EXPECT_EQ(y->get_name(), "RMSNormNode::Y");
    EXPECT_TRUE(y->get_is_virtual());
    EXPECT_EQ(invRms->get_name(), "RMSNormNode::INV_RMS");
    EXPECT_TRUE(invRms->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, RMSNormNodeCreationWithBias)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 64, 1, 1}).set_stride({64, 1, 1, 1}).set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 64, 1, 1}).set_stride({64, 1, 1, 1}).set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("RMSNormWithBias");
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_bias(bias);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto [y, invRms] = graph.rmsnorm(x, scale, rmsnormAttrs);

    EXPECT_EQ(y->get_name(), "RMSNormWithBias::Y");
    EXPECT_TRUE(y->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, LayernormNodeCreationTrainingPhase)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 6, 4}).set_stride({24, 4, 1}).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({6, 4}).set_stride({4, 1}).set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({6, 4}).set_stride({4, 1}).set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("Epsilon").set_value(1e-5f);

    LayernormAttributes attributes;
    attributes.set_name("LayernormNodeTraining");
    attributes.set_forward_phase(NormFwdPhase::TRAINING);
    attributes.set_epsilon(epsilon);

    auto [y, mean, invVariance] = graph.layernorm(x, scale, bias, attributes);

    EXPECT_EQ(y->get_name(), "LayernormNodeTraining::Y");
    EXPECT_TRUE(y->get_is_virtual());
    // In training mode, mean and inv_variance should be created
    ASSERT_NE(mean, nullptr);
    EXPECT_EQ(mean->get_name(), "LayernormNodeTraining::MEAN");
    ASSERT_NE(invVariance, nullptr);
    EXPECT_EQ(invVariance->get_name(), "LayernormNodeTraining::INV_VARIANCE");

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BuildAndSerializeLayernormGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("SerializedLayernormGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({2, 6, 4})
        .set_stride({24, 4, 1})
        .set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim({6, 4})
        .set_stride({4, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim({6, 4})
        .set_stride({4, 1});

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(4).set_name("Epsilon").set_value(1e-5f);

    LayernormAttributes layernormAttributes;
    layernormAttributes.set_name("LayerNormNode");
    layernormAttributes.set_forward_phase(NormFwdPhase::INFERENCE);
    layernormAttributes.set_epsilon(epsilon);

    auto [y, mean, invVariance] = graph.layernorm(x, scale, bias, layernormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedLayernormGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 5u);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1u);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*x, tensorLookup[x->get_uid()]);
    validateTensor(*scale, tensorLookup[scale->get_uid()]);
    validateTensor(*bias, tensorLookup[bias->get_uid()]);
    validateTensor(*epsilon, tensorLookup[epsilon->get_uid()]);
    validateTensor(*y, tensorLookup[y->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "LayerNormNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::LayernormAttributes);
    auto deserializedLayernormAttributes
        = deserializedGraph->nodes[0]->attributes.AsLayernormAttributes();
    ASSERT_NE(deserializedLayernormAttributes, nullptr);
    EXPECT_EQ(deserializedLayernormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedLayernormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedLayernormAttributes->bias_tensor_uid, bias->get_uid());
    EXPECT_EQ(deserializedLayernormAttributes->epsilon_tensor_uid, epsilon->get_uid());
    EXPECT_EQ(deserializedLayernormAttributes->y_tensor_uid, y->get_uid());
}

TEST_F(TestGraph, BuildAndSerializeRMSNormGraph)
{
    Graph graph;

    graph.set_name("SerializedRMSNormGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1})
        .set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2)
        .set_name("Scale")
        .set_dim({1, 64, 1, 1})
        .set_stride({64, 1, 1, 1})
        .set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(3).set_name("Epsilon");

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("RMSNormNode");
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto [y, invRms] = graph.rmsnorm(x, scale, rmsnormAttrs);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedRMSNormGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 5); // x, scale, epsilon, y, inv_rms
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "RMSNormNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormAttributes);
    auto deserializedRMSNormAttributes
        = deserializedGraph->nodes[0]->attributes.AsRMSNormAttributes();
    ASSERT_NE(deserializedRMSNormAttributes, nullptr);
    EXPECT_EQ(deserializedRMSNormAttributes->x_tensor_uid, x->get_uid());
    EXPECT_EQ(deserializedRMSNormAttributes->scale_tensor_uid, scale->get_uid());
    EXPECT_EQ(deserializedRMSNormAttributes->epsilon_tensor_uid, epsilon->get_uid());
    EXPECT_EQ(deserializedRMSNormAttributes->y_tensor_uid, y->get_uid());
    EXPECT_EQ(deserializedRMSNormAttributes->forward_phase,
              hipdnn_data_sdk::data_objects::NormFwdPhase::TRAINING);
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

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedConvolutionDgradGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 3);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*dy, tensorLookup[dy->get_uid()]);
    validateTensor(*w, tensorLookup[w->get_uid()]);
    validateTensor(*dx, tensorLookup[dx->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "ConvolutionDgradNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes);
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

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto xPointwise = std::make_shared<TensorAttributes>();
    xPointwise->set_uid(6)
        .set_name("X_Pointwise")
        .set_dim(dims)
        .set_stride(strides)
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pointwiseAttributes;
    pointwiseAttributes.set_name("PointwiseNode");
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto dy = graph.pointwise(xPointwise, pointwiseAttributes);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(3)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(4)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormBackwardNode");
    batchnormAttributes.set_saved_mean_and_inv_variance(mean, invVariance);

    auto [dx, dscale, dbias] = graph.batchnorm_backward(dy, x, scale, batchnormAttributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedGraphTest");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->tensors.size(), 9);
    EXPECT_EQ(deserializedGraph->nodes.size(), 2);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
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
              hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes);
    auto deserializedPointwiseAttributes
        = deserializedGraph->nodes[0]->attributes.AsPointwiseAttributes();
    EXPECT_EQ(deserializedPointwiseAttributes->in_0_tensor_uid, xPointwise->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->out_0_tensor_uid, dy->get_uid());
    EXPECT_EQ(deserializedPointwiseAttributes->operation,
              hipdnn_data_sdk::data_objects::PointwiseMode::RELU_FWD);

    EXPECT_EQ(deserializedGraph->nodes[1]->name, "BatchnormBackwardNode");
    EXPECT_EQ(deserializedGraph->nodes[1]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes);
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

TEST_F(TestGraph, BuildOperationGraphViaDescriptorsFailsWhenNodeFails)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    graph.set_name("FailTest").set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_TENSOR_X_DIMS))
        .set_stride(toVec(K_TENSOR_X_STRIDES));
    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_TENSOR_W_UID)
        .set_name("W")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_TENSOR_W_DIMS))
        .set_stride(toVec(K_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_x(x);
    convAttrs.set_w(w);
    convAttrs.set_pre_padding(toVec(K_CONV_PADDING));
    convAttrs.set_post_padding(toVec(K_CONV_PADDING));
    convAttrs.set_stride(toVec(K_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_CONV_DILATION));

    graph.conv_fprop(x, w, convAttrs);

    // All descriptor creation fails
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(_, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    auto result = graph.build_operation_graph_via_descriptors(_handle);
    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestGraph, BuildOperationGraphViaDescriptorsFailsWhenGraphCreateFails)
{
    ::testing::FLAGS_gmock_verbose = "error";
    GraphTestUtils graph;
    graph.set_name("GraphCreateFail")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_TENSOR_X_DIMS))
        .set_stride(toVec(K_TENSOR_X_STRIDES));
    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_TENSOR_W_UID)
        .set_name("W")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_TENSOR_W_DIMS))
        .set_stride(toVec(K_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_x(x);
    convAttrs.set_w(w);
    convAttrs.set_compute_data_type(DataType::FLOAT);
    convAttrs.set_pre_padding(toVec(K_CONV_PADDING));
    convAttrs.set_post_padding(toVec(K_CONV_PADDING));
    convAttrs.set_stride(toVec(K_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_CONV_DILATION));

    graph.conv_fprop(x, w, convAttrs);

    // Validate graph to propagate graph-level attributes (e.g. io_data_type)
    // to node/tensor attributes, matching the real build() flow
    ASSERT_TRUE(graph.validate().is_good());

    // Tensor/operation descriptors succeed, but graph descriptor creation fails
    EXPECT_CALL(*_mockBackend,
                backendCreateDescriptor(Ne(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR), _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    auto result = graph.build_operation_graph_via_descriptors(_handle);
    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestGraph, BuildOperationGraphViaDescriptorsFailsOnEmptyGraph)
{
    GraphTestUtils graph;
    graph.set_name("EmptyGraph").set_compute_data_type(DataType::FLOAT);

    auto result = graph.build_operation_graph_via_descriptors(_handle);
    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE);
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
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
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

TEST_F(TestGraph, PreferredEngineIdSelectsSpecificConfig)
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

    // Set preferred engine ID
    int64_t preferredEngineId = 42;
    graph.set_preferred_engine_id_ext(preferredEngineId);

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

    // First call: elementCount query - return 2 configs available
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
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineConfigDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2346);
    auto engineDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    auto engineDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3346);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc1](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&engineConfigDesc2](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc1));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc1,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc1](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc1;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc1, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc2));

    // Get engine from second config (ID = 42 - our preferred one)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc2,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc2](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from second engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc2, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([preferredEngineId](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      int64_t,
                                      int64_t*,
                                      void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = preferredEngineId;
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

TEST_F(TestGraph, PreferredEngineIdFallsBackToTopConfig)
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

    // Set preferred engine ID that doesn't exist
    int64_t preferredEngineId = 999;
    graph.set_preferred_engine_id_ext(preferredEngineId);

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

    // First call: elementCount query - return 2 configs available
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
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineConfigDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2346);
    auto engineDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    auto engineDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3346);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc1](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&engineConfigDesc2](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Second call: actual data retrieval
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc1));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc1,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc1](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc1;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine (neither will match preferred ID 999)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc1, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc2));

    // Get engine from second config (ID = 42)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc2,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc2](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from second engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc2, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 42;
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

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
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

    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    graph.create_execution_plans(heurModes);

    auto result = graph.check_support();
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(result.get_message(), "");
}

TEST_F(TestGraph, ExecutionPlanisFinalizedAfterBuildPlans)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
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

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
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

    EXPECT_CALL(*_mockBackend, backendFinalize(executionPlanDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
    auto result = graph.create_execution_plans(heurModes);
    EXPECT_TRUE(result.is_good());

    result = graph.build_plans();
    EXPECT_TRUE(result.is_good());
    EXPECT_EQ(result.get_message(), "");
}

TEST_F(TestGraph, WorkspaceSizeIsRetrievedFromExecutionPlan)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
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

    auto heurDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce(
            [&heurDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = heurDesc;
                return HIPDNN_STATUS_SUCCESS;
            });
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc));

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc));

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
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

    const std::vector<HeuristicMode> heurModes = {HeuristicMode::FALLBACK};
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
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
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

    // Get engine from first config (ID = 10)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineCfgDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Get ID from first engine
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
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
    graph.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    // Node 0: batchnorm1
    BatchnormInferenceAttributes bnAttrs1;
    bnAttrs1.set_name("batchnorm1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, bnAttrs1);

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(x->get_dim());
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

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
    graph.set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);
    x->set_uid(1);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    BatchnormInferenceAttributes attributes1;
    attributes1.set_name("BatchnormNode1");
    auto y1 = graph.batchnorm_inference(x, mean, invVariance, scale, bias, attributes1);

    BatchnormInferenceAttributes attributes2;
    attributes2.set_name("BatchnormNode2");
    auto y2 = graph.batchnorm_inference(y1, mean, invVariance, scale, bias, attributes2);

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(x->get_dim());
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

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

TEST_F(TestGraph, GetTensorsByUidReturnsMap)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(42)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({24, 12, 4, 1})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("PointwiseNode");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);
    y->set_uid(99);

    // Get tensor map by UID
    auto tensorsByUid = graph.getTensorsByUid();

    // Map contains expected tensors
    EXPECT_EQ(tensorsByUid.size(), 2);

    auto itX = tensorsByUid.find(42);
    ASSERT_NE(itX, tensorsByUid.end());
    EXPECT_EQ(itX->second->get_uid(), 42);
    EXPECT_EQ(itX->second->get_name(), "X");
    EXPECT_EQ(itX->second, x);

    auto itY = tensorsByUid.find(99);
    ASSERT_NE(itY, tensorsByUid.end());
    EXPECT_EQ(itY->second->get_uid(), 99);
    EXPECT_EQ(itY->second, y);

    // Non-existent UID not found in map
    auto notFound = tensorsByUid.find(999);
    EXPECT_EQ(notFound, tensorsByUid.end());
}

TEST_F(TestGraph, GetTensorsByNameReturnsMap)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("InputTensor")
        .set_dim({1, 2, 3, 4})
        .set_stride({24, 12, 4, 1})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("PointwiseNode");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);

    // Get tensor map by name
    auto tensorsByName = graph.getTensorsByName();

    // Map contains expected tensors
    EXPECT_EQ(tensorsByName.size(), 2);

    auto itX = tensorsByName.find("InputTensor");
    ASSERT_NE(itX, tensorsByName.end());
    EXPECT_EQ(itX->second->get_name(), "InputTensor");
    EXPECT_EQ(itX->second, x);

    // Output tensor has auto-generated name
    auto itY = tensorsByName.find("PointwiseNode::OUT_0");
    ASSERT_NE(itY, tensorsByName.end());
    EXPECT_EQ(itY->second, y);

    // Non-existent name not found in map
    auto notFound = tensorsByName.find("NonExistentTensor");
    EXPECT_EQ(notFound, tensorsByName.end());
}

TEST_F(TestGraph, GetTensorsByUidAndNameIncludePeerStatTensors)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("InputX")
        .set_dim({1, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1})
        .set_data_type(DataType::FLOAT);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2)
        .set_name("ScaleTensor")
        .set_dim({64})
        .set_stride({1})
        .set_data_type(DataType::FLOAT);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3)
        .set_name("BiasTensor")
        .set_dim({64})
        .set_stride({1})
        .set_data_type(DataType::FLOAT);

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(4);

    // Create peer stat tensors (these go in the separate peer_stats vector, not outputs)
    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(20).set_name("PeerStat1");

    auto peerStat2 = std::make_shared<TensorAttributes>();
    peerStat2->set_uid(21).set_name("PeerStat2");

    BatchnormAttributes bnAttrs;
    bnAttrs.set_name("BN");
    bnAttrs.set_epsilon(epsilon);
    bnAttrs.set_peer_stats({peerStat1, peerStat2}); // Set peer stats separately

    auto [y, savedMean, savedInvVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, bnAttrs);

    // Assign UIDs to output tensors
    y->set_uid(10);
    savedMean->set_uid(11);
    savedInvVariance->set_uid(12);

    // Test getTensorsByUid()
    auto tensorsByUid = graph.getTensorsByUid();

    // Should have:
    // Inputs: x(1), scale(2), bias(3), epsilon(4) = 4
    // Outputs: y(10), savedMean(11), savedInvVariance(12) = 3
    // Peer stats: peerStat1(20), peerStat2(21) = 2
    // Total = 9
    EXPECT_EQ(tensorsByUid.size(), 9);

    // Verify input tensors are present by UID
    EXPECT_NE(tensorsByUid.find(1), tensorsByUid.end()); // x
    EXPECT_NE(tensorsByUid.find(2), tensorsByUid.end()); // scale
    EXPECT_NE(tensorsByUid.find(3), tensorsByUid.end()); // bias
    EXPECT_NE(tensorsByUid.find(4), tensorsByUid.end()); // epsilon

    // Verify output tensors are present by UID
    EXPECT_NE(tensorsByUid.find(10), tensorsByUid.end()); // y
    EXPECT_NE(tensorsByUid.find(11), tensorsByUid.end()); // savedMean
    EXPECT_NE(tensorsByUid.find(12), tensorsByUid.end()); // savedInvVariance

    // Verify peer stat tensors are present by UID (tests the specialized gather override)
    EXPECT_NE(tensorsByUid.find(20), tensorsByUid.end()); // peerStat1
    EXPECT_NE(tensorsByUid.find(21), tensorsByUid.end()); // peerStat2

    // Verify the pointers match
    EXPECT_EQ(tensorsByUid[1], x);
    EXPECT_EQ(tensorsByUid[10], y);
    EXPECT_EQ(tensorsByUid[20], peerStat1);
    EXPECT_EQ(tensorsByUid[21], peerStat2);

    // Test getTensorsByName()
    auto tensorsByName = graph.getTensorsByName();

    // Should have all named tensors:
    // Inputs with names: InputX, ScaleTensor, BiasTensor (3) - epsilon has no name
    // Outputs with names: BN::Y, BN::MEAN, BN::INV_VARIANCE (3)
    // Peer stats with names: PeerStat1, PeerStat2 (2)
    // Total = 8
    EXPECT_EQ(tensorsByName.size(), 8);

    // Verify input tensors are present by name
    EXPECT_NE(tensorsByName.find("InputX"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("ScaleTensor"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("BiasTensor"), tensorsByName.end());

    // Verify output tensors are present by name
    EXPECT_NE(tensorsByName.find("BN::Y"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("BN::MEAN"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("BN::INV_VARIANCE"), tensorsByName.end());

    // Verify peer stat tensors are present by name (tests the specialized gather override)
    EXPECT_NE(tensorsByName.find("PeerStat1"), tensorsByName.end());
    EXPECT_NE(tensorsByName.find("PeerStat2"), tensorsByName.end());

    // Verify the pointers match
    EXPECT_EQ(tensorsByName["InputX"], x);
    EXPECT_EQ(tensorsByName["BN::Y"], y);
    EXPECT_EQ(tensorsByName["PeerStat1"], peerStat1);
    EXPECT_EQ(tensorsByName["PeerStat2"], peerStat2);
}

TEST_F(TestGraph, GetRankedEngineIdsReturnsRankedList)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("RankedEngineIdsTestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({1, 2, 3, 4})
        .set_stride({24, 12, 4, 1})
        .set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2).set_name("Mean").set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3).set_name("InvVariance").set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4).set_name("Scale").set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5).set_name("Bias").set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    graph.build_operation_graph(_handle);

    // Mock heuristic descriptor creation
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
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting engine count (3 engines available)
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
            *elementCount = 3;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Create 3 engine config descriptors
    auto engineConfigDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2001);
    auto engineConfigDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2002);
    auto engineConfigDesc3 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2003);
    auto engineDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3001);
    auto engineDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3002);
    auto engineDesc3 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3003);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc1](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&engineConfigDesc2](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc2;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&engineConfigDesc3](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc3;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock getting engine configs
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(heurDesc,
                                    HIPDNN_ATTR_ENGINEHEUR_RESULTS,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    3,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* retrievedCount,
                     void*) {
            *retrievedCount = 3;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Finalize and get IDs for each engine config
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc1,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc1](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc1;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc1, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 100;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc2,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc2](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc2, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 200;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc3))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc3,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc3](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 int64_t,
                                 int64_t*,
                                 void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc3;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc3, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 300;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<int64_t> rankedEngineIds;
    auto result = graph.get_ranked_engine_ids(rankedEngineIds);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(rankedEngineIds.size(), 3);
    EXPECT_EQ(rankedEngineIds[0], 100);
    EXPECT_EQ(rankedEngineIds[1], 200);
    EXPECT_EQ(rankedEngineIds[2], 300);
}

TEST_F(TestGraph, BuildMethodSucceedsWithValidGraph)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;

    graph.set_name("BuildMethodTestGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    std::vector<int64_t> dims = {1, 2, 3, 4};
    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(2)
        .set_name("Mean")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(3)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(4)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(5)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim(derivedDims)
        .set_stride(derivedStrides);

    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_name("BatchnormNode");

    graph.batchnorm_inference(x, mean, invVariance, scale, bias, batchnormAttributes);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock create_execution_plans
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
                     int64_t* elementCount,
                     void*) {
            *elementCount = 1;
            return HIPDNN_STATUS_SUCCESS;
        });

    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_ENGINECFG_ENGINE,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    nullptr,
                                    _))
        .WillOnce([&engineDesc](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t*,
                                void* arrayOfElements) {
            *static_cast<hipdnnBackendDescriptor_t*>(arrayOfElements) = engineDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, _))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t*,
                     void* arrayOfElements) {
            *static_cast<int64_t*>(arrayOfElements) = 10;
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

    // Mock build_plans
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(executionPlanDesc,
                                    HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(executionPlanDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto result = graph.build(_handle);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanExtWithKnobSettings)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation for get_knob_lookup_for_engine
    // and during initializeEngineConfig(engineId)
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(2)
        .WillRepeatedly(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Create flatbuffer knob for "global.deterministic"
    flatbuffers::FlatBufferBuilder builder;
    auto knobOffset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder,
        "global.deterministic",
        "Enable deterministic execution",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder, static_cast<int64_t>(0)).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
        hipdnn_data_sdk::data_objects::CreateIntConstraint(
            builder, static_cast<int64_t>(0), static_cast<int64_t>(1), static_cast<int64_t>(1))
            .Union(),
        false);
    builder.Finish(knobOffset);
    auto knobBuffer = builder.Release();

    // Mock getting knob count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
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

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([&knobBuffer](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t* actualCount,
                                void* arrayOfElements) {
            *actualCount = 1;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knobBuffer.data();
            fbDataArray[0].size = knobBuffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock setting knob settings on engine config
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Create knob setting for "global.deterministic" = 1
    std::vector<KnobSetting> settings;
    settings.emplace_back("global.deterministic", static_cast<int64_t>(1));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanWithInt64Knobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(2)
        .WillRepeatedly(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Create flatbuffer knob
    flatbuffers::FlatBufferBuilder builder;
    auto knobOffset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder,
        "global.deterministic",
        "Enable deterministic execution",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder, static_cast<int64_t>(0)).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
        hipdnn_data_sdk::data_objects::CreateIntConstraint(
            builder, static_cast<int64_t>(0), static_cast<int64_t>(1), static_cast<int64_t>(1))
            .Union(),
        false);
    builder.Finish(knobOffset);
    auto knobBuffer = builder.Release();

    // Mock getting knob count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
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

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([&knobBuffer](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t* actualCount,
                                void* arrayOfElements) {
            *actualCount = 1;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knobBuffer.data();
            fbDataArray[0].size = knobBuffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock setting knob settings on engine config
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Use the new API with vector of KnobSetting
    std::vector<KnobSetting> settings;
    settings.emplace_back("global.deterministic", static_cast<int64_t>(1));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanExtWithMultipleKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(2)
        .WillRepeatedly(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Create multiple flatbuffer knobs
    flatbuffers::FlatBufferBuilder builder1;
    auto knob1Offset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder1,
        "global.deterministic",
        "Enable deterministic execution",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder1, static_cast<int64_t>(0)).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
        hipdnn_data_sdk::data_objects::CreateIntConstraint(
            builder1, static_cast<int64_t>(0), static_cast<int64_t>(1), static_cast<int64_t>(1))
            .Union(),
        false);
    builder1.Finish(knob1Offset);
    auto knob1Buffer = builder1.Release();

    flatbuffers::FlatBufferBuilder builder2;
    auto knob2Offset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder2,
        "performance.threads",
        "Number of threads",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder2, static_cast<int64_t>(4)).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
        hipdnn_data_sdk::data_objects::CreateIntConstraint(
            builder2, static_cast<int64_t>(1), static_cast<int64_t>(16), static_cast<int64_t>(1))
            .Union(),
        false);
    builder2.Finish(knob2Offset);
    auto knob2Buffer = builder2.Release();

    // Mock getting knob count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([&knob1Buffer, &knob2Buffer](hipdnnBackendDescriptor_t,
                                               hipdnnBackendAttributeName_t,
                                               hipdnnBackendAttributeType_t,
                                               int64_t,
                                               int64_t* actualCount,
                                               void* arrayOfElements) {
            *actualCount = 2;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knob1Buffer.data();
            fbDataArray[0].size = knob1Buffer.size();
            fbDataArray[1].ptr = knob2Buffer.data();
            fbDataArray[1].size = knob2Buffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock setting knob settings on engine config (2 knobs)
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    2,
                                    NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Create multiple knob settings
    std::vector<KnobSetting> settings;
    settings.emplace_back("global.deterministic", static_cast<int64_t>(1));
    settings.emplace_back("performance.threads", static_cast<int64_t>(8));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanExtFailsWithoutGraphBuilt)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    int64_t engineId = 42;
    std::vector<KnobSetting> settings;
    settings.emplace_back("global.deterministic", static_cast<int64_t>(1));

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.get_message(),
              "Graph has not been built, build the operation graph first. Cannot create "
              "execution plan.");
}

TEST_F(TestGraph, CreateExecutionPlanExtWithEmptySettings)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(2)
        .WillRepeatedly(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting knob count - return 0 (no knobs available)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Empty settings - should not call backendSetAttribute for knobs
    std::vector<KnobSetting> settings;
    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, CreateExecutionPlanExtIgnoresUnsupportedKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(2)
        .WillRepeatedly(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Create flatbuffer knob - only "global.deterministic" is supported
    flatbuffers::FlatBufferBuilder builder;
    auto knobOffset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder,
        "global.deterministic",
        "Enable deterministic execution",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder, static_cast<int64_t>(0)).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
        hipdnn_data_sdk::data_objects::CreateIntConstraint(
            builder, static_cast<int64_t>(0), static_cast<int64_t>(1), static_cast<int64_t>(1))
            .Union(),
        false);
    builder.Finish(knobOffset);
    auto knobBuffer = builder.Release();

    // Mock getting knob count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
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

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([&knobBuffer](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t* actualCount,
                                void* arrayOfElements) {
            *actualCount = 1;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knobBuffer.data();
            fbDataArray[0].size = knobBuffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock setting knob settings - only 1 valid knob should be set
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Create settings with one supported and one unsupported knob
    std::vector<KnobSetting> settings;
    settings.emplace_back("global.deterministic", static_cast<int64_t>(1)); // Supported
    settings.emplace_back("unsupported.knob", static_cast<int64_t>(999)); // Not supported

    auto result = graph.create_execution_plan_ext(engineId, settings);
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

TEST_F(TestGraph, BuildMethodFailsWhenValidationFails)
{
    Graph graph;

    // Create an invalid graph (compute type not set)
    graph.set_name("TestGraph")
        .set_compute_data_type(DataType::NOT_SET)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_dim({1, 2, 3, 4}).set_stride({5, 6, 7, 8}).set_data_type(DataType::FLOAT);

    PointwiseAttributes attributes;
    attributes.set_name("PointwiseNode");
    attributes.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(in0, attributes);

    auto result = graph.build(_handle);
    EXPECT_FALSE(result.is_good());
}

// ============================================================================
// get_knobs_for_engine Tests
// ============================================================================

TEST_F(TestGraph, GetKnobsForEngineReturnsEmptyVectorWhenNoKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillOnce(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce([engineId](hipdnnBackendDescriptor_t,
                             hipdnnBackendAttributeName_t,
                             hipdnnBackendAttributeType_t,
                             int64_t,
                             const void* arrayOfElements) {
            EXPECT_EQ(*static_cast<const int64_t*>(arrayOfElements), engineId);
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting knob count - return 0 (no knobs)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_TRUE(knobs.empty());
}

TEST_F(TestGraph, GetKnobsForEngineReturnsKnobsWhenAvailable)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillOnce(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Create flatbuffer knobs for testing
    flatbuffers::FlatBufferBuilder builder1;
    auto knob1Offset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder1,
        "test_knob_1",
        "First test knob",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder1, 10).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
        hipdnn_data_sdk::data_objects::CreateIntConstraint(builder1, 0, 100, 1).Union(),
        false);
    builder1.Finish(knob1Offset);
    auto knob1Buffer = builder1.Release();

    flatbuffers::FlatBufferBuilder builder2;
    auto knob2Offset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder2,
        "test_knob_2",
        "Second test knob",
        hipdnn_data_sdk::data_objects::KnobValue::FloatValue,
        hipdnn_data_sdk::data_objects::CreateFloatValue(builder2, 0.5).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::FloatConstraint,
        hipdnn_data_sdk::data_objects::CreateFloatConstraint(builder2, 0.0, 1.0).Union(),
        false);
    builder2.Finish(knob2Offset);
    auto knob2Buffer = builder2.Release();

    // Mock getting knob count - return 2
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([&knob1Buffer, &knob2Buffer](hipdnnBackendDescriptor_t,
                                               hipdnnBackendAttributeName_t,
                                               hipdnnBackendAttributeType_t,
                                               int64_t,
                                               int64_t* actualCount,
                                               void* arrayOfElements) {
            *actualCount = 2;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knob1Buffer.data();
            fbDataArray[0].size = knob1Buffer.size();
            fbDataArray[1].ptr = knob2Buffer.data();
            fbDataArray[1].size = knob2Buffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(engineId, knobs);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(knobs.size(), 2);

    // Verify first knob
    EXPECT_EQ(knobs[0].knobId(), "test_knob_1");
    EXPECT_EQ(knobs[0].description(), "First test knob");
    EXPECT_EQ(knobs[0].valueType(), KnobValueType::INT64);
    EXPECT_FALSE(knobs[0].isDeprecated());
    auto defaultValue1 = std::get_if<int64_t>(&knobs[0].defaultValue());
    ASSERT_NE(defaultValue1, nullptr);
    EXPECT_EQ(*defaultValue1, 10);

    // Verify second knob
    EXPECT_EQ(knobs[1].knobId(), "test_knob_2");
    EXPECT_EQ(knobs[1].description(), "Second test knob");
    EXPECT_EQ(knobs[1].valueType(), KnobValueType::FLOAT64);
    EXPECT_FALSE(knobs[1].isDeprecated());
    auto defaultValue2 = std::get_if<double>(&knobs[1].defaultValue());
    ASSERT_NE(defaultValue2, nullptr);
    EXPECT_DOUBLE_EQ(*defaultValue2, 0.5);
}

TEST_F(TestGraph, GetKnobsForEngineHandlesDeprecatedKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillOnce(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Create a deprecated knob
    flatbuffers::FlatBufferBuilder builder;
    auto knobOffset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder,
        "deprecated_knob",
        "This knob is deprecated",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder, 5).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::NONE,
        0,
        true); // deprecated = true
    builder.Finish(knobOffset);
    auto knobBuffer = builder.Release();

    // Mock getting knob count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
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

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([&knobBuffer](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t* actualCount,
                                void* arrayOfElements) {
            *actualCount = 1;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knobBuffer.data();
            fbDataArray[0].size = knobBuffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(42, knobs);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(knobs.size(), 1);
    EXPECT_TRUE(knobs[0].isDeprecated());
    EXPECT_EQ(knobs[0].knobId(), "deprecated_knob");
}

TEST_F(TestGraph, GetKnobsForEngineHandlesStringKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillOnce(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Create a string knob with valid values constraint
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<flatbuffers::String>> validValues;
    validValues.push_back(builder.CreateString("option1"));
    validValues.push_back(builder.CreateString("option2"));
    validValues.push_back(builder.CreateString("option3"));

    auto constraintOffset = hipdnn_data_sdk::data_objects::CreateStringConstraint(
        builder, 100, builder.CreateVector(validValues));

    auto knobOffset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder,
        "string_knob",
        "String knob with choices",
        hipdnn_data_sdk::data_objects::KnobValue::StringValue,
        hipdnn_data_sdk::data_objects::CreateStringValueDirect(builder, "option1").Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint,
        constraintOffset.Union(),
        false);
    builder.Finish(knobOffset);
    auto knobBuffer = builder.Release();

    // Mock getting knob count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
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

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([&knobBuffer](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t* actualCount,
                                void* arrayOfElements) {
            *actualCount = 1;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knobBuffer.data();
            fbDataArray[0].size = knobBuffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(42, knobs);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(knobs.size(), 1);
    EXPECT_EQ(knobs[0].knobId(), "string_knob");
    EXPECT_EQ(knobs[0].valueType(), KnobValueType::STRING);
    auto defaultValue = std::get_if<std::string>(&knobs[0].defaultValue());
    ASSERT_NE(defaultValue, nullptr);
    EXPECT_EQ(*defaultValue, "option1");
}

TEST_F(TestGraph, GetKnobsForEngineFailsWhenGraphNotBuilt)
{
    Graph graph;
    createBasicBatchnormGraph(graph);

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(42, knobs);

    // Should fail because graph hasn't been built yet
    EXPECT_FALSE(result.is_good());
}

TEST_F(TestGraph, GetKnobsForEngineHandlesCountMismatch)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillOnce(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting knob count - return 2
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock getting actual knob data - but return different count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* actualCount,
                     void*) {
            *actualCount = 1; // Mismatch!
            return HIPDNN_STATUS_SUCCESS;
        });

    std::vector<Knob> knobs;
    auto result = graph.get_knobs_for_engine(42, knobs);

    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.get_message().find("Mismatch"), std::string::npos);
}

// ============================================================================
// get_knob_lookup_for_engine Tests
// ============================================================================

TEST_F(TestGraph, GetKnobLookupForEngineReturnsMapByKnobId)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillOnce(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    std::string alphaId = "knob_alpha";
    std::string betaId = "knob_beta";

    // Create flatbuffer knobs
    flatbuffers::FlatBufferBuilder builder1;
    auto knob1Offset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder1,
        alphaId.c_str(),
        "Alpha parameter",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder1, 1).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::NONE,
        0,
        false);
    builder1.Finish(knob1Offset);
    auto knob1Buffer = builder1.Release();

    flatbuffers::FlatBufferBuilder builder2;
    auto knob2Offset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder2,
        betaId.c_str(),
        "Beta parameter",
        hipdnn_data_sdk::data_objects::KnobValue::FloatValue,
        hipdnn_data_sdk::data_objects::CreateFloatValue(builder2, 0.1).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::NONE,
        0,
        false);
    builder2.Finish(knob2Offset);
    auto knob2Buffer = builder2.Release();

    // Mock getting knob count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 2;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    2,
                                    _,
                                    NotNull()))
        .WillOnce([&knob1Buffer, &knob2Buffer](hipdnnBackendDescriptor_t,
                                               hipdnnBackendAttributeName_t,
                                               hipdnnBackendAttributeType_t,
                                               int64_t,
                                               int64_t* actualCount,
                                               void* arrayOfElements) {
            *actualCount = 2;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knob1Buffer.data();
            fbDataArray[0].size = knob1Buffer.size();
            fbDataArray[1].ptr = knob2Buffer.data();
            fbDataArray[1].size = knob2Buffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    std::unordered_map<KnobType_t, Knob> knobLookup;
    auto result = graph.get_knob_lookup_for_engine(42, knobLookup);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_EQ(knobLookup.size(), 2);

    // Verify knobs are accessible by their IDs

    EXPECT_NE(knobLookup.find(alphaId), knobLookup.end());
    EXPECT_NE(knobLookup.find(betaId), knobLookup.end());

    EXPECT_EQ(knobLookup.at(alphaId).knobId(), alphaId);
    EXPECT_EQ(knobLookup.at(betaId).knobId(), betaId);
}

TEST_F(TestGraph, GetKnobLookupForEngineReturnsEmptyMapWhenNoKnobs)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .WillOnce(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock getting knob count - return 0
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    0,
                                    _,
                                    nullptr))
        .WillOnce([](hipdnnBackendDescriptor_t,
                     hipdnnBackendAttributeName_t,
                     hipdnnBackendAttributeType_t,
                     int64_t,
                     int64_t* elementCount,
                     void*) {
            *elementCount = 0;
            return HIPDNN_STATUS_SUCCESS;
        });

    std::unordered_map<KnobType_t, Knob> knobLookup;
    auto result = graph.get_knob_lookup_for_engine(42, knobLookup);

    EXPECT_TRUE(result.is_good()) << result.get_message();
    EXPECT_TRUE(knobLookup.empty());
}

TEST_F(TestGraph, SetPreferredEngineIdByName)
{
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_STRING_OVERLOAD";

    // Set by name
    graph.set_preferred_engine_id_ext(testEngineName);

    // Verify it was converted to the correct ID
    auto expectedId = hipdnn_data_sdk::utilities::engineNameToId(testEngineName);
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), expectedId);
}

TEST_F(TestGraph, SetPreferredEngineIdByEmptyStringClearsPreference)
{
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_STRING_OVERLOAD";

    // First set a preference
    graph.set_preferred_engine_id_ext(testEngineName);
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());

    // Then clear it with empty string
    graph.set_preferred_engine_id_ext("");

    // Verify no preferred engine ID is set
    EXPECT_FALSE(graph.get_preferred_engine_id_ext().has_value());
}

TEST_F(TestGraph, SetPreferredEngineIdByNameThenById)
{
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_STRING_OVERLOAD";

    // Set by name first
    graph.set_preferred_engine_id_ext(testEngineName);

    // Then override with a different ID
    int64_t overrideId = 999;
    graph.set_preferred_engine_id_ext(std::optional<int64_t>(overrideId));

    // Verify the ID overload took precedence
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), overrideId);
}

TEST_F(TestGraph, SetPreferredEngineIdByIdThenByName)
{
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_STRING_OVERLOAD";
    auto expectedId = hipdnn_data_sdk::utilities::engineNameToId(testEngineName);

    // Set by ID first
    graph.set_preferred_engine_id_ext(std::optional<int64_t>(999));

    // Then override with name
    graph.set_preferred_engine_id_ext(testEngineName);

    // Verify the name overload took precedence
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), expectedId);
}

TEST_F(TestGraph, MethodChaining)
{
    Graph graph;

    const char* testEngineName = "TEST_ENGINE_FOR_CHAINING";
    auto expectedEngineId = hipdnn_data_sdk::utilities::engineNameToId(testEngineName);

    // Test that all setters return reference to self for chaining
    auto& ref1 = graph.set_name("ChainedGraph");
    auto& ref2 = ref1.set_compute_data_type(DataType::FLOAT);
    auto& ref3 = ref2.set_intermediate_data_type(DataType::HALF);
    auto& ref4 = ref3.set_io_data_type(DataType::BFLOAT16);
    auto& ref5 = ref4.set_preferred_engine_id_ext(12345);

    // All references should point to the same object
    EXPECT_EQ(&graph, &ref1);
    EXPECT_EQ(&graph, &ref2);
    EXPECT_EQ(&graph, &ref3);
    EXPECT_EQ(&graph, &ref4);
    EXPECT_EQ(&graph, &ref5);

    // Verify all values were set correctly
    EXPECT_EQ(graph.get_name(), "ChainedGraph");
    EXPECT_EQ(graph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(graph.get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(graph.get_io_data_type(), DataType::BFLOAT16);
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), 12345);

    // Test chaining with string overload
    auto& ref6 = graph.set_preferred_engine_id_ext(testEngineName);
    EXPECT_EQ(&graph, &ref6);
    EXPECT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), expectedEngineId);
}

// Test that create_execution_plan_ext handles deprecated knobs correctly.
// Deprecated knobs should still work (the setting gets passed to the backend)
// but the user receives a warning. This tests the deprecation handling path
// without requiring the complex validation mocking of the removed test.
TEST_F(TestGraph, CreateExecutionPlanExtWithDeprecatedKnob)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock engine descriptor creation
    auto engineDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR, _))
        .Times(2)
        .WillRepeatedly(
            [&engineDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = engineDesc;
                return HIPDNN_STATUS_SUCCESS;
            });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineDesc, HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    int64_t engineId = 42;
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(engineDesc, HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, _))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineDesc))
        .Times(2)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Create a DEPRECATED knob (deprecated = true)
    flatbuffers::FlatBufferBuilder builder;
    auto knobOffset = hipdnn_data_sdk::data_objects::CreateKnobDirect(
        builder,
        "deprecated.feature",
        "This feature is deprecated",
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        hipdnn_data_sdk::data_objects::CreateIntValue(builder, static_cast<int64_t>(0)).Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
        hipdnn_data_sdk::data_objects::CreateIntConstraint(
            builder, static_cast<int64_t>(0), static_cast<int64_t>(1), static_cast<int64_t>(1))
            .Union(),
        true); // deprecated = true
    builder.Finish(knobOffset);
    auto knobBuffer = builder.Release();

    // Mock getting knob count
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
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

    // Mock getting actual knob data
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(engineDesc,
                                    HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    _,
                                    NotNull()))
        .WillOnce([&knobBuffer](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                int64_t,
                                int64_t* actualCount,
                                void* arrayOfElements) {
            *actualCount = 1;
            auto fbDataArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
            fbDataArray[0].ptr = knobBuffer.data();
            fbDataArray[0].size = knobBuffer.size();
            return HIPDNN_STATUS_SUCCESS;
        });

    // Mock engine config descriptor creation
    auto engineConfigDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2345);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&engineConfigDesc](hipdnnBackendDescriptorType_t,
                                      hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = engineConfigDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(
            engineConfigDesc, HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(*_mockBackend, backendFinalize(engineConfigDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock setting knob settings on engine config (deprecated knob still works)
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(engineConfigDesc,
                                    HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE_EXT,
                                    HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                    1,
                                    NotNull()))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Mock execution plan descriptor creation
    auto executionPlanDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9876);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, _))
        .WillOnce([&executionPlanDesc](hipdnnBackendDescriptorType_t,
                                       hipdnnBackendDescriptor_t* descriptor) {
            *descriptor = executionPlanDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    // Set a deprecated knob with a valid value - should succeed with warning
    std::vector<KnobSetting> settings;
    settings.emplace_back("deprecated.feature", static_cast<int64_t>(1));

    auto result = graph.create_execution_plan_ext(engineId, settings);

    // Should succeed (deprecated knobs still work, just log warning)
    EXPECT_TRUE(result.is_good()) << result.get_message();
}

// Test that get_ranked_engine_ids returns an error when the backend fails
// to finalize the heuristic descriptor. This ensures proper error propagation
// from the backend through the frontend API.
TEST_F(TestGraph, GetRankedEngineIdsFailsWhenHeuristicCreationFails)
{
    ::testing::FLAGS_gmock_verbose = "error";
    Graph graph;
    createBasicBatchnormGraph(graph);

    // Mock build_operation_graph
    auto graphDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&graphDesc](hipdnnBackendDescriptor_t* descriptor, const uint8_t*, size_t) {
            *descriptor = graphDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(graphDesc, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(graphDesc)).WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good());

    // Mock heuristic descriptor creation
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
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_CALL(
        *_mockBackend,
        backendSetAttribute(heurDesc, HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    // Make finalize fail for the heuristic descriptor
    EXPECT_CALL(*_mockBackend, backendFinalize(heurDesc))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));

    std::vector<int64_t> rankedEngineIds;
    auto result = graph.get_ranked_engine_ids(rankedEngineIds);

    EXPECT_FALSE(result.is_good());
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_NE(result.get_message().find("Failed to finalize engine heuristic descriptor"),
              std::string::npos);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(TestGraph, MoveConstruction)
{
    Graph originalGraph;
    originalGraph.set_name("OriginalGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    // Move construct
    Graph movedGraph(std::move(originalGraph));

    // Verify moved graph has the original state
    EXPECT_EQ(movedGraph.get_name(), "OriginalGraph");
    EXPECT_EQ(movedGraph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(movedGraph.get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(movedGraph.get_io_data_type(), DataType::FLOAT);
    EXPECT_EQ(originalGraph.get_name(), ""); // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(originalGraph.getTensorsByName().empty());
}

TEST_F(TestGraph, MoveAssignment)
{
    Graph originalGraph;
    originalGraph.set_name("OriginalGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    Graph movedGraph;
    movedGraph.set_name("TargetGraph");

    // Move assign
    movedGraph = std::move(originalGraph);

    // Verify moved graph has the original state
    EXPECT_EQ(movedGraph.get_name(), "OriginalGraph");
    EXPECT_EQ(movedGraph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(movedGraph.get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(movedGraph.get_io_data_type(), DataType::FLOAT);
}

TEST_F(TestGraph, MoveConstructionWithNodes)
{
    Graph originalGraph;
    originalGraph.set_name("GraphWithNodes")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Add a batchnorm node to the graph
    auto y = createBasicBatchnormGraph(originalGraph);
    EXPECT_NE(y, nullptr);

    // Get tensor count before move
    auto tensorsBeforeMove = originalGraph.getTensorsByName();
    size_t tensorCountBeforeMove = tensorsBeforeMove.size();
    EXPECT_GT(tensorCountBeforeMove, 0);

    // Move construct
    Graph movedGraph(std::move(originalGraph));

    // Verify moved graph has the nodes
    auto tensorsAfterMove = movedGraph.getTensorsByName();
    EXPECT_EQ(tensorsAfterMove.size(), tensorCountBeforeMove);

    // Verify graph name was moved
    EXPECT_EQ(movedGraph.get_name(), "SerializedGraphTest");
}

TEST_F(TestGraph, MoveAssignmentWithNodes)
{
    Graph originalGraph;
    originalGraph.set_name("GraphWithNodes")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Add a batchnorm node to the graph
    auto y = createBasicBatchnormGraph(originalGraph);
    EXPECT_NE(y, nullptr);

    // Get tensor count before move
    auto tensorsBeforeMove = originalGraph.getTensorsByName();
    size_t tensorCountBeforeMove = tensorsBeforeMove.size();
    EXPECT_GT(tensorCountBeforeMove, 0);

    Graph movedGraph;
    movedGraph.set_name("TargetGraph");

    // Move assign
    movedGraph = std::move(originalGraph);

    // Verify moved graph has the nodes
    auto tensorsAfterMove = movedGraph.getTensorsByName();
    EXPECT_EQ(tensorsAfterMove.size(), tensorCountBeforeMove);

    // Verify graph name was moved
    EXPECT_EQ(movedGraph.get_name(), "SerializedGraphTest");
}

TEST_F(TestGraph, MoveConstructionWithPreferredEngineId)
{
    Graph originalGraph;
    originalGraph.set_name("GraphWithEngineId")
        .set_compute_data_type(DataType::FLOAT)
        .set_preferred_engine_id_ext(42);

    EXPECT_TRUE(originalGraph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(originalGraph.get_preferred_engine_id_ext().value(), 42);

    // Move construct
    Graph movedGraph(std::move(originalGraph));

    // Verify preferred engine id was moved
    EXPECT_TRUE(movedGraph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(movedGraph.get_preferred_engine_id_ext().value(), 42);
}

TEST_F(TestGraph, MoveAssignmentToEmptyGraph)
{
    Graph sourceGraph;
    sourceGraph.set_name("SourceGraph").set_compute_data_type(DataType::FLOAT);

    Graph targetGraph;
    // Target starts empty
    EXPECT_EQ(targetGraph.get_name(), "");

    // Move assign
    targetGraph = std::move(sourceGraph);

    // Target now has source's state
    EXPECT_EQ(targetGraph.get_name(), "SourceGraph");
    EXPECT_EQ(targetGraph.get_compute_data_type(), DataType::FLOAT);
}

// ── Engine Override Config integration ───────────────────────────────────────

// Helper: build a conv_fprop graph with fixed tensor dims (x={1,3,32,32}, w={64,3,3,3})
// and return (x, w) so callers can use them for matching assertions.
static std::pair<std::shared_ptr<TensorAttributes>, std::shared_ptr<TensorAttributes>>
    buildConvFpropGraph(Graph& graph)
{
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_dim({1, 3, 32, 32})
        .set_stride({3072, 1024, 32, 1})
        .set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2).set_dim({64, 3, 3, 3}).set_stride({27, 9, 3, 1}).set_data_type(DataType::FLOAT);

    ConvFpropAttributes convAttr;
    convAttr.set_name("EngineOverrideConv")
        .set_pre_padding({1, 1})
        .set_post_padding({1, 1})
        .set_stride({1, 1})
        .set_dilation({1, 1});

    graph.conv_fprop(x, w, convAttr);
    return {x, w};
}

// Test 1: an explicitly set engine ID must survive build_operation_graph.
// applyEngineOverride() is gated on !_preferredEngineId.has_value(), so any
// value the user sets before build must be left intact.
TEST_F(TestGraph, EngineOverrideDoesNotReplaceExplicitlySetEngineId)
{
    Graph graph;
    buildConvFpropGraph(graph);

    constexpr int64_t EXPLICIT_ENGINE_ID = 42;
    graph.set_preferred_engine_id_ext(EXPLICIT_ENGINE_ID);

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    EXPECT_TRUE(graph.build_operation_graph(_handle).is_good());

    ASSERT_TRUE(graph.get_preferred_engine_id_ext().has_value());
    EXPECT_EQ(graph.get_preferred_engine_id_ext().value(), EXPLICIT_ENGINE_ID);
}

// Test 2: EngineOverrideConfig::matchOperation identifies conv_fprop tensors
// with the same dims that getPreferredIdFromOverrideConfig() would pass to
// checkEngineOverride() at build time.
TEST_F(TestGraph, EngineOverrideConfigMatchesConvFpropTensors)
{
    using namespace hipdnn_frontend::engine_override;

    // Tensors with the same dims used by buildConvFpropGraph
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 3, 32, 32}).set_stride({3072, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_dim({64, 3, 3, 3}).set_stride({27, 9, 3, 1}).set_data_type(DataType::FLOAT);

    // Exact rule for this shape
    OperationRule exactRule;
    exactRule.op = "conv_fprop";
    exactRule.engineName = hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_NAME;
    exactRule.tensors = {TensorPattern{{1, 3, 32, 32}, {}}, TensorPattern{{64, 3, 3, 3}, {}}};

    EngineOverrideConfig config({std::move(exactRule)});

    auto result = config.matchOperation("conv_fprop", {x, w});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);

    // Wrong op must not match
    EXPECT_FALSE(config.matchOperation("conv_dgrad", {x, w}).has_value());

    // Different batch size must not match (no wildcard in rule)
    auto x8 = std::make_shared<TensorAttributes>();
    x8->set_dim({8, 3, 32, 32}).set_data_type(DataType::FLOAT);
    EXPECT_FALSE(config.matchOperation("conv_fprop", {x8, w}).has_value());
}

// Test 3: loading a JSON config from an in-memory string and matching against
// conv_fprop tensors.  This exercises the full loadFromContent() → matchOperation()
// path with the same shapes that the graph presents during build_operation_graph().
TEST_F(TestGraph, EngineOverrideConfigFromContentMatchesConvFpropGraph)
{
    using namespace hipdnn_frontend::engine_override;

    const int64_t kEngine = hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;

    const std::string kJson = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 32, 32] },
        { "dim": [64, 3, 3, 3] }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(kJson);
    ASSERT_TRUE(config.has_value());

    // Same tensor dims as buildConvFpropGraph
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({1, 3, 32, 32}).set_stride({3072, 1024, 32, 1}).set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_dim({64, 3, 3, 3}).set_stride({27, 9, 3, 1}).set_data_type(DataType::FLOAT);

    auto result = config->matchOperation("conv_fprop", {x, w});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kEngine);

    // A different batch size must not match the exact rule
    auto x8 = std::make_shared<TensorAttributes>();
    x8->set_dim({8, 3, 32, 32}).set_data_type(DataType::FLOAT);
    EXPECT_FALSE(config->matchOperation("conv_fprop", {x8, w}).has_value());
}

TEST_F(TestGraph, SdpaFpropNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 16, 64}).set_stride({8192, 1024, 64, 1}).set_data_type(DataType::FLOAT);

    auto k = std::make_shared<TensorAttributes>();
    k->set_dim({2, 8, 32, 64}).set_stride({16384, 2048, 64, 1}).set_data_type(DataType::FLOAT);

    auto v = std::make_shared<TensorAttributes>();
    v->set_dim({2, 8, 32, 64}).set_stride({16384, 2048, 64, 1}).set_data_type(DataType::FLOAT);

    SdpaAttributes attributes;
    attributes.set_name("SdpaNode");

    auto [o, stats] = graph.sdpa(q, k, v, attributes);

    EXPECT_EQ(o->get_name(), "SdpaNode::O");
    EXPECT_TRUE(o->get_is_virtual());
    EXPECT_EQ(stats, nullptr);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, SdpaFpropNodeCreationWithStats)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 16, 64}).set_stride({8192, 1024, 64, 1}).set_data_type(DataType::FLOAT);

    auto k = std::make_shared<TensorAttributes>();
    k->set_dim({2, 8, 32, 64}).set_stride({16384, 2048, 64, 1}).set_data_type(DataType::FLOAT);

    auto v = std::make_shared<TensorAttributes>();
    v->set_dim({2, 8, 32, 64}).set_stride({16384, 2048, 64, 1}).set_data_type(DataType::FLOAT);

    SdpaAttributes attributes;
    attributes.set_name("SdpaNodeStats");
    attributes.set_generate_stats(true);

    auto [o, stats] = graph.sdpa(q, k, v, attributes);

    EXPECT_EQ(o->get_name(), "SdpaNodeStats::O");
    EXPECT_TRUE(o->get_is_virtual());
    ASSERT_NE(stats, nullptr);
    EXPECT_EQ(stats->get_name(), "SdpaNodeStats::STATS");
    EXPECT_TRUE(stats->get_is_virtual());

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BuildAndSerializeSdpaFpropGraph)
{
    Graph graph;
    graph.set_name("SerializedSdpaGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(1)
        .set_name("Q")
        .set_dim({2, 8, 16, 64})
        .set_stride({8192, 1024, 64, 1})
        .set_data_type(DataType::FLOAT);

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(2)
        .set_name("K")
        .set_dim({2, 8, 32, 64})
        .set_stride({16384, 2048, 64, 1})
        .set_data_type(DataType::FLOAT);

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(3)
        .set_name("V")
        .set_dim({2, 8, 32, 64})
        .set_stride({16384, 2048, 64, 1})
        .set_data_type(DataType::FLOAT);

    SdpaAttributes attributes;
    attributes.set_name("SdpaNode");

    auto [o, stats] = graph.sdpa(q, k, v, attributes);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedSdpaGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    // 4 tensors: Q, K, V, O (no stats since generate_stats not set)
    EXPECT_EQ(deserializedGraph->tensors.size(), 4);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*q, tensorLookup[q->get_uid()]);
    validateTensor(*k, tensorLookup[k->get_uid()]);
    validateTensor(*v, tensorLookup[v->get_uid()]);
    validateTensor(*o, tensorLookup[o->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "SdpaNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::SdpaAttributes);
    auto deserializedSdpaAttributes = deserializedGraph->nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(deserializedSdpaAttributes, nullptr);
    EXPECT_EQ(deserializedSdpaAttributes->q_tensor_uid, q->get_uid());
    EXPECT_EQ(deserializedSdpaAttributes->k_tensor_uid, k->get_uid());
    EXPECT_EQ(deserializedSdpaAttributes->v_tensor_uid, v->get_uid());
    EXPECT_EQ(deserializedSdpaAttributes->o_tensor_uid, o->get_uid());
    EXPECT_FALSE(deserializedSdpaAttributes->stats_tensor_uid.has_value());
}

TEST_F(TestGraph, BuildAndSerializeSdpaFpropGraphWithStats)
{
    Graph graph;
    graph.set_name("SerializedSdpaStatsGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto q = std::make_shared<TensorAttributes>();
    q->set_uid(1)
        .set_name("Q")
        .set_dim({2, 8, 16, 64})
        .set_stride({8192, 1024, 64, 1})
        .set_data_type(DataType::FLOAT);

    auto k = std::make_shared<TensorAttributes>();
    k->set_uid(2)
        .set_name("K")
        .set_dim({2, 8, 32, 64})
        .set_stride({16384, 2048, 64, 1})
        .set_data_type(DataType::FLOAT);

    auto v = std::make_shared<TensorAttributes>();
    v->set_uid(3)
        .set_name("V")
        .set_dim({2, 8, 32, 64})
        .set_stride({16384, 2048, 64, 1})
        .set_data_type(DataType::FLOAT);

    SdpaAttributes attributes;
    attributes.set_name("SdpaStatsNode");
    attributes.set_generate_stats(true);

    auto [o, stats] = graph.sdpa(q, k, v, attributes);
    ASSERT_NE(stats, nullptr);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedSdpaStatsGraph");
    // 5 tensors: Q, K, V, O, STATS
    EXPECT_EQ(deserializedGraph->tensors.size(), 5);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "SdpaStatsNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::SdpaAttributes);
    auto deserializedSdpaAttributes = deserializedGraph->nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(deserializedSdpaAttributes, nullptr);
    EXPECT_EQ(deserializedSdpaAttributes->q_tensor_uid, q->get_uid());
    EXPECT_EQ(deserializedSdpaAttributes->k_tensor_uid, k->get_uid());
    EXPECT_EQ(deserializedSdpaAttributes->v_tensor_uid, v->get_uid());
    EXPECT_EQ(deserializedSdpaAttributes->o_tensor_uid, o->get_uid());
    ASSERT_TRUE(deserializedSdpaAttributes->stats_tensor_uid.has_value());
    EXPECT_EQ(deserializedSdpaAttributes->stats_tensor_uid.value(), stats->get_uid());
    ASSERT_TRUE(deserializedSdpaAttributes->generate_stats.has_value());
    EXPECT_TRUE(deserializedSdpaAttributes->generate_stats.value());
}

TEST_F(TestGraph, CustomOpNodeCreation)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto inputA = std::make_shared<TensorAttributes>();
    inputA->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto inputB = std::make_shared<TensorAttributes>();
    inputB->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    CustomOpAttributes attributes;
    attributes.set_name("MyCustomOp").set_custom_op_id("example.my_add");

    auto outputs = graph.custom_op({inputA, inputB}, 2, attributes);

    EXPECT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0]->get_name(), "MyCustomOp::output_0");
    EXPECT_EQ(outputs[1]->get_name(), "MyCustomOp::output_1");
    EXPECT_TRUE(outputs[0]->get_is_virtual());
    EXPECT_TRUE(outputs[1]->get_is_virtual());

    // Custom ops are opaque — output dims must be set explicitly by the caller
    outputs[0]->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
    outputs[1]->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();
}

TEST_F(TestGraph, BuildAndSerializeCustomOpGraph)
{
    Graph graph;

    graph.set_name("SerializedCustomOpGraph")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::HALF)
        .set_io_data_type(DataType::FLOAT);

    auto inputA = std::make_shared<TensorAttributes>();
    inputA->set_uid(1).set_name("A").set_dim({4, 8}).set_stride({8, 1}).set_data_type(
        DataType::FLOAT);

    auto inputB = std::make_shared<TensorAttributes>();
    inputB->set_uid(2).set_name("B").set_dim({4, 8}).set_stride({8, 1}).set_data_type(
        DataType::FLOAT);

    std::vector<uint8_t> payload = {0xDE, 0xAD};

    CustomOpAttributes attributes;
    attributes.set_name("CustomOpNode").set_custom_op_id("example.my_add").set_data(payload);

    auto outputs = graph.custom_op({inputA, inputB}, 1, attributes);
    outputs[0]->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    auto validationResult = graph.validate();
    EXPECT_TRUE(validationResult.is_good()) << validationResult.get_message();

    std::unique_ptr<hipdnn_data_sdk::data_objects::GraphT> deserializedGraph;
    expectGraphSerializedToBackendDescriptor(deserializedGraph);

    auto buildResult = graph.build_operation_graph(_handle);
    EXPECT_TRUE(buildResult.is_good()) << buildResult.get_message();

    EXPECT_EQ(deserializedGraph->name, "SerializedCustomOpGraph");
    EXPECT_EQ(deserializedGraph->compute_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(deserializedGraph->intermediate_data_type,
              hipdnn_data_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(deserializedGraph->io_data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    // 3 tensors: inputA, inputB, output
    EXPECT_EQ(deserializedGraph->tensors.size(), 3);
    EXPECT_EQ(deserializedGraph->nodes.size(), 1);

    std::unordered_map<int64_t, hipdnn_data_sdk::data_objects::TensorAttributesT> tensorLookup;
    for(auto& tensor : deserializedGraph->tensors)
    {
        tensorLookup[tensor->uid] = *tensor;
    }

    validateTensor(*inputA, tensorLookup[inputA->get_uid()]);
    validateTensor(*inputB, tensorLookup[inputB->get_uid()]);
    validateTensor(*outputs[0], tensorLookup[outputs[0]->get_uid()]);

    EXPECT_EQ(deserializedGraph->nodes[0]->name, "CustomOpNode");
    EXPECT_EQ(deserializedGraph->nodes[0]->attributes.type,
              hipdnn_data_sdk::data_objects::NodeAttributes::CustomOpAttributes);
    auto deserializedCustomOpAttributes
        = deserializedGraph->nodes[0]->attributes.AsCustomOpAttributes();
    ASSERT_NE(deserializedCustomOpAttributes, nullptr);
    EXPECT_EQ(deserializedCustomOpAttributes->custom_op_id, "example.my_add");
    EXPECT_EQ(deserializedCustomOpAttributes->input_tensor_uids.size(), 2u);
    EXPECT_EQ(deserializedCustomOpAttributes->input_tensor_uids[0], inputA->get_uid());
    EXPECT_EQ(deserializedCustomOpAttributes->input_tensor_uids[1], inputB->get_uid());
    EXPECT_EQ(deserializedCustomOpAttributes->output_tensor_uids.size(), 1u);
    EXPECT_EQ(deserializedCustomOpAttributes->output_tensor_uids[0], outputs[0]->get_uid());
    EXPECT_EQ(deserializedCustomOpAttributes->data, payload);
}

TEST_F(TestGraph, CustomOpValidateFailsWithNullInput)
{
    Graph graph;
    graph.set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    auto validInput = std::make_shared<TensorAttributes>();
    validInput->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    CustomOpAttributes attributes;
    attributes.set_name("NullInputOp").set_custom_op_id("example.my_add");

    // Pass a nullptr alongside a valid input
    auto outputs = graph.custom_op({validInput, nullptr}, 1, attributes);
    outputs[0]->set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    // This must return an error, not crash
    auto err = graph.validate();
    EXPECT_FALSE(err.is_good());
}
