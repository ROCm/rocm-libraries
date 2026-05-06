// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>

#include <hipdnn_frontend/detail/OperationUnpacker.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNodeVarianceExt.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/BlockScaleDequantizeNode.hpp>
#include <hipdnn_frontend/node/BlockScaleQuantizeNode.hpp>
// #include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/CustomOpNode.hpp>
#include <hipdnn_frontend/node/LayerNormNode.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>
#include <hipdnn_frontend/node/SdpaBwdNode.hpp>
#include <hipdnn_frontend/node/SdpaFwdNode.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

namespace
{

// Concrete INode subclass that does NOT override unpack_from_descriptor.
// Used to verify the default implementation returns an error.
class FakeNodeNoUnpack : public INode
{
public:
    explicit FakeNodeNoUnpack(GraphAttributes attrs = GraphAttributes())
        : INode(std::move(attrs))
    {
    }

    std::string getNodeName() const override
    {
        return "FakeNodeNoUnpack";
    }

    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& /*tensorDescs*/,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& /*operations*/) const override
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Not implemented in test"};
    }
};

} // namespace

// ---------------------------------------------------------------------------
// INode::unpack_from_descriptor default implementation tests
// ---------------------------------------------------------------------------

TEST(TestOperationUnpacker, DefaultUnpackFromDescriptorReturnsError)
{
    FakeNodeNoUnpack node;
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;

    auto err = node.unpack_from_descriptor(nullptr, tensorMap);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("not implemented") != std::string::npos);
    EXPECT_TRUE(err.get_message().find("FakeNodeNoUnpack") != std::string::npos);
}

TEST(TestOperationUnpacker, DefaultUnpackFromDescriptorIncludesNodeName)
{
    FakeNodeNoUnpack node;
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;

    auto err = node.unpack_from_descriptor(nullptr, tensorMap);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("FakeNodeNoUnpack") != std::string::npos);
}

// ---------------------------------------------------------------------------
// queryOperationType tests
// ---------------------------------------------------------------------------

class TestQueryOperationType : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestQueryOperationType, ReturnsErrorWhenQueryFails)
{
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_OPERATION_TYPE_EXT, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_NOT_SUPPORTED));

    // Need to also expect getLastErrorString to be called
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    hipdnnBackendDescriptor_t desc = nullptr;
    auto [result, err] = queryOperationType(desc);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
    EXPECT_EQ(result, HIPDNN_OPERATION_TYPE_NOT_SET_EXT);
}

TEST_F(TestQueryOperationType, PassesThroughUnknownOperationType)
{
    auto unknownType = static_cast<hipdnnOperationType_ext_t>(999);

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([unknownType](hipdnnBackendDescriptor_t,
                                             hipdnnBackendAttributeName_t,
                                             hipdnnBackendAttributeType_t,
                                             int64_t,
                                             int64_t*,
                                             void* arrayOfElements) {
                            std::memcpy(
                                arrayOfElements, &unknownType, sizeof(hipdnnOperationType_ext_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    hipdnnBackendDescriptor_t desc = nullptr;
    auto [result, err] = queryOperationType(desc);
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(result, unknownType);
}

// ---------------------------------------------------------------------------
// unpackOperation tests
// ---------------------------------------------------------------------------

class TestUnpackOperation : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestUnpackOperation, FailsWhenTypeCannotBeDetermined)
{
    // Query fails
    EXPECT_CALL(*_mockBackend, backendGetAttribute(_, HIPDNN_ATTR_OPERATION_TYPE_EXT, _, _, _, _))
        .WillOnce(Return(HIPDNN_STATUS_NOT_SUPPORTED));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    const GraphAttributes graphAttrs;
    hipdnnBackendDescriptor_t desc = nullptr;

    auto [node, err] = unpackOperation(desc, tensorMap, graphAttrs);
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("Failed to determine operation type") != std::string::npos);
}

TEST_F(TestUnpackOperation, FailsForUnsupportedOperationType)
{
    auto unknownType = static_cast<hipdnnOperationType_ext_t>(999);

    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([unknownType](hipdnnBackendDescriptor_t,
                                             hipdnnBackendAttributeName_t,
                                             hipdnnBackendAttributeType_t,
                                             int64_t,
                                             int64_t*,
                                             void* arrayOfElements) {
                            std::memcpy(
                                arrayOfElements, &unknownType, sizeof(hipdnnOperationType_ext_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    const GraphAttributes graphAttrs;
    hipdnnBackendDescriptor_t desc = nullptr;

    auto [node, err] = unpackOperation(desc, tensorMap, graphAttrs);
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_TRUE(err.get_message().find("Unsupported operation type") != std::string::npos);
    EXPECT_TRUE(err.get_message().find("999") != std::string::npos)
        << "Error should include the unsupported type id, got: " << err.get_message();
}

TEST_F(TestUnpackOperation, FailsImmediatelyOnUnpackError)
{
    // Type query succeeds with CONV_FORWARD
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(
                    _, HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, _, _))
        .WillOnce(DoAll(SetArgPointee<4>(int64_t{1}),
                        Invoke([](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  int64_t*,
                                  void* arrayOfElements) {
                            auto value = HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT;
                            std::memcpy(arrayOfElements, &value, sizeof(hipdnnOperationType_ext_t));
                        }),
                        Return(HIPDNN_STATUS_SUCCESS)));

    // Unpacking the ConvFwd operation fails (getDescriptorAttrDesc for X tensor fails)
    EXPECT_CALL(*_mockBackend,
                backendGetAttribute(_,
                                    HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                    HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                    1,
                                    _,
                                    _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    const GraphAttributes graphAttrs;
    hipdnnBackendDescriptor_t desc = nullptr;

    auto [node, err] = unpackOperation(desc, tensorMap, graphAttrs);
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_FALSE(err.get_message().empty());
}

// ---------------------------------------------------------------------------
// createNodeForType tests
// ---------------------------------------------------------------------------

TEST(TestCreateNodeForType, CreatesBatchnormNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_BATCHNORM_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<BatchnormNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesBatchnormBackwardNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_BATCHNORM_BACKWARD_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<BatchnormBackwardNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesBatchnormInferenceNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<BatchnormInferenceNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesBatchnormInferenceNodeVarianceExt)
{
    const GraphAttributes graphAttrs;
    auto [node, err]
        = createNodeForType(HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_VARIANCE_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<BatchnormInferenceNodeVarianceExt>(node);
    EXPECT_NE(typedNode, nullptr);
}

// TEST(TestCreateNodeForType, CreatesBlockScaleDequantizeNode)
// {
//     const GraphAttributes graphAttrs;
//     auto [node, err]
//         = createNodeForType(HIPDNN_OPERATION_TYPE_BLOCK_SCALE_DEQUANTIZE_EXT, graphAttrs);
//     EXPECT_EQ(err.code, ErrorCode::OK);
//     ASSERT_NE(node, nullptr);
//     auto typedNode = std::dynamic_pointer_cast<BlockScaleDequantizeNode>(node);
//     EXPECT_NE(typedNode, nullptr);
// }

// TEST(TestCreateNodeForType, CreatesBlockScaleQuantizeNode)
// {
//     const GraphAttributes graphAttrs;
//     auto [node, err]
//         = createNodeForType(HIPDNN_OPERATION_TYPE_BLOCK_SCALE_QUANTIZE_EXT, graphAttrs);
//     EXPECT_EQ(err.code, ErrorCode::OK);
//     ASSERT_NE(node, nullptr);
//     auto typedNode = std::dynamic_pointer_cast<BlockScaleQuantizeNode>(node);
//     EXPECT_NE(typedNode, nullptr);
// }

// TEST(TestCreateNodeForType, CreatesConvolutionDgradNode)
// {
//     const GraphAttributes graphAttrs;
//     auto [node, err]
//         = createNodeForType(HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_DATA_EXT, graphAttrs);
//     EXPECT_EQ(err.code, ErrorCode::OK);
//     ASSERT_NE(node, nullptr);
//     auto typedNode = std::dynamic_pointer_cast<ConvolutionDgradNode>(node);
//     EXPECT_NE(typedNode, nullptr);
// }

TEST(TestCreateNodeForType, CreatesConvFpropNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_CONVOLUTION_FORWARD_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto convNode = std::dynamic_pointer_cast<ConvolutionFpropNode>(node);
    EXPECT_NE(convNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesBlockScaleQuantizeNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err]
        = createNodeForType(HIPDNN_OPERATION_TYPE_BLOCK_SCALE_QUANTIZE_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto bsqNode = std::dynamic_pointer_cast<BlockScaleQuantizeNode>(node);
    EXPECT_NE(bsqNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesConvWgradNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err]
        = createNodeForType(HIPDNN_OPERATION_TYPE_CONVOLUTION_BACKWARD_WEIGHTS_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto wgradNode = std::dynamic_pointer_cast<ConvolutionWgradNode>(node);
    EXPECT_NE(wgradNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesCustomOpNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_CUSTOM_OP_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<CustomOpNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesLayerNormNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_LAYERNORM_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<LayerNormNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesMatmulNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_MATMUL_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<MatmulNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesPointwiseNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_POINTWISE_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<PointwiseNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesRMSNormNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_RMSNORM_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<RMSNormNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesSdpaBwdNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_SDPA_BACKWARD_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto sdpaNode = std::dynamic_pointer_cast<SdpaBwdNode>(node);
    EXPECT_NE(sdpaNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesSdpaFwdNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_SDPA_FORWARD_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<SdpaFwdNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesBlockScaleDequantizeNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err]
        = createNodeForType(HIPDNN_OPERATION_TYPE_BLOCK_SCALE_DEQUANTIZE_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto bsdNode = std::dynamic_pointer_cast<BlockScaleDequantizeNode>(node);
    EXPECT_NE(bsdNode, nullptr);
}

TEST(TestCreateNodeForType, CreatesReductionNode)
{
    const GraphAttributes graphAttrs;
    auto [node, err] = createNodeForType(HIPDNN_OPERATION_TYPE_REDUCTION_EXT, graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    ASSERT_NE(node, nullptr);
    auto typedNode = std::dynamic_pointer_cast<ReductionNode>(node);
    EXPECT_NE(typedNode, nullptr);
}

TEST(TestCreateNodeForType, ReturnsErrorForUnsupportedType)
{
    const GraphAttributes graphAttrs;
    auto unknownType = static_cast<hipdnnOperationType_ext_t>(999);
    auto [node, err] = createNodeForType(unknownType, graphAttrs);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(node, nullptr);
    EXPECT_TRUE(err.get_message().find("Unsupported operation type") != std::string::npos);
    EXPECT_TRUE(err.get_message().find("999") != std::string::npos)
        << "Error should include the unsupported type id, got: " << err.get_message();
}
