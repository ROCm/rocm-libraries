// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNodeVarianceExt.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/BlockScaleDequantizeNode.hpp>
#include <hipdnn_frontend/node/BlockScaleQuantizeNode.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/LayerNormNode.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#include <hipdnn_frontend/node/SdpaFpropNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

// Minimal INode subclass that does NOT override getNodeType(),
// so it inherits the default UNKNOWN return value.
class StubNode : public INode
{
public:
    explicit StubNode(GraphAttributes attrs = GraphAttributes())
        : INode(std::move(attrs))
    {
    }
};

} // namespace

TEST(TestNodeType, INodeDefaultReturnsUnknown)
{
    StubNode node;
    EXPECT_EQ(node.getNodeType(), NodeType::UNKNOWN);
}

TEST(TestNodeType, ConvolutionFpropNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    ConvolutionFpropNode node(ConvFpropAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::CONVOLUTION_FPROP);
}

TEST(TestNodeType, ConvolutionDgradNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    ConvolutionDgradNode node(ConvDgradAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::CONVOLUTION_DGRAD);
}

TEST(TestNodeType, ConvolutionWgradNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    ConvolutionWgradNode node(ConvWgradAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::CONVOLUTION_WGRAD);
}

TEST(TestNodeType, BatchnormNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    BatchnormNode node(BatchnormAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BATCHNORM);
}

TEST(TestNodeType, BatchnormInferenceNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    BatchnormInferenceNode node(BatchnormInferenceAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BATCHNORM_INFERENCE);
}

TEST(TestNodeType, BatchnormBackwardNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    BatchnormBackwardNode node(BatchnormBackwardAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BATCHNORM_BACKWARD);
}

TEST(TestNodeType, BatchnormInferenceNodeVarianceExtReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    BatchnormInferenceNodeVarianceExt node(BatchnormInferenceAttributesVarianceExt{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BATCHNORM_INFERENCE_VARIANCE_EXT);
}

TEST(TestNodeType, PointwiseNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    PointwiseNode node(PointwiseAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::POINTWISE);
}

TEST(TestNodeType, MatmulNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    MatmulNode node(MatmulAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::MATMUL);
}

TEST(TestNodeType, LayerNormNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    LayerNormNode node(LayernormAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::LAYER_NORM);
}

TEST(TestNodeType, RMSNormNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    RMSNormNode node(RMSNormAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::RMS_NORM);
}

TEST(TestNodeType, SdpaFpropNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    SdpaFpropNode node(SdpaAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::SDPA_FPROP);
}

TEST(TestNodeType, BlockScaleQuantizeNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    BlockScaleQuantizeNode node(BlockScaleQuantizeAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BLOCK_SCALE_QUANTIZE);
}

TEST(TestNodeType, BlockScaleDequantizeNodeReturnsCorrectType)
{
    GraphAttributes graphAttrs;
    BlockScaleDequantizeNode node(BlockScaleDequantizeAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BLOCK_SCALE_DEQUANTIZE);
}
