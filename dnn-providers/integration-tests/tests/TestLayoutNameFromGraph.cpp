// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/golden/GoldenBundleDiscovery.hpp"

namespace
{

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_integration_tests::golden::layoutNameFromGraph;

const int64_t K_PRIMARY_UID = 1;

struct LayoutStrides
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

const LayoutStrides K_NCHW = {{1, 3, 4, 5}, {60, 20, 5, 1}};
const LayoutStrides K_NHWC = {{1, 3, 4, 5}, {60, 1, 15, 3}};
const LayoutStrides K_NCDHW = {{1, 3, 2, 4, 5}, {120, 40, 20, 5, 1}};
const LayoutStrides K_NDHWC = {{1, 3, 2, 4, 5}, {120, 1, 60, 15, 3}};
const LayoutStrides K_NCL = {{2, 3, 4}, {12, 4, 1}};
const LayoutStrides K_NLC = {{2, 3, 4}, {12, 1, 3}};
const LayoutStrides K_BHSD = {{2, 4, 8, 16}, {512, 128, 16, 1}};
const LayoutStrides K_BSHD = {{2, 4, 8, 16}, {512, 16, 64, 1}};
const LayoutStrides K_2D_BIAS = {{1, 3}, {3, 1}};

const Graph* buildAndGet(flatbuffers::FlatBufferBuilder& fbb,
                         flatbuffers::Offset<Graph> graphOffset)
{
    fbb.Finish(graphOffset);
    return flatbuffers::GetRoot<Graph>(fbb.GetBufferPointer());
}

flatbuffers::Offset<TensorAttributes>
    makeTensor(flatbuffers::FlatBufferBuilder& fbb, int64_t uid, const LayoutStrides& ls)
{
    return CreateTensorAttributesDirect(fbb, uid, nullptr, DataType::FLOAT, &ls.strides, &ls.dims);
}

flatbuffers::Offset<Node> makeNode(flatbuffers::FlatBufferBuilder& fbb,
                                   NodeAttributes attrType,
                                   flatbuffers::Offset<void> attrs)
{
    return CreateNodeDirect(fbb, "node", DataType::FLOAT, attrType, attrs);
}

flatbuffers::Offset<Graph>
    makeGraph(flatbuffers::FlatBufferBuilder& fbb,
              const std::vector<flatbuffers::Offset<TensorAttributes>>& tensors,
              const std::vector<flatbuffers::Offset<Node>>& nodes)
{
    return CreateGraphDirect(
        fbb, "test", DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, &tensors, &nodes);
}

// --- A. Standalone op tests ---

TEST(TestLayoutNameFromGraph, ConvFwdNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, ConvFwdNhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, ConvBwdNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateConvolutionBwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionBwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, ConvWrwNhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateConvolutionWrwAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionWrwAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, BatchnormInferenceNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateBatchnormInferenceAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::BatchnormInferenceAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, BatchnormInferenceVarianceExtNhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateBatchnormInferenceAttributesVarianceExt(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::BatchnormInferenceAttributesVarianceExt, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, BatchnormNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateBatchnormAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::BatchnormAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, BatchnormBwdNhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateBatchnormBackwardAttributes(fbb, 2, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::BatchnormBackwardAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, SdpaBhsd)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_BHSD);
    auto attr = CreateSdpaAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::SdpaAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "bhsd");
}

TEST(TestLayoutNameFromGraph, SdpaBshd)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_BSHD);
    auto attr = CreateSdpaAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::SdpaAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "bshd");
}

TEST(TestLayoutNameFromGraph, SdpaBwdBhsd)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_BHSD);
    auto attr = CreateSdpaBackwardAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::SdpaBackwardAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "bhsd");
}

TEST(TestLayoutNameFromGraph, MatmulNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateMatmulAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::MatmulAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, LayernormNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateLayernormAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::LayernormAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, LayernormBwdNhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateLayernormBackwardAttributes(fbb, 2, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::LayernormBackwardAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, RmsNormNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateRMSNormAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::RMSNormAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, RmsNormBwdNhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateRMSNormBackwardAttributes(fbb, 2, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::RMSNormBackwardAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, ReductionNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateReductionAttributes(fbb, ReductionMode::NOT_SET, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ReductionAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, BlockScaleQuantizeNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateBlockScaleQuantizeAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::BlockScaleQuantizeAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, BlockScaleDequantizeNhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateBlockScaleDequantizeAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::BlockScaleDequantizeAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, ResampleFwdNhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateResampleFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ResampleFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, CustomOpNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto opId = fbb.CreateString("my_custom_op");
    const std::vector<int64_t> inputUids = {K_PRIMARY_UID};
    auto inputVec = fbb.CreateVector(inputUids);
    auto attr = CreateCustomOpAttributes(fbb, opId, inputVec);
    auto n = makeNode(fbb, NodeAttributes::CustomOpAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

// --- B. Fused graph tests ---

TEST(TestLayoutNameFromGraph, PointwiseThenConvUsesConvLayout)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto pwAttr = CreatePointwiseAttributes(fbb);
    auto pwNode = makeNode(fbb, NodeAttributes::PointwiseAttributes, pwAttr.Union());
    auto convAttr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto convNode = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, convAttr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {pwNode, convNode};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

TEST(TestLayoutNameFromGraph, ConvThenPointwiseUsesConvLayout)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto convAttr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto convNode = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, convAttr.Union());
    auto pwAttr = CreatePointwiseAttributes(fbb);
    auto pwNode = makeNode(fbb, NodeAttributes::PointwiseAttributes, pwAttr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {convNode, pwNode};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

TEST(TestLayoutNameFromGraph, PointwiseThenSdpaUsesSdpaLayout)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_BHSD);
    auto pwAttr = CreatePointwiseAttributes(fbb);
    auto pwNode = makeNode(fbb, NodeAttributes::PointwiseAttributes, pwAttr.Union());
    auto sdpaAttr = CreateSdpaAttributes(fbb, K_PRIMARY_UID);
    auto sdpaNode = makeNode(fbb, NodeAttributes::SdpaAttributes, sdpaAttr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {pwNode, sdpaNode};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "bhsd");
}

// --- C. Dimensionality variants ---

TEST(TestLayoutNameFromGraph, ConvFwd3dNcl)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCL);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "ncl");
}

TEST(TestLayoutNameFromGraph, ConvFwd3dNlc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NLC);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nlc");
}

TEST(TestLayoutNameFromGraph, ConvFwd5dNcdhw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCDHW);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "ncdhw");
}

TEST(TestLayoutNameFromGraph, ConvFwd5dNdhwc)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NDHWC);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "ndhwc");
}

// --- D. BHSD vs NCHW disambiguation ---

TEST(TestLayoutNameFromGraph, SdpaWithNchwStridesReturnsBhsd)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto attr = CreateSdpaAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::SdpaAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "bhsd");
}

TEST(TestLayoutNameFromGraph, ConvWithBhsdStridesReturnsNchw)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_BHSD);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nchw");
}

// --- E. Edge cases ---

TEST(TestLayoutNameFromGraph, AllPointwiseReturnsUnknown)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_NCHW);
    auto pwAttr = CreatePointwiseAttributes(fbb);
    auto pwNode = makeNode(fbb, NodeAttributes::PointwiseAttributes, pwAttr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {pwNode};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "unknown");
}

TEST(TestLayoutNameFromGraph, EmptyGraphReturnsUnknown)
{
    flatbuffers::FlatBufferBuilder fbb;
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const std::vector<flatbuffers::Offset<Node>> nodes;
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "unknown");
}

TEST(TestLayoutNameFromGraph, NoTensorsReturnsUnknown)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "unknown");
}

TEST(TestLayoutNameFromGraph, PrimaryTensorTooFewDimsReturnsUnknown)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, K_PRIMARY_UID, K_2D_BIAS);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "unknown");
}

TEST(TestLayoutNameFromGraph, UnmatchedStrideOrderReturnsUnknown)
{
    flatbuffers::FlatBufferBuilder fbb;
    const LayoutStrides weird = {{2, 3, 4, 5}, {6, 12, 1, 2}};
    auto t = makeTensor(fbb, K_PRIMARY_UID, weird);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "unknown");
}

TEST(TestLayoutNameFromGraph, NullGraphReturnsUnknown)
{
    EXPECT_EQ(layoutNameFromGraph(nullptr), "unknown");
}

TEST(TestLayoutNameFromGraph, TensorUidMismatchReturnsUnknown)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto t = makeTensor(fbb, 99, K_NCHW);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {t};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "unknown");
}

TEST(TestLayoutNameFromGraph, MultipleTensorsFindsCorrectUid)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto tBias = makeTensor(fbb, 10, K_2D_BIAS);
    auto tPrimary = makeTensor(fbb, K_PRIMARY_UID, K_NHWC);
    auto attr = CreateConvolutionFwdAttributes(fbb, K_PRIMARY_UID);
    auto n = makeNode(fbb, NodeAttributes::ConvolutionFwdAttributes, attr.Union());
    const std::vector<flatbuffers::Offset<TensorAttributes>> tensors = {tBias, tPrimary};
    const std::vector<flatbuffers::Offset<Node>> nodes = {n};
    auto* g = buildAndGet(fbb, makeGraph(fbb, tensors, nodes));
    EXPECT_EQ(layoutNameFromGraph(g), "nhwc");
}

} // namespace
