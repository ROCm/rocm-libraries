// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

namespace hipblaslt_plugin::test
{

using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;

// Flip the virtual flag of exactly one tensor to an invalid value, to exercise
// the builder's virtual/non-virtual contract checks. The valid graph keeps the
// FP8 inputs, scales, and D non-virtual and the dequant Y outputs virtual.
enum class VirtualOverride
{
    NONE,
    XA_VIRTUAL, // FP8 A input marked virtual (invalid)
    XB_VIRTUAL, // FP8 B input marked virtual (invalid)
    SCALE_A_VIRTUAL, // A scale marked virtual (invalid)
    SCALE_B_VIRTUAL, // B scale marked virtual (invalid)
    YA_NON_VIRTUAL, // A dequant output marked non-virtual (invalid)
    YB_NON_VIRTUAL, // B dequant output marked non-virtual (invalid)
    D_VIRTUAL, // output D marked virtual without an epilogue (invalid)
};

// Corrupt exactly one scale tensor's dtype or shape to exercise the builder's
// scale-tensor checks. The valid graph keeps both scales as FP8_E8M0 (UE8M0)
// with element counts M*(K/blockSize) for A and (K/blockSize)*N for B.
enum class ScaleOverride
{
    NONE,
    A_WRONG_TYPE, // scale_a dtype != FP8_E8M0 (invalid)
    B_WRONG_TYPE, // scale_b dtype != FP8_E8M0 (invalid)
    A_WRONG_SHAPE, // scale_a element count != M*(K/blockSize) (invalid)
    B_WRONG_SHAPE, // scale_b element count != (K/blockSize)*N (invalid)
    A_TRANSPOSED_SHAPE, // scale_a is [K/blockSize, M]: right total, K split on the wrong axis
    B_TRANSPOSED_SHAPE, // scale_b is [N, K/blockSize]: right total, K split on the wrong axis
};

// Build a 3-node FP8 dequant+dequant+matmul flatbuffer graph. Valid by default;
// the override knobs (dims, types, opA/opB, batch, epilogue, VirtualOverride,
// ScaleOverride, ...) each corrupt one aspect to drive the builder's negative tests.
// Dimensions default to a VEC32-compliant shape (M=32, K=128, N=32).
// A is [M, K] col-major (opA=T): strides=[1, M]
// B is [K, N] row-major (opB=N): strides=[N, 1]
// D is [M, N] row-major: strides=[N, 1]
// Scale_A is [M, K/32], Scale_B is [K/32, N]: FP8_E8M0 (UE8M0), one scale per
// 32-element block. The builder verifies the scale dtype and element count
// (M*(K/32) for A, (K/32)*N for B); ScaleOverride corrupts one of those.
// When batch > 1, a leading (contiguous) batch dimension is prepended to every tensor.
// When withEpilogue is true, a 4th Pointwise(RELU_FWD) node consuming D is appended.
// When swapDequantOrder is true, the two dequant nodes are emitted in B-then-A order
// (node 0 produces matmul B, node 1 produces matmul A) to exercise order-independent
// A/B resolution; uids and matmul wiring are unchanged.
inline flatbuffers::FlatBufferBuilder
    createMxMatmulGraph(int64_t m = 32,
                        int64_t k = 128,
                        int64_t n = 32,
                        DT xType = DT::FP8_E4M3,
                        DT dType = DT::HALF,
                        int32_t blockSize = 32,
                        bool opAIsT = true,
                        bool opBIsN = true,
                        int64_t batch = 1,
                        bool withEpilogue = false,
                        bool swapDequantOrder = false,
                        DT matmulComputeType = DT::FLOAT,
                        int64_t dnOverride = 0,
                        VirtualOverride virtualOverride = VirtualOverride::NONE,
                        ScaleOverride scaleOverride = ScaleOverride::NONE)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Prepend a leading contiguous batch dimension when batch > 1.
    auto prependBatch = [batch](std::vector<int64_t>& dims, std::vector<int64_t>& strides) {
        if(batch > 1)
        {
            const int64_t leadStride = dims[0] * dims[1];
            dims.insert(dims.begin(), batch);
            strides.insert(strides.begin(), leadStride);
        }
    };

    int64_t uid = 1;

    // A FP8: [m, k] col-major (stride[-2]==1 → opA=T)
    std::vector<int64_t> aDims = {m, k};
    std::vector<int64_t> aStrides = opAIsT
                                        ? std::vector<int64_t>{1, m} // col-major → opA=T
                                        : std::vector<int64_t>{k, 1}; // row-major → opA=N (invalid)
    prependBatch(aDims, aStrides);
    const int64_t xAUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        xAUid,
        "x_a",
        xType,
        &aStrides,
        &aDims,
        virtualOverride == VirtualOverride::XA_VIRTUAL));

    // Scale_A: [m, k/32]. A_WRONG_SHAPE perturbs the inner dim so the element
    // count no longer equals M*(K/blockSize); A_TRANSPOSED_SHAPE swaps the axes so
    // the count is right but K is split on the wrong axis.
    std::vector<int64_t> scalADims
        = (scaleOverride == ScaleOverride::A_TRANSPOSED_SHAPE)
              ? std::vector<int64_t>{k / blockSize, m}
              : std::vector<int64_t>{m,
                                     (scaleOverride == ScaleOverride::A_WRONG_SHAPE)
                                         ? (k / blockSize) + 1
                                         : k / blockSize};
    std::vector<int64_t> scalAStrides = {scalADims[1], 1};
    prependBatch(scalADims, scalAStrides);
    const DT scaleAType = (scaleOverride == ScaleOverride::A_WRONG_TYPE) ? DT::FLOAT : DT::FP8_E8M0;
    const int64_t scaleAUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleAUid,
        "scale_a",
        scaleAType,
        &scalAStrides,
        &scalADims,
        virtualOverride == VirtualOverride::SCALE_A_VIRTUAL));

    // Virtual Y_A (dequant output A): [m, k] row-major
    std::vector<int64_t> yADims = {m, k};
    std::vector<int64_t> yAStrides = {k, 1};
    prependBatch(yADims, yAStrides);
    const int64_t yAUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        yAUid,
        "y_a",
        DT::FLOAT,
        &yAStrides,
        &yADims,
        virtualOverride != VirtualOverride::YA_NON_VIRTUAL /*virtual*/));

    // B FP8: [k, n] row-major (stride[-1]==1 → opB=N)
    std::vector<int64_t> bDims = {k, n};
    std::vector<int64_t> bStrides = opBIsN
                                        ? std::vector<int64_t>{n, 1} // row-major → opB=N
                                        : std::vector<int64_t>{1, k}; // col-major → opB=T (invalid)
    prependBatch(bDims, bStrides);
    const int64_t xBUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        xBUid,
        "x_b",
        xType,
        &bStrides,
        &bDims,
        virtualOverride == VirtualOverride::XB_VIRTUAL));

    // Scale_B: [k/32, n]. B_WRONG_SHAPE perturbs the outer dim so the element
    // count no longer equals (K/blockSize)*N; B_TRANSPOSED_SHAPE swaps the axes so
    // the count is right but K is split on the wrong axis.
    std::vector<int64_t> scalBDims
        = (scaleOverride == ScaleOverride::B_TRANSPOSED_SHAPE)
              ? std::vector<int64_t>{n, k / blockSize}
              : std::vector<int64_t>{(scaleOverride == ScaleOverride::B_WRONG_SHAPE)
                                         ? (k / blockSize) + 1
                                         : k / blockSize,
                                     n};
    std::vector<int64_t> scalBStrides = {scalBDims[1], 1};
    prependBatch(scalBDims, scalBStrides);
    const DT scaleBType = (scaleOverride == ScaleOverride::B_WRONG_TYPE) ? DT::FLOAT : DT::FP8_E8M0;
    const int64_t scaleBUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleBUid,
        "scale_b",
        scaleBType,
        &scalBStrides,
        &scalBDims,
        virtualOverride == VirtualOverride::SCALE_B_VIRTUAL));

    // Virtual Y_B (dequant output B): [k, n] row-major
    std::vector<int64_t> yBDims = {k, n};
    std::vector<int64_t> yBStrides = {n, 1};
    prependBatch(yBDims, yBStrides);
    const int64_t yBUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        yBUid,
        "y_b",
        DT::FLOAT,
        &yBStrides,
        &yBDims,
        virtualOverride != VirtualOverride::YB_NON_VIRTUAL /*virtual*/));

    // D output: [m, n] row-major. Virtual only when an epilogue node consumes it.
    // dnOverride forces a D N-dim that differs from B's N to exercise the
    // input/output shape-consistency check.
    const int64_t dNDim = (dnOverride != 0) ? dnOverride : n;
    std::vector<int64_t> dDims = {m, dNDim};
    std::vector<int64_t> dStrides = {dNDim, 1};
    prependBatch(dDims, dStrides);
    const bool dVirtual = withEpilogue || virtualOverride == VirtualOverride::D_VIRTUAL;
    const int64_t dUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, dUid, "d", dType, &dStrides, &dDims, dVirtual /*virtual*/));

    int64_t epilogueOutUid = 0;
    if(withEpilogue)
    {
        std::vector<int64_t> eDims = {m, n};
        std::vector<int64_t> eStrides = {n, 1};
        prependBatch(eDims, eStrides);
        epilogueOutUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, epilogueOutUid, "relu_out", dType, &eStrides, &eDims));
    }

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BlockScaleDequantize A and B. The matmul A/B inputs are the dequant Y outputs,
    // so wiring is fixed by uids; only the emission order of the two nodes varies.
    const std::vector<int32_t> blockSizeVec = {blockSize};
    auto deqAttrA
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleDequantizeAttributesDirect(
            builder, xAUid, scaleAUid, yAUid, &blockSizeVec, false);
    auto deqNodeA = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "deq_a",
        DT::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
        deqAttrA.Union());

    auto deqAttrB
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleDequantizeAttributesDirect(
            builder, xBUid, scaleBUid, yBUid, &blockSizeVec, false);
    auto deqNodeB = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "deq_b",
        DT::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
        deqAttrB.Union());

    if(swapDequantOrder)
    {
        nodes.push_back(deqNodeB);
        nodes.push_back(deqNodeA);
    }
    else
    {
        nodes.push_back(deqNodeA);
        nodes.push_back(deqNodeB);
    }

    // Node 2: Matmul (inputs are virtual Y_A and Y_B; output is D)
    auto matmulAttr
        = hipdnn_flatbuffers_sdk::data_objects::CreateMatmulAttributes(builder, yAUid, yBUid, dUid);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "matmul",
        matmulComputeType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes,
        matmulAttr.Union()));

    // Node 3 (optional): Pointwise(RELU_FWD) epilogue consuming D — makes the graph
    // a 4-node graph, which the MX builder must reject (no fused epilogue allowed).
    if(withEpilogue)
    {
        auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
            flatbuffers::nullopt, // relu_lower_clip
            flatbuffers::nullopt, // relu_upper_clip
            flatbuffers::nullopt, // relu_lower_clip_slope
            flatbuffers::nullopt, // axis_tensor_uid
            dUid, // in_0_tensor_uid (D)
            flatbuffers::nullopt, // in_1_tensor_uid
            flatbuffers::nullopt, // in_2_tensor_uid
            epilogueOutUid); // out_0_tensor_uid
        nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
            builder,
            "relu_fwd",
            DT::FLOAT,
            hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
            pointwiseAttributes.Union()));
    }

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder, "mx_matmul_test", DT::FLOAT, DT::HALF, DT::BFLOAT16, &tensorAttributes, &nodes);
    builder.Finish(graphOffset);
    return builder;
}

} // namespace hipblaslt_plugin::test
