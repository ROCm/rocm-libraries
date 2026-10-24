// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// OperationGraphBuilder rules: UID merging, operation order, graph validation
// and the engine index space.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "CudnnShimV8BuilderFixture.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::Dims;
using hipdnn_shim_test::TensorSpec;
using hipdnn_shim_test::v8fe::DataType_t;
using hipdnn_shim_test::v8fe::DescriptorType_t;
using hipdnn_shim_test::v8fe::PointwiseMode_t;

namespace v8fe = hipdnn_shim_test::v8fe;

using Ops = std::vector<const v8fe::Operation*>;
using TensorTweak = std::function<void(v8fe::TensorBuilder&)>;

const Dims SHAPE = {2, 3, 4, 5};
const Dims SCALAR = {1, 1, 1, 1};
const Dims CONV_Y = {1, 8, 8, 8};

// Engines are told apart through their knob's maximum: KNOB_ID_BASE + engine ID.
constexpr std::int64_t KNOB_ID_BASE = 1000;

class TestCudnnShimV8BuilderGraph : public hipdnn_shim_test::ShimV8BuilderFixture
{
protected:
    static v8fe::Operation pointwise(PointwiseMode_t mode,
                                     const v8fe::Tensor& x,
                                     const v8fe::Tensor* b,
                                     const v8fe::Tensor& y)
    {
        v8fe::OperationBuilder builder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
        builder
            .setpwDesc(v8fe::PointWiseDescBuilder()
                           .setMode(mode)
                           .setComputeType(DataType_t::FLOAT)
                           .build())
            .setxDesc(x)
            .setyDesc(y);
        if(b != nullptr)
        {
            builder.setbDesc(*b);
        }
        return builder.build();
    }

    static v8fe::Operation unary(PointwiseMode_t mode, const v8fe::Tensor& x, const v8fe::Tensor& y)
    {
        return pointwise(mode, x, nullptr, y);
    }

    static v8fe::Operation binary(PointwiseMode_t mode,
                                  const v8fe::Tensor& x,
                                  const v8fe::Tensor& b,
                                  const v8fe::Tensor& y)
    {
        return pointwise(mode, x, &b, y);
    }

    // conv(x:1, w:2) -> y, 3x3 filter with unit padding, stride and dilation.
    static v8fe::Operation convolution(const v8fe::Tensor& y)
    {
        const std::array<std::int64_t, 2> ones = {1, 1};
        const v8fe::ConvDesc conv = v8fe::ConvDescBuilder()
                                        .setComputeType(DataType_t::FLOAT)
                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                        .setSpatialDimCount(2)
                                        .setPrePadding(2, ones.data())
                                        .setPostPadding(2, ones.data())
                                        .setDilation(2, ones.data())
                                        .setSpatialStride(2, ones.data())
                                        .build();
        return v8fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            .setxDesc(tensor(1, {1, 4, 8, 8}))
            .setwDesc(tensor(2, {8, 4, 3, 3}))
            .setyDesc(y)
            .setcDesc(conv)
            .build();
    }

    static v8fe::Operation smallConvolution(bool yVirtual = true)
    {
        return convolution(tensor(3, CONV_Y, yVirtual));
    }

    static v8fe::Tensor tweaked(const TensorSpec& spec, const TensorTweak& tweak)
    {
        v8fe::TensorBuilder builder;
        describe(builder, spec);
        tweak(builder);
        return builder.build();
    }

    // INT32 [3,1,1,1] offsets with UID 9.
    static std::shared_ptr<v8fe::Tensor> raggedOffsets()
    {
        return std::make_shared<v8fe::Tensor>(tensor(9, {3, 1, 1, 1}, false, DataType_t::INT32));
    }

    // A [1,1,1,1] compile-time constant.
    template <typename T>
    static v8fe::Tensor constant(std::int64_t uid, DataType_t dataType, T value)
    {
        TensorSpec spec{uid, SCALAR};
        spec.dataType = dataType;
        return tweaked(spec,
                       [value](v8fe::TensorBuilder& builder) { builder.setConstValue(value); });
    }

    // Each hipDNN graph operation's pointwise mode, or -1 for a non-pointwise one.
    std::vector<int> pointwiseModesInGraphOrder() const
    {
        std::vector<int> modes;
        for(hipdnnBackendDescriptor_t op : graphOps())
        {
            const auto mode = callOn(op, HIPDNN_ATTR_POINTWISE_MODE);
            modes.push_back(mode ? static_cast<int>(mode->value<hipdnnPointwiseMode_t>()) : -1);
        }
        return modes;
    }

    std::vector<std::int64_t> pointwiseOutputsInGraphOrder() const
    {
        std::vector<std::int64_t> uids;
        for(hipdnnBackendDescriptor_t op : graphOps())
        {
            uids.push_back(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT));
        }
        return uids;
    }

    void tagEnginesWithKnobs(const std::vector<std::int64_t>& ids)
    {
        for(const std::int64_t id : ids)
        {
            installEngineKnobs(id, {{"prov.tile_size", 0, 0, KNOB_ID_BASE + id, 1, {}}});
        }
    }

    // The engine ID at dense `index`, read back from the knob tagEnginesWithKnobs gave it.
    static std::optional<std::int64_t> engineIdAt(const v8fe::OperationGraph& graph,
                                                  std::int64_t index)
    {
        v8fe::EngineBuilder builder;
        builder.setOperationGraph(graph).setGlobalEngineIdx(index);
        const v8fe::Engine engine = builder.build();
        const auto& knobs = engine.getFinalizedKnobs();
        if(knobs.size() != 1)
        {
            ADD_FAILURE() << "engine at index " << index << " has " << knobs.size() << " knobs";
            return std::nullopt;
        }
        return knobs.front().getMaxValue() - KNOB_ID_BASE;
    }

    static std::vector<std::optional<std::int64_t>>
        engineIdsByIndex(const v8fe::OperationGraph& graph)
    {
        const std::int64_t count = graph.getEngineCount();
        std::vector<std::optional<std::int64_t>> ids;
        ids.reserve(static_cast<std::size_t>(std::max<std::int64_t>(count, 0)));
        for(std::int64_t index = 0; index < count; ++index)
        {
            ids.push_back(engineIdAt(graph, index));
        }
        return ids;
    }

    static BuildFailure engineFailureAt(const v8fe::OperationGraph& graph, std::int64_t index)
    {
        v8fe::EngineBuilder builder;
        builder.setOperationGraph(graph).setGlobalEngineIdx(index);
        return hipdnn_shim_test::buildFailure(builder);
    }
};

using OptionalIds = std::vector<std::optional<std::int64_t>>;

TEST_F(TestCudnnShimV8BuilderGraph, SharedUidLowersToOneTensor)
{
    // Two tensors with UID 3: the convolution's output and the ReLU's input.
    const v8fe::Operation conv = smallConvolution();
    const v8fe::Operation relu
        = unary(PointwiseMode_t::RELU_FWD, tensor(3, CONV_Y, true), tensor(4, CONV_Y));
    graph({&conv, &relu});

    std::vector<std::int64_t> uids;
    for(const auto& call : callsNamed(HIPDNN_ATTR_TENSOR_UNIQUE_ID))
    {
        uids.push_back(call.value<std::int64_t>());
    }
    std::sort(uids.begin(), uids.end());
    EXPECT_EQ(uids, (std::vector<std::int64_t>{1, 2, 3, 4}));
}

TEST_F(TestCudnnShimV8BuilderGraph, ConflictingUidRejectedWithUid)
{
    constexpr std::int64_t SHARED_UID = 37;
    const TensorTweak same = [](v8fe::TensorBuilder&) {};
    std::shared_ptr<v8fe::Tensor> offsets = raggedOffsets();
    const std::array<std::int64_t, 4> otherStrides = {2, 2, 2, 2};

    struct Conflict
    {
        const char* field;
        TensorTweak first;
        TensorTweak second;
    };
    const std::vector<Conflict> conflicts = {
        {"dimensions",
         same,
         [](v8fe::TensorBuilder& builder) {
             const std::array<std::int64_t, 4> dims = {1, 3, 1, 1};
             const Dims strides = hipdnn_shim_test::packedStrides({1, 3, 1, 1});
             builder.setDim(4, dims.data()).setStride(4, strides.data());
         }},
        {"strides",
         same,
         [&otherStrides](v8fe::TensorBuilder& builder) {
             builder.setStride(4, otherStrides.data());
         }},
        {"data type",
         same,
         [](v8fe::TensorBuilder& builder) { builder.setDataType(DataType_t::HALF); }},
        {"virtualness", same, [](v8fe::TensorBuilder& builder) { builder.setVirtual(true); }},
        {"by-value", same, [](v8fe::TensorBuilder& builder) { builder.setByValue(true); }},
        {"constant value",
         [](v8fe::TensorBuilder& builder) { builder.setConstValue(2.5F); },
         [](v8fe::TensorBuilder& builder) { builder.setConstValue(3.5F); }},
        {"ragged offset",
         same,
         [&offsets](v8fe::TensorBuilder& builder) { builder.setRaggedOffset(offsets); }},
    };

    for(const Conflict& conflict : conflicts)
    {
        SCOPED_TRACE(conflict.field);
        const v8fe::Operation first = binary(PointwiseMode_t::ADD,
                                             tensor(1, SHAPE),
                                             tweaked({SHARED_UID, SCALAR}, conflict.first),
                                             tensor(2, SHAPE));
        const v8fe::Operation second = binary(PointwiseMode_t::ADD,
                                              tensor(3, SHAPE),
                                              tweaked({SHARED_UID, SCALAR}, conflict.second),
                                              tensor(4, SHAPE));

        const std::size_t graphsBefore = created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).size();
        const BuildFailure failure = graphFailure({&first, &second});
        EXPECT_EQ(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).size(), graphsBefore);
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
        const std::string uid = "UID " + std::to_string(SHARED_UID);
        EXPECT_NE(failure.what.find(uid), std::string::npos) << failure.what;
    }
}

TEST_F(TestCudnnShimV8BuilderGraph, OrderedChainKeepsOrder)
{
    const v8fe::Operation conv = smallConvolution();
    const v8fe::Operation add = binary(PointwiseMode_t::ADD,
                                       tensor(3, CONV_Y, true),
                                       tensor(5, {1, 8, 1, 1}),
                                       tensor(6, CONV_Y, true));
    const v8fe::Operation relu
        = unary(PointwiseMode_t::RELU_FWD, tensor(6, CONV_Y, true), tensor(7, CONV_Y));
    graph({&conv, &add, &relu});

    EXPECT_EQ(pointwiseModesInGraphOrder(),
              (std::vector<int>{-1, HIPDNN_POINTWISE_ADD, HIPDNN_POINTWISE_RELU_FWD}));
}

TEST_F(TestCudnnShimV8BuilderGraph, UnorderedChainSortedStably)
{
    const v8fe::Operation conv = smallConvolution();
    const v8fe::Operation add = binary(PointwiseMode_t::ADD,
                                       tensor(3, CONV_Y, true),
                                       tensor(5, {1, 8, 1, 1}),
                                       tensor(6, CONV_Y, true));
    const v8fe::Operation relu
        = unary(PointwiseMode_t::RELU_FWD, tensor(6, CONV_Y, true), tensor(7, CONV_Y));
    // An independent op stays ahead of the chain it was listed before.
    const v8fe::Operation neg = unary(PointwiseMode_t::NEG, tensor(8, SHAPE), tensor(9, SHAPE));
    graph({&neg, &relu, &add, &conv});

    EXPECT_EQ(pointwiseModesInGraphOrder(),
              (std::vector<int>{
                  HIPDNN_POINTWISE_NEG, -1, HIPDNN_POINTWISE_ADD, HIPDNN_POINTWISE_RELU_FWD}));
}

TEST_F(TestCudnnShimV8BuilderGraph, DisconnectedComponentsAccepted)
{
    const v8fe::Operation first = unary(PointwiseMode_t::ABS, tensor(1, SHAPE), tensor(2, SHAPE));
    const v8fe::Operation second = unary(PointwiseMode_t::NEG, tensor(3, SHAPE), tensor(4, SHAPE));
    graph({&second, &first});

    EXPECT_EQ(pointwiseOutputsInGraphOrder(), (std::vector<std::int64_t>{4, 2}));
}

TEST_F(TestCudnnShimV8BuilderGraph, CycleRejected)
{
    const v8fe::Operation first
        = unary(PointwiseMode_t::ABS, tensor(1, SHAPE, true), tensor(2, SHAPE, true));
    const v8fe::Operation second
        = unary(PointwiseMode_t::NEG, tensor(2, SHAPE, true), tensor(1, SHAPE, true));

    const BuildFailure failure = graphFailure({&first, &second});
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("cycle"), std::string::npos) << failure.what;
}

TEST_F(TestCudnnShimV8BuilderGraph, UidWrittenTwiceRejected)
{
    const v8fe::Operation first = unary(PointwiseMode_t::ABS, tensor(1, SHAPE), tensor(5, SHAPE));
    const v8fe::Operation second = unary(PointwiseMode_t::NEG, tensor(2, SHAPE), tensor(5, SHAPE));

    const BuildFailure failure = graphFailure({&first, &second});
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("UID 5"), std::string::npos) << failure.what;
}

TEST_F(TestCudnnShimV8BuilderGraph, AlignmentForwardedOnlyBelowSixteen)
{
    TensorSpec wide{1, SHAPE};
    wide.alignment = 32;
    TensorSpec narrow{2, SHAPE};
    narrow.alignment = 4;
    const v8fe::Operation add
        = binary(PointwiseMode_t::ADD, tensor(wide), tensor(narrow), tensor(3, SHAPE));
    graph({&add});

    const auto alignments = callsNamed(HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT);
    ASSERT_EQ(alignments.size(), 1U);
    EXPECT_EQ(alignments.front().value<std::int64_t>(), 4);
    EXPECT_EQ(alignments.front().descriptor, hipdnnTensor(2));
}

TEST_F(TestCudnnShimV8BuilderGraph, SameUidAlignmentUsesSmallest)
{
    for(const std::array<std::int64_t, 2>& order :
        {std::array<std::int64_t, 2>{8, 4}, std::array<std::int64_t, 2>{4, 8}})
    {
        SCOPED_TRACE(order.front());
        TensorSpec produced{3, CONV_Y, true};
        produced.alignment = order.front();
        TensorSpec consumed{3, CONV_Y, true};
        consumed.alignment = order.back();
        const v8fe::Operation conv = convolution(tensor(produced));
        const v8fe::Operation relu
            = unary(PointwiseMode_t::RELU_FWD, tensor(consumed), tensor(4, CONV_Y));
        const std::size_t before = callsNamed(HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT).size();
        graph({&conv, &relu});

        const auto alignments = callsNamed(HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT);
        ASSERT_EQ(alignments.size(), before + 1);
        EXPECT_EQ(alignments.back().value<std::int64_t>(), 4);
        EXPECT_EQ(alignments.back().descriptor, hipdnnTensor(3));
    }
}

TEST_F(TestCudnnShimV8BuilderGraph, ScalarsLowerAsRuntimePassByValue)
{
    const v8fe::Operation layer
        = v8fe::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
              .setNormalizationMode(v8fe::NormMode_t::LAYER_NORM)
              .setNormFwdPhase(v8fe::NormFwdPhase_t::INFERENCE)
              .setxDesc(tensor(1, {2, 3, 4, 4}))
              .setScaleAndBias(tensor(2, {1, 3, 4, 4}), tensor(3, {1, 3, 4, 4}))
              .setEpsilonTensor(scalar(4))
              .setyDesc(tensor(5, {2, 3, 4, 4}))
              .build();
    graph({&layer});

    const auto byValue = callsNamed(HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT);
    ASSERT_EQ(byValue.size(), 1U);
    EXPECT_EQ(byValue.front().descriptor, hipdnnTensor(4));
}

TEST_F(TestCudnnShimV8BuilderGraph, ConstantTensorLowersAsValue)
{
    const auto addConstant = [](const v8fe::Tensor& constantOperand, std::int64_t firstUid) {
        return binary(PointwiseMode_t::ADD,
                      tensor(firstUid, SHAPE),
                      constantOperand,
                      tensor(firstUid + 1, SHAPE));
    };
    const v8fe::Operation floatAdd = addConstant(constant(10, DataType_t::FLOAT, 2.5F), 1);
    const v8fe::Operation intAdd = addConstant(constant(20, DataType_t::INT32, std::int32_t{7}), 3);
    const v8fe::Operation boolAdd = addConstant(constant(30, DataType_t::BOOLEAN, true), 5);
    graph({&floatAdd, &intAdd, &boolAdd});

    hipdnnBackendDescriptor_t floatConstant = hipdnnTensor(10);
    EXPECT_EQ(callOn(floatConstant, HIPDNN_ATTR_TENSOR_VALUE_EXT)->values<float>(),
              (std::vector<float>{2.5F}));
    EXPECT_EQ(callOn(floatConstant, HIPDNN_ATTR_TENSOR_DATA_TYPE)->value<hipdnnDataType_t>(),
              HIPDNN_DATA_FLOAT);
    EXPECT_EQ(callOn(hipdnnTensor(20), HIPDNN_ATTR_TENSOR_VALUE_EXT)->values<std::int32_t>(),
              (std::vector<std::int32_t>{7}));
    EXPECT_EQ(callOn(hipdnnTensor(30), HIPDNN_ATTR_TENSOR_VALUE_EXT)->values<std::uint8_t>(),
              (std::vector<std::uint8_t>{1}));
    for(const std::int64_t uid : {10, 20, 30})
    {
        SCOPED_TRACE(uid);
        EXPECT_FALSE(callOn(hipdnnTensor(uid), HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT));
        // The value is set before the shape, which would otherwise collapse to {1}.
        EXPECT_EQ(callOn(hipdnnTensor(uid), HIPDNN_ATTR_TENSOR_DIMENSIONS)->values<std::int64_t>(),
                  SCALAR);
    }

    const v8fe::Operation unsupported
        = addConstant(constant(40, DataType_t::INT8, std::int8_t{1}), 7);
    const BuildFailure failure = graphFailure({&unsupported});
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8BuilderGraph, RaggedOffsetLinksAuxiliaryTensor)
{
    std::shared_ptr<v8fe::Tensor> offsets = raggedOffsets();
    const v8fe::Tensor x
        = tweaked({1, SHAPE}, [&offsets](v8fe::TensorBuilder& b) { b.setRaggedOffset(offsets); });
    const v8fe::Operation abs = unary(PointwiseMode_t::ABS, x, tensor(2, SHAPE));
    graph({&abs});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    const auto input = callOn(op, HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT);
    ASSERT_TRUE(input);
    const auto ragged
        = callOn(input->value<hipdnnBackendDescriptor_t>(), HIPDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC);
    ASSERT_TRUE(ragged);
    EXPECT_EQ(ragged->value<hipdnnBackendDescriptor_t>(), hipdnnTensor(9));
}

TEST_F(TestCudnnShimV8BuilderGraph, FrontendRejectionIsBadParam)
{
    // ADD whose second input does not broadcast to the output.
    const v8fe::Operation add
        = binary(PointwiseMode_t::ADD, tensor(1, SHAPE), tensor(2, {2, 7, 4, 5}), tensor(3, SHAPE));
    ASSERT_EQ(add.get_status(), CUDNN_STATUS_SUCCESS);

    const BuildFailure failure = graphFailure({&add});
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderGraph, MissingHandleRejected)
{
    const v8fe::Operation relu
        = unary(PointwiseMode_t::RELU_FWD, tensor(1, SHAPE), tensor(2, SHAPE));
    Ops ops = {&relu};
    v8fe::OperationGraphBuilder builder;
    builder.setOperationGraph(static_cast<std::int64_t>(ops.size()), ops.data());

    const BuildFailure failure = hipdnn_shim_test::buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderGraph, EmptyOpsRejected)
{
    v8fe::OperationGraphBuilder unset;
    unset.setHandle(_handle);
    const BuildFailure unsetFailure = hipdnn_shim_test::buildFailure(unset);
    ASSERT_TRUE(unsetFailure.thrown);
    EXPECT_EQ(unsetFailure.status, CUDNN_STATUS_BAD_PARAM);

    const BuildFailure emptyFailure = graphFailure({});
    ASSERT_TRUE(emptyFailure.thrown);
    EXPECT_EQ(emptyFailure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderGraph, AddOperationRejectsNonOperation)
{
    const v8fe::Tensor notAnOperation = tensor(1, SHAPE);
    v8fe::OperationGraphBuilder builder;
    builder.setHandle(_handle).addOperation(notAnOperation.get_desc());

    const BuildFailure failure = hipdnn_shim_test::buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8BuilderGraph, OpsSetReplacesEarlierOps)
{
    const v8fe::Operation abs = unary(PointwiseMode_t::ABS, tensor(1, SHAPE), tensor(2, SHAPE));
    const v8fe::Operation neg = unary(PointwiseMode_t::NEG, tensor(3, SHAPE), tensor(4, SHAPE));
    Ops replaced = {&neg};
    v8fe::OperationGraphBuilder builder;
    describe(builder, {&abs, &neg})
        .setOperationGraph(static_cast<std::int64_t>(replaced.size()), replaced.data());
    const v8fe::OperationGraph built = builder.build();

    EXPECT_EQ(built.getOpCount(), 1U);
    EXPECT_EQ(pointwiseModesInGraphOrder(), (std::vector<int>{HIPDNN_POINTWISE_NEG}));
}

TEST_F(TestCudnnShimV8BuilderGraph, ShapeFlagsNotSupported)
{
    const v8fe::Operation relu
        = unary(PointwiseMode_t::RELU_FWD, tensor(1, SHAPE), tensor(2, SHAPE));
    using Flag = v8fe::OperationGraphBuilder& (v8fe::OperationGraphBuilder::*)(bool);
    const std::array<std::pair<const char*, Flag>, 2> flags = {{
        {"dynamic shape", &v8fe::OperationGraphBuilder::setIsDynamicShapeEnabled},
        {"override shape", &v8fe::OperationGraphBuilder::setIsOverrideShapeEnabled},
    }};
    for(const auto& [name, flag] : flags)
    {
        SCOPED_TRACE(name);
        v8fe::OperationGraphBuilder accepted;
        (describe(accepted, {&relu}).*flag)(false);
        const v8fe::OperationGraph built = accepted.build();
        EXPECT_EQ(built.get_status(), CUDNN_STATUS_SUCCESS);

        v8fe::OperationGraphBuilder rejected;
        (describe(rejected, {&relu}).*flag)(true);
        const BuildFailure failure = hipdnn_shim_test::buildFailure(rejected);
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderGraph, FinalizeRunsNoHeuristics)
{
    installHeuristicResults({-2});
    const v8fe::Operation conv = smallConvolution(false);
    graph({&conv});

    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).empty());
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).empty());
    const auto names = getAttributeNames();
    EXPECT_EQ(std::count(names.begin(), names.end(), HIPDNN_ATTR_ENGINEHEUR_RESULTS), 0);
    hipdnnBackendDescriptor_t hipdnnGraph = onlyCreated(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    const auto finalizedDescriptors = finalized();
    EXPECT_NE(std::find(finalizedDescriptors.begin(), finalizedDescriptors.end(), hipdnnGraph),
              finalizedDescriptors.end());
}

TEST_F(TestCudnnShimV8BuilderGraph, EngineIndexFollowsSignedId)
{
    const std::vector<std::int64_t> ranked = {5, -2, 300, -40};
    installHeuristicResults(ranked);
    tagEnginesWithKnobs(ranked);
    const v8fe::Operation conv = smallConvolution(false);
    const v8fe::OperationGraph built = graph({&conv});

    EXPECT_EQ(built.getEngineCount(), 4);
    EXPECT_EQ(engineIdsByIndex(built), (OptionalIds{-40, -2, 5, 300}));
    const BuildFailure pastEnd = engineFailureAt(built, 4);
    ASSERT_TRUE(pastEnd.thrown);
    EXPECT_EQ(pastEnd.status, CUDNN_STATUS_BAD_PARAM);

    hipdnnBackendDescriptor_t heuristic = onlyCreated(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    EXPECT_EQ(callOn(heuristic, HIPDNN_ATTR_ENGINEHEUR_MODE)->value<hipdnnBackendHeurMode_t>(),
              HIPDNN_HEUR_MODE_FALLBACK);
    EXPECT_EQ(callOn(heuristic, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH)
                  ->value<hipdnnBackendDescriptor_t>(),
              onlyCreated(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR));
}

TEST_F(TestCudnnShimV8BuilderGraph, EnginesEnumeratedOnceAcrossThreads)
{
    using ::testing::_;
    installHeuristicResults({3, 1, 2});
    const v8fe::Operation conv = smallConvolution(false);
    const v8fe::OperationGraph built = graph({&conv});

    // A slow heuristic create keeps the first sweep open while the others arrive.
    ON_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillByDefault([this](hipdnnBackendDescriptorType_t type, hipdnnBackendDescriptor_t* out) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            *out = newMockDescriptor(type);
            return HIPDNN_STATUS_SUCCESS;
        });

    constexpr std::size_t THREAD_COUNT = 8;
    std::atomic<bool> start{false};
    std::array<std::int64_t, THREAD_COUNT> counts{};
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(std::size_t i = 0; i < THREAD_COUNT; ++i)
    {
        threads.emplace_back([&, i] {
            while(!start.load())
            {
                std::this_thread::yield();
            }
            counts.at(i) = built.getEngineCount();
        });
    }
    start.store(true);
    for(std::thread& thread : threads)
    {
        thread.join();
    }

    for(const std::int64_t count : counts)
    {
        EXPECT_EQ(count, 3);
    }
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).size(), 1U);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), 3U);
}

TEST_F(TestCudnnShimV8BuilderGraph, FailingConfigDropsEngine)
{
    installHeuristicResults({7, -1, 4}, {-1});
    tagEnginesWithKnobs({7, -1, 4});
    const v8fe::Operation conv = smallConvolution(false);
    const v8fe::OperationGraph built = graph({&conv});

    EXPECT_EQ(built.getEngineCount(), 2);
    EXPECT_EQ(engineIdsByIndex(built), (OptionalIds{4, 7}));
    const BuildFailure pastEnd = engineFailureAt(built, 2);
    EXPECT_TRUE(pastEnd.thrown);
    EXPECT_EQ(pastEnd.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderGraph, NoApplicableEngineCountsZero)
{
    const v8fe::Operation conv = smallConvolution(false);
    const v8fe::OperationGraph built = graph({&conv});

    EXPECT_EQ(built.getEngineCount(), 0);
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).empty());
    const BuildFailure failure = engineFailureAt(built, 0);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderGraph, FailedEnumerationRetried)
{
    using ::testing::_;
    installHeuristicResults({-2});
    const v8fe::Operation conv = smallConvolution(false);
    const v8fe::OperationGraph built = graph({&conv});

    int attempts = 0;
    ON_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillByDefault(
            [this, &attempts](hipdnnBackendDescriptorType_t type, hipdnnBackendDescriptor_t* out) {
                if(attempts++ == 0)
                {
                    return HIPDNN_STATUS_INTERNAL_ERROR;
                }
                *out = newMockDescriptor(type);
                return HIPDNN_STATUS_SUCCESS;
            });

    const BuildFailure failure = hipdnn_shim_test::failureOf(
        [&built] { [[maybe_unused]] const std::int64_t count = built.getEngineCount(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_INTERNAL_ERROR);
    EXPECT_EQ(built.getEngineCount(), 1);
    EXPECT_EQ(attempts, 2);
}

TEST_F(TestCudnnShimV8BuilderGraph, DuplicateEngineIdKeptOnce)
{
    installHeuristicResults({5, 5, -1});
    tagEnginesWithKnobs({5, -1});
    const v8fe::Operation conv = smallConvolution(false);
    const v8fe::OperationGraph built = graph({&conv});

    EXPECT_EQ(built.getEngineCount(), 2);
    EXPECT_EQ(engineIdsByIndex(built), (OptionalIds{-1, 5}));
}

TEST_F(TestCudnnShimV8BuilderGraph, EngineCountEnumeratesOnce)
{
    installHeuristicResults({-2});
    const v8fe::Operation conv = smallConvolution(false);
    const v8fe::OperationGraph built = graph({&conv});
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).empty());

    for(int query = 0; query < 3; ++query)
    {
        EXPECT_EQ(built.getEngineCount(), 1);
    }
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).size(), 1U);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), 1U);
}

} // namespace
